import argparse
import csv
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from CFG_JUDGE import CFG_JUDGE


DEFAULT_MODEL = "deepseek-r1"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TASK_TYPE_ORDER = ["FAU", "CIA", "TR", "CR", "CRS", "FI", "GI", "CIT", "PM", "SE", "CEU"]
VALID_TASK_TYPES = set(TASK_TYPE_ORDER)
CLOSE_ENDED_TASK_TYPES = {"FAU", "TR", "CR"}
DEFAULT_TASK_TYPES = TASK_TYPE_ORDER


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def derive_summary_output(output_path):
    output_name = Path(output_path).name
    if output_name.endswith(".json"):
        output_name = output_name[: -len(".json")]
    return str(Path("./outputs/summary") / f"summary_{output_name}.csv")


def save_summary_csv(path, summary):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_type", "n", "accuracy", "avg_correctness", "avg_detailedness", "avg_total_score"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task_type, row in summary.items():
            output_row = {"task_type": task_type, "n": row["n"]}
            if task_type in CLOSE_ENDED_TASK_TYPES:
                output_row["accuracy"] = row["accuracy"]
            else:
                output_row["avg_correctness"] = row["avg_correctness"]
                output_row["avg_detailedness"] = row["avg_detailedness"]
                output_row["avg_total_score"] = row["avg_total_score"]
            writer.writerow(output_row)


def video_key(value):
    return os.path.basename(str(value or "").strip())


def qa_key(item):
    question_key = normalize_text(item.get("question_key"))
    if question_key:
        return question_key
    video_name = normalize_text(item.get("video_name"))
    question = normalize_text(item.get("question"))
    return f"{video_name}||{question}"


def load_correct_answer_mapping(path):
    mapping = {}
    if not path:
        return mapping
    for item in load_json(path):
        if not isinstance(item, dict):
            continue
        correct_answer = item.get("correct_answer")
        if isinstance(correct_answer, str) and correct_answer.strip():
            mapping[qa_key(item)] = correct_answer.strip()
    return mapping


def load_caption_mapping(path):
    mapping = {}

    def add_item(item):
        if not isinstance(item, dict):
            return
        key = video_key(item.get("video_name") or item.get("video_path") or item.get("path"))
        caption = item.get("caption") or item.get("caption_en")
        if isinstance(item.get("annotations"), dict):
            caption = caption or item["annotations"].get("caption")
        if key and isinstance(caption, str) and caption.strip():
            mapping[key] = caption.strip()

    try:
        data = load_json(path)
        if isinstance(data, list):
            for item in data:
                add_item(item)
            return mapping
    except json.JSONDecodeError:
        pass

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                add_item(json.loads(line))
            except json.JSONDecodeError:
                continue
    return mapping


def strip_code_fence(text):
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    elif text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def parse_json_object(text):
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def build_prompt(question, correct_answer, candidate, caption, task_type):
    return (
        CFG_JUDGE.replace("{question}", str(question))
        .replace("{result}", str(candidate))
        .replace("{correct_answer}", str(correct_answer))
        .replace("{caption}", str(caption))
        .replace("{task_type}", str(task_type))
    )


def call_judge(client, model, prompt, temperature, max_retries):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return parse_json_object(completion.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(10, 2**attempt))
    raise RuntimeError(last_error)


def iter_open_ended_evaluations(item, target_model=None):
    evaluations = item.get("evaluation", [])
    if not isinstance(evaluations, list):
        return
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        if target_model and str(evaluation.get("model") or "") != target_model:
            continue
        candidate = evaluation.get("open_ended_result")
        if isinstance(candidate, str) and candidate.strip():
            yield evaluation, candidate.strip()


def iter_close_ended_evaluations(item, target_model=None):
    evaluations = item.get("evaluation", [])
    if not isinstance(evaluations, list):
        return
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        if target_model and str(evaluation.get("model") or "") != target_model:
            continue
        candidate = evaluation.get("result")
        if isinstance(candidate, str) and candidate.strip():
            yield evaluation, candidate.strip()


def is_judged(evaluation):
    judge = evaluation.get("eval")
    if not isinstance(judge, dict):
        return False
    if judge.get("type") == "close_ended" and isinstance(judge.get("is_correct"), bool):
        return True
    return isinstance(judge.get("correctness"), int) or isinstance(judge.get("detailedness"), int)


def error_result(message):
    return {
        "evidence_spans": [],
        "correctness_reasoning": message,
        "detailedness_reasoning": message,
        "correctness": None,
        "detailedness": None,
        "status": "error",
        "error": message,
    }


def normalize_text(value):
    return str(value or "").strip()


def score_close_ended(item, evaluation, candidate, correct_answer_map):
    correct_answer = normalize_text(correct_answer_map.get(qa_key(item)) or item.get("correct_answer"))
    prediction = normalize_text(candidate)
    is_correct = bool(correct_answer and prediction == correct_answer)
    evaluation["eval"] = {
        "type": "close_ended",
        "prediction": prediction,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "accuracy": 1.0 if is_correct else 0.0,
    }
    return is_correct


def parse_task_types(values):
    task_types = {value.strip().upper() for value in values if value and value.strip()}
    invalid_task_types = sorted(task_types - VALID_TASK_TYPES)
    if invalid_task_types:
        raise ValueError(f"Invalid --task_type value(s): {', '.join(invalid_task_types)}")
    if not task_types:
        raise ValueError("--task_type must contain at least one task type.")
    return task_types


def ordered_task_types(task_types):
    return [task_type for task_type in TASK_TYPE_ORDER if task_type in task_types]


def task_type_of(item):
    return normalize_text(item.get("task_type")).upper()


def add_open_metric(summary, task_type, correctness, detailedness):
    row = summary.setdefault(
        task_type,
        {
            "count": 0,
            "correctness_sum": 0.0,
            "detailedness_sum": 0.0,
            "total_score_sum": 0.0,
        },
    )
    row["count"] += 1
    row["correctness_sum"] += correctness
    row["detailedness_sum"] += detailedness
    row["total_score_sum"] += (correctness + detailedness) / 2


def add_close_metric(summary, task_type, is_correct):
    row = summary.setdefault(task_type, {"count": 0, "correct": 0})
    row["count"] += 1
    row["correct"] += int(bool(is_correct))


def summarize_scores(data, task_types):
    summary = {}
    for item in data:
        task_type = task_type_of(item)
        if task_type not in task_types:
            continue
        evaluations = item.get("evaluation", [])
        if not isinstance(evaluations, list):
            continue
        for evaluation in evaluations:
            if not isinstance(evaluation, dict) or not isinstance(evaluation.get("eval"), dict):
                continue
            judge = evaluation["eval"]
            if task_type in CLOSE_ENDED_TASK_TYPES:
                is_correct = judge.get("is_correct")
                if isinstance(is_correct, bool):
                    add_close_metric(summary, task_type, is_correct)
                continue
            correctness = judge.get("correctness")
            detailedness = judge.get("detailedness")
            if not isinstance(correctness, (int, float)) or not isinstance(detailedness, (int, float)):
                continue
            add_open_metric(summary, task_type, float(correctness), float(detailedness))

    results = {}
    for task_type in ordered_task_types(task_types):
        row = summary.get(task_type)
        if task_type in CLOSE_ENDED_TASK_TYPES:
            if not row or row["count"] == 0:
                results[task_type] = {"n": 0, "accuracy": None}
            else:
                results[task_type] = {
                    "n": row["count"],
                    "accuracy": row["correct"] / row["count"],
                }
            continue
        if not row or row["count"] == 0:
            results[task_type] = {
                "n": 0,
                "avg_correctness": None,
                "avg_detailedness": None,
                "avg_total_score": None,
            }
            continue
        results[task_type] = {
            "n": row["count"],
            "avg_correctness": row["correctness_sum"] / row["count"],
            "avg_detailedness": row["detailedness_sum"] / row["count"],
            "avg_total_score": row["total_score_sum"] / row["count"],
        }
    return results


def print_summary(summary):
    print("\n=== Judge Summary by task_type ===")
    for task_type, row in summary.items():
        count = row["n"]
        if count == 0:
            print(f"{task_type}: no judged items")
            continue
        if task_type in CLOSE_ENDED_TASK_TYPES:
            print(f"{task_type}: n={count}, accuracy={row['accuracy'] * 100:.2f}%")
            continue
        parts = [
            f"{task_type}: n={count}",
            f"avg_correctness={row['avg_correctness']:.2f}",
            f"avg_detailedness={row['avg_detailedness']:.2f}",
            f"avg_total_score={row['avg_total_score']:.2f}",
        ]
        print(", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Judge CFG-Bench evaluation results with exact-match and DeepSeek-R1.")
    parser.add_argument("--input", default="./outputs/evaluate/gemini_evaluation.json", help="Input JSON with evaluation results.")
    parser.add_argument("--qa-file", default="./qa_pairs.json", help="Original QA JSON file used to read correct_answer for close-ended tasks.")
    parser.add_argument("--caption-file", default="./caption.json", help="Caption JSON or JSONL file.")
    parser.add_argument("--output", default="./outputs/judge/gemini_evaluation_judge.json", help="Path to save judged results.")
    parser.add_argument("--save-summary", action="store_true", help="Save task_type summary CSV under outputs/summary.")
    parser.add_argument("--summary-output", default=None, help="Optional summary CSV path used only with --save-summary.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL), help="OpenAI-compatible API base URL.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY"),
        help="API key. Defaults to DASHSCOPE_API_KEY.",
    )
    parser.add_argument("--task_type", nargs="+", default=DEFAULT_TASK_TYPES, help="Task types to judge. Default: all CFG-Bench task types.")
    parser.add_argument("--target-model", default=None, help="Only judge answers from this evaluated model.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Judge model temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for each judge request.")
    parser.add_argument("--resume", action="store_true", help="Skip evaluation items that already have judge scores.")
    parser.add_argument("--save-every", type=int, default=1, help="Save after this many judged items.")
    args = parser.parse_args()

    task_types = parse_task_types(args.task_type)
    data = load_json(args.output if args.resume and os.path.exists(args.output) else args.input)
    correct_answer_map = load_correct_answer_mapping(args.qa_file)
    summary_output = args.summary_output or derive_summary_output(args.output)

    close_ended_count = 0
    close_ended_scored = 0
    open_ended_count = 0
    already_judged_count = 0
    tasks = []

    for item in data:
        task_type = task_type_of(item)
        if task_type not in task_types:
            continue
        if task_type in CLOSE_ENDED_TASK_TYPES:
            for evaluation, candidate in iter_close_ended_evaluations(item, target_model=args.target_model):
                close_ended_count += 1
                if args.resume and is_judged(evaluation):
                    already_judged_count += 1
                    continue
                score_close_ended(item, evaluation, candidate, correct_answer_map)
                close_ended_scored += 1
            continue

        for evaluation, candidate in iter_open_ended_evaluations(item, target_model=args.target_model):
            open_ended_count += 1
            if args.resume and is_judged(evaluation):
                already_judged_count += 1
                continue
            tasks.append((item, evaluation, candidate))

    print(f"Loaded {len(data)} QA items from {args.input}")
    print(f"Loaded {len(correct_answer_map)} correct answers from {args.qa_file}")
    print(f"Selected task_type={','.join(ordered_task_types(task_types))}")
    print(f"Found {close_ended_count} close-ended answers")
    print(f"Found {open_ended_count} open-ended answers")
    if args.resume:
        print(f"Skipped {already_judged_count} already judged answers")

    if close_ended_scored:
        save_json(args.output, data)

    if tasks and not args.api_key:
        raise ValueError("Please set DASHSCOPE_API_KEY, or pass --api-key.")

    captions = load_caption_mapping(args.caption_file) if tasks else {}
    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if tasks else None

    judged = 0
    for item, evaluation, candidate in tqdm(tasks, desc="Judging", unit="answer"):
        caption = captions.get(video_key(item.get("video_name")))
        if not caption:
            evaluation["eval"] = error_result(f"No caption found for {item.get('video_name')}")
            continue

        prompt = build_prompt(
            question=item.get("question", ""),
            correct_answer=item.get("correct_answer", ""),
            candidate=candidate,
            caption=caption,
            task_type=item.get("task_type", ""),
        )
        try:
            evaluation["eval"] = call_judge(client, args.model, prompt, args.temperature, args.max_retries)
        except Exception as exc:
            evaluation["eval"] = error_result(str(exc))

        judged += 1
        if args.save_every > 0 and judged % args.save_every == 0:
            save_json(args.output, data)

    save_json(args.output, data)
    summary = summarize_scores(data, task_types)
    print_summary(summary)
    if args.save_summary:
        save_summary_csv(summary_output, summary)
    print(f"\nScored {close_ended_scored} close-ended answers.")
    print(f"Judged {judged} open-ended answers.")
    print(f"Saved results to {args.output}")
    if args.save_summary:
        print(f"Saved summary to {summary_output}")


if __name__ == "__main__":
    main()
