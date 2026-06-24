import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


CLOSE_ENDED_PROMPT = """You are a meticulous video understanding evaluator. Watch the video and answer one multiple-choice question strictly based on the video.

Return only a raw JSON object with this schema:
{{"result": "<the exact full text of one option>", "reason": "<brief evidence-based explanation>"}}

The "result" value must exactly match one option below.

Question:
{question}

Options:
{options}
"""

COUNTERFACTUAL_QA_PROMPT = """
You are a careful video understanding evaluator. You will watch the provided video and answer ONE question.

Important instruction:
- The question below may contain a description that contradicts the actual video content.
- If you find any contradiction, you MUST follow the actual video content.
- Directly point out and correct the contradiction before answering the question.
- Do NOT accommodate, accept, or continue from a false premise in the question.
- If the question is consistent with the video, answer it normally based on the video.

Task:
- Provide a concise answer grounded in the video.
- For counterfactual or contradiction-based questions, explicitly state what is wrong in the question and what actually happens in the video.
- Keep the response focused on the asked intention, causal relation, sequence, or process.

Your output must follow ALL of these rules strictly:
1) Return ONLY a raw JSON object (no markdown code fences, no extra text) with exactly these keys:
   {{"open_ended_result": "<concise answer, correcting any false premise if present>", "open_ended_reason": "<brief evidence-based explanation from the video>"}}
2) Do not say that you cannot answer merely because the question contains a false premise.
3) If there is a contradiction, start the answer by making the correction clear.
4) In the explanation, cite the key visual evidence that supports the correction or answer.

Question:
{question}
"""

DEFAULT_MODEL = "qwen3-vl-8b-instruct"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TASK_TYPE_ORDER = ["FAU", "CIA", "TR", "CR", "CRS", "FI", "GI", "CIT", "PM", "SE", "CEU"]
VALID_TASK_TYPES = set(TASK_TYPE_ORDER)
DEFAULT_TASK_TYPES = TASK_TYPE_ORDER
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_video_index(video_dir):
    index = {}
    for path in Path(video_dir).rglob("*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            index[path.name] = str(path)
            index[path.name.lower()] = str(path)
    return index


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


def build_prompt(item):
    question = item["question"]
    options = item.get("options")
    if options:
        option_text = "\n".join(f"- {option}" for option in options)
        return CLOSE_ENDED_PROMPT.format(question=question, options=option_text), "close_ended"
    return COUNTERFACTUAL_QA_PROMPT.format(question=question), "open_ended"


def normalize_choice(result, options):
    if not isinstance(result, str):
        return None
    result = result.strip()
    for option in options:
        if result == option.strip():
            return option
    for option in options:
        if result.endswith(option.strip()):
            return option
    return None


def get_evaluation(item, model):
    for evaluation in item.get("evaluation", []):
        if isinstance(evaluation, dict) and evaluation.get("model") == model:
            return evaluation
    return None


def write_evaluation(item, model, mode, response):
    if "evaluation" not in item or not isinstance(item["evaluation"], list):
        item["evaluation"] = []

    evaluation = get_evaluation(item, model)
    if evaluation is None:
        evaluation = {"model": model}
        item["evaluation"].append(evaluation)
    else:
        evaluation.clear()
        evaluation["model"] = model

    if mode == "close_ended":
        options = item.get("options", [])
        result = normalize_choice(response.get("result"), options)
        evaluation["result"] = result
        evaluation["reason"] = response.get("reason", "")
    else:
        evaluation["open_ended_result"] = response.get("open_ended_result") or response.get("result", "")
        evaluation["open_ended_reason"] = response.get("open_ended_reason") or response.get("reason", "")


def should_skip(item, model, resume):
    if not resume:
        return False
    evaluation = get_evaluation(item, model)
    if not isinstance(evaluation, dict):
        return False
    if item.get("options"):
        return "result" in evaluation
    return isinstance(evaluation.get("open_ended_result"), str)


def file_to_data_url(video_path):
    mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
    with open(video_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_video_url(video_path, video_name, video_url_template):
    if video_url_template:
        return video_url_template.format(
            video_name=os.path.basename(video_name),
            video_stem=Path(video_name).stem,
            video_path=video_path,
        )
    return file_to_data_url(video_path)


def extract_stream_text(stream):
    answer_parts = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            answer_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    answer_parts.append(part["text"])

    return "".join(answer_parts).strip()


def call_dashscope_qwen(client, model, video_url, prompt, fps, max_retries):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_url}, "fps": fps},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                stream=True,
            )
            text = extract_stream_text(stream)
            if not text:
                raise ValueError("Empty response text")
            return parse_json_object(text)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(10, 2**attempt))
    raise RuntimeError(last_error)


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFG-Bench videos with DashScope Qwen.")
    parser.add_argument("--input", default="./qa_pairs.json", help="Path to the CFG-Bench QA JSON file.")
    parser.add_argument("--video-dir", default="./videos", help="Directory containing benchmark videos.")
    parser.add_argument("--output", default="./outputs/evaluate/qwen3-vl-8b-instruct_evaluation.json", help="Path to save the evaluation JSON file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope Qwen model name.")
    parser.add_argument("--task_type", nargs="+", default=DEFAULT_TASK_TYPES, help="Task types to evaluate. Default: all CFG-Bench task types.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY"),
        help="DashScope API key. Defaults to DASHSCOPE_API_KEY.",
    )
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL), help="DashScope OpenAI-compatible base URL.")
    parser.add_argument("--video-url-template", default=None, help="Optional video URL template with {video_name}, {video_stem}, or {video_path}.")
    parser.add_argument("--fps", type=int, default=4, help="Video sampling FPS passed to DashScope.")
    parser.add_argument("--max-video-mb", type=float, default=100.0, help="Skip videos larger than this size when using data URLs.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for each DashScope request.")
    parser.add_argument("--resume", action="store_true", help="Skip items already evaluated by the same model.")
    args = parser.parse_args()

    task_types = parse_task_types(args.task_type)
    source_data = load_json(args.output if args.resume and os.path.exists(args.output) else args.input)
    data = [item for item in source_data if str(item.get("task_type") or "").strip().upper() in task_types]

    print(f"Loaded {len(source_data)} QA items")
    print(f"Selected {len(data)} QA items for task_type={','.join(ordered_task_types(task_types))}")
    if not data:
        print("No QA items match --task_type. Nothing to evaluate.")
        return

    if not args.api_key:
        raise ValueError("Please set DASHSCOPE_API_KEY or pass --api-key.")

    video_index = build_video_index(args.video_dir)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    for index, item in enumerate(tqdm(data, desc="Evaluating", unit="sample"), start=1):
        if should_skip(item, args.model, args.resume):
            continue

        video_name = item.get("video_name")
        video_path = video_index.get(video_name) or video_index.get(str(video_name).lower())
        if not video_path:
            tqdm.write(f"[{index}/{len(data)}] Missing video: {video_name}")
            continue

        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if not args.video_url_template and size_mb > args.max_video_mb:
            tqdm.write(f"[{index}/{len(data)}] Skip large video: {video_name} ({size_mb:.2f} MB)")
            continue

        try:
            video_url = build_video_url(video_path, str(video_name), args.video_url_template)
            prompt, mode = build_prompt(item)
            response = call_dashscope_qwen(
                client=client,
                model=args.model,
                video_url=video_url,
                prompt=prompt,
                fps=args.fps,
                max_retries=args.max_retries,
            )
            write_evaluation(item, args.model, mode, response)
        except Exception as exc:
            tqdm.write(f"[{index}/{len(data)}] Failed: {item.get('question_key', video_name)} - {exc}")

        save_json(args.output, data)

    save_json(args.output, data)
    print(f"Saved {len(data)} selected QA items to {args.output}")


if __name__ == "__main__":
    main()
