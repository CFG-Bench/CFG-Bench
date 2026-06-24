#!/usr/bin/env python3
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

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

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
TASK_TYPE_ORDER = ["FAU", "CIA", "TR", "CR", "CRS", "FI", "GI", "CIT", "PM", "SE", "CEU"]
VALID_TASK_TYPES = set(TASK_TYPE_ORDER)
DEFAULT_TASK_TYPES = TASK_TYPE_ORDER


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input JSON must be a list")
    return data


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_video_index(video_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in Path(video_dir).rglob("*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            index[path.name] = str(path)
            index[path.name.lower()] = str(path)
    return index


def find_video(video_index: Dict[str, str], video_name: str) -> Optional[str]:
    key = Path(video_name).name.strip()
    return video_index.get(key) or video_index.get(key.lower())


def build_prompt(item: Dict[str, Any]) -> tuple[str, str, List[str]]:
    question = str(item.get("question", ""))
    options = [str(option) for option in item.get("options", [])]
    if options:
        option_text = "\n".join(f"- {option}" for option in options)
        return CLOSE_ENDED_PROMPT.format(question=question, options=option_text), "close_ended", options
    return COUNTERFACTUAL_QA_PROMPT.format(question=question), "open_ended", options


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    text = strip_code_fence(text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"result": text}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else {"result": text}
            except Exception:
                pass
    return {"result": text}


def normalize_close_result(result: Any, options: List[str]) -> str:
    if not isinstance(result, str):
        return ""
    result = result.strip()
    for option in options:
        if result == option.strip():
            return option
    for option in options:
        if option.strip() and option.strip() in result:
            return option
    return result


class QwenMinimalRunner:
    def __init__(self, model_path: str) -> None:
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def generate(self, video_path: str, prompt: str, fps: float) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(Path(video_path).resolve()),
                        "fps": float(fps),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs:
            videos = []
            video_metadata = []
            for video_item in video_inputs:
                if isinstance(video_item, tuple):
                    video, metadata = video_item
                    videos.append(video)
                    video_metadata.append(metadata)
                else:
                    videos.append(video_item)
            video_inputs = videos
            if video_metadata:
                video_kwargs["video_metadata"] = video_metadata
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **(video_kwargs or {}),
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_json_object(response)


def get_evaluation(item: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    for evaluation in item.get("evaluation", []):
        if isinstance(evaluation, dict) and evaluation.get("model") == model:
            return evaluation
    return None


def write_evaluation(item: Dict[str, Any], model: str, mode: str, response: Dict[str, Any], options: List[str]) -> None:
    if not isinstance(item.get("evaluation"), list):
        item["evaluation"] = []

    evaluation = get_evaluation(item, model)
    if evaluation is None:
        evaluation = {"model": model}
        item["evaluation"].append(evaluation)
    else:
        evaluation.clear()
        evaluation["model"] = model

    if mode == "close_ended":
        evaluation["result"] = normalize_close_result(response.get("result"), options)
        evaluation["reason"] = str(response.get("reason", ""))
    else:
        evaluation["open_ended_result"] = str(response.get("open_ended_result") or response.get("result") or "")
        evaluation["open_ended_reason"] = str(response.get("open_ended_reason") or response.get("reason") or "")


def write_error(item: Dict[str, Any], model: str, error: str) -> None:
    if not isinstance(item.get("evaluation"), list):
        item["evaluation"] = []

    evaluation = get_evaluation(item, model)
    if evaluation is None:
        item["evaluation"].append({"model": model, "error": error})
    else:
        evaluation.clear()
        evaluation["model"] = model
        evaluation["error"] = error


def call_qwen(
    runner: QwenMinimalRunner,
    video_path: str,
    prompt: str,
    fps: float,
    max_retries: int,
) -> Dict[str, Any]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return runner.generate(video_path, prompt, fps)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(10, 2**attempt))
    raise RuntimeError(last_error)


def should_skip(item: Dict[str, Any], model: str, resume: bool) -> bool:
    if not resume:
        return False
    evaluation = get_evaluation(item, model)
    if not isinstance(evaluation, dict):
        return False
    if item.get("options"):
        return "result" in evaluation
    return isinstance(evaluation.get("open_ended_result"), str)


def parse_task_types(values: List[str]) -> set[str]:
    task_types = {value.strip().upper() for value in values if value and value.strip()}
    invalid_task_types = sorted(task_types - VALID_TASK_TYPES)
    if invalid_task_types:
        raise ValueError(f"Invalid --task_type value(s): {', '.join(invalid_task_types)}")
    if not task_types:
        raise ValueError("--task_type must contain at least one task type.")
    return task_types


def ordered_task_types(task_types: set[str]) -> List[str]:
    return [task_type for task_type in TASK_TYPE_ORDER if task_type in task_types]


def main() -> None:
    parser = ArgumentParser(description="Minimal Qwen3-VL evaluator for CFG-Bench")
    parser.add_argument("--input", default="./qa_pairs.json", help="input QA JSON")
    parser.add_argument("--output", default="./outputs/Qwen3-VL-8B-Instruct_evaluation.json", help="output QA JSON")
    parser.add_argument("--model", default="./model/Qwen3-VL-8B-Instruct", help="local model path")
    parser.add_argument("--task_type", nargs="+", default=DEFAULT_TASK_TYPES, help="task types to evaluate, e.g. TR CRS")
    parser.add_argument("--video-dir", default="./videos", help="Directory containing benchmark videos.")
    parser.add_argument("--fps", type=float, default=4.0, help="video sampling fps")
    parser.add_argument("--max-retries", type=int, default=3, help="retry count for each Qwen request")
    parser.add_argument("--resume", action="store_true", help="skip items already evaluated by the same model")
    args = parser.parse_args()

    selected_task_types = parse_task_types(args.task_type)
    model_name = Path(args.model.rstrip("/")).name
    source_items = load_json(args.output if args.resume and os.path.exists(args.output) else args.input)
    items = [
        item
        for item in source_items
        if str(item.get("task_type", "")).strip().upper() in selected_task_types
    ]

    print(f"Loaded {len(source_items)} QA items")
    print(f"Selected {len(items)} QA items for task_type={','.join(ordered_task_types(selected_task_types))}")
    if not items:
        print("No QA items match --task_type. Nothing to evaluate.")
        return

    video_index = build_video_index(args.video_dir)
    runner: Optional[QwenMinimalRunner] = None
    updated = False

    for index, qa in enumerate(tqdm(items, desc="Evaluating", unit="qa"), start=1):
        if should_skip(qa, model_name, args.resume):
            continue

        video_path = find_video(video_index, str(qa.get("video_name", "")))
        if not video_path:
            write_error(qa, model_name, f"video not found: {qa.get('video_name')}")
            tqdm.write(f"[{index}/{len(items)}] Missing video: {qa.get('video_name')}")
            updated = True
            save_json(args.output, items)
            continue

        prompt, mode, options = build_prompt(qa)

        try:
            if runner is None:
                runner = QwenMinimalRunner(args.model)
            result = call_qwen(runner, video_path, prompt, args.fps, args.max_retries)
        except Exception as exc:
            write_error(qa, model_name, repr(exc))
            save_json(args.output, items)
            tqdm.write(f"[{index}/{len(items)}] Failed: {qa.get('question_key', qa.get('video_name'))} - {exc}")
            updated = True
            continue

        write_evaluation(qa, model_name, mode, result, options)
        updated = True
        save_json(args.output, items)

    if updated:
        save_json(args.output, items)
        print(f"saved {len(items)} items to {args.output}")
    else:
        print("No new results were written.")


if __name__ == "__main__":
    main()
