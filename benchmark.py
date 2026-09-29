"""Measure local end-to-end and model latency for Mapika/decider-2b-vision."""

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "decider"))

from decider.infer import Example, Q
from decider.vision import VisionDecisionModel


def make_image(size):
    image = Image.new("RGB", size, "#f1f3f5")
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle((w * 0.30, h * 0.25, w * 0.70, h * 0.75), fill="#e53935")
    draw.ellipse((w * 0.76, h * 0.14, w * 0.91, h * 0.34), fill="#1e88e5")
    return image


def summary(values):
    ordered = sorted(values)
    return {
        "median_ms": round(statistics.median(ordered), 2),
        "mean_ms": round(statistics.mean(ordered), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


def run_once(model, image, example):
    torch.mps.synchronize()
    start = time.perf_counter()
    inputs = model.prepare([(image, example)])
    prepared = time.perf_counter()
    with torch.inference_mode():
        logits = model.slot_logits(inputs)
        probabilities = torch.softmax(logits, dim=-1).cpu()
    finished = time.perf_counter()
    return {
        "prepare_ms": (prepared - start) * 1000,
        "model_and_readout_ms": (finished - prepared) * 1000,
        "total_ms": (finished - start) * 1000,
        "probabilities": probabilities[:, :4].tolist(),
        "text_tokens": int(inputs["input_ids"].shape[-1]),
        "pixel_patch_rows": int(inputs["pixel_values"].shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal (MPS) is unavailable")

    started = time.perf_counter()
    model = VisionDecisionModel(str(ROOT / "model"), grad_ckpt=False).to("mps").eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"Model loaded on MPS in {load_seconds:.2f} s", flush=True)

    question = Q("What colour is the large central square?", ["red", "blue", "green", "none of these"])
    five_questions = [
        question,
        Q("Is a blue circle visible?", ["yes", "no"]),
        Q("Where is the square?", ["left", "centre", "right", "not visible"]),
        Q("Is the background mostly light or dark?", ["light", "dark"]),
        Q("Which shape is larger?", ["square", "circle", "same size"]),
    ]
    cases = [
        ("256x240_one_question", (256, 240), [question]),
        ("1000x750_one_question", (1000, 750), [question]),
        ("256x240_five_questions", (256, 240), five_questions),
    ]

    result = {
        "device": "Apple Metal (MPS)",
        "torch": torch.__version__,
        "model_load_seconds": round(load_seconds, 2),
        "warmup_per_case": args.warmup,
        "repetitions_per_case": args.reps,
        "cases": {},
    }
    for name, size, questions in cases:
        image = make_image(size)
        example = Example("Answer the visual questions about this image.", questions)
        first = run_once(model, image, example)
        for _ in range(args.warmup):
            run_once(model, image, example)
        samples = [run_once(model, image, example) for _ in range(args.reps)]
        result["cases"][name] = {
            "image_size": size,
            "questions": len(questions),
            "first_request_ms": round(first["total_ms"], 2),
            "text_tokens": first["text_tokens"],
            "pixel_patch_rows": first["pixel_patch_rows"],
            "prepare": summary([s["prepare_ms"] for s in samples]),
            "model_and_readout": summary([s["model_and_readout_ms"] for s in samples]),
            "end_to_end": summary([s["total_ms"] for s in samples]),
            "sample_probabilities": first["probabilities"],
        }
        print(f"{name}: {result['cases'][name]['end_to_end']}", flush=True)

    result["max_process_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3), 2)
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
