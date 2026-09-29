"""Probe the close-window localization failure for determinism and recoveries."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "region_results" / "close_window_probe"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "decider"))

from decider.infer import Example, Q
from decider.vision import VisionDecisionModel
from region_search import TASKS, mock_screen, quadrant, target_centre

OPTIONS = ["top-left", "top-right", "bottom-left", "bottom-right"]
START_RECT = (0, 0, 256, 192)
TARGET_BOX = TASKS["close_window"]["target_box"]
TARGET = target_centre(TARGET_BOX)
SEMANTIC = "the small red circular window-close control in the dark title bar"
VISUAL = "the leftmost red circle in the row of three coloured circles"


def resize_crop(screen, rect):
    crop = screen.crop(rect)
    method = Image.Resampling.NEAREST if crop.width < 128 else Image.Resampling.BICUBIC
    return crop.resize((512, 384), method)


def infer(model, examples):
    torch.mps.synchronize()
    started = time.perf_counter()
    inputs = model.prepare(examples)
    with torch.inference_mode():
        logits = model.slot_logits(inputs)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    torch.mps.synchronize()
    return probs, (time.perf_counter() - started) * 1000


def binary_crop_scores(model, screen, rect, description):
    children = [quadrant(rect, i) for i in range(4)]
    question = (f"Does this image show {description}? Answer yes only if that exact "
                "control is visible in this crop.")
    examples = [
        (resize_crop(screen, child), Example("Inspect the screenshot crop.",
                                             [Q(question, ["yes", "no"])]))
        for child in children
    ]
    probs, latency = infer(model, examples)
    return [row[0] for row in probs], latency


def original_binary_crop_scores(model, screen, rect):
    """Reproduce region_score_search.py wording byte for byte."""
    children = [quadrant(rect, i) for i in range(4)]
    question = (f"Does this image show {SEMANTIC}? "
                "Answer yes only if that specific control is visible in this crop.")
    examples = [
        (resize_crop(screen, child), Example("Inspect the screenshot crop.",
                                             [Q(question, ["yes", "no"])]))
        for child in children
    ]
    probs, latency = infer(model, examples)
    return [row[0] for row in probs], latency


def grid_image(screen, rect, labelled=False):
    image = resize_crop(screen, rect)
    draw = ImageDraw.Draw(image)
    draw.line((256, 0, 256, 384), fill="#f4ce41", width=5)
    draw.line((0, 192, 512, 192), fill="#f4ce41", width=5)
    if labelled:
        labels = [("A", 12, 10), ("B", 268, 10), ("C", 12, 202), ("D", 268, 202)]
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
        for label, x, y in labels:
            draw.rounded_rectangle((x, y, x + 46, y + 42), radius=7, fill="#111827")
            draw.text((x + 12, y + 4), label, font=font, fill="white")
    return image


def grid_choice_scores(model, screen, rect, description, labelled=False):
    image = grid_image(screen, rect, labelled)
    if labelled:
        options = ["A: top-left", "B: top-right", "C: bottom-left", "D: bottom-right"]
        question = (f"Which labelled region contains the center of {description}? "
                    "Choose the label, even when the control overlaps a boundary.")
    else:
        options = OPTIONS
        question = (f"The yellow lines divide this screenshot into four equal quarters. "
                    f"Which quarter contains the center of {description}?")
    probs, latency = infer(model, [(image, Example("Locate the specified control.",
                                                   [Q(question, options)]))])
    return probs[0][:4], latency


def highlighted_candidate_scores(model, screen, rect, description):
    base = resize_crop(screen, rect)
    x0, y0, x1, y1 = rect
    sx, sy = 512 / (x1 - x0), 384 / (y1 - y0)
    examples = []
    for i in range(4):
        child = quadrant(rect, i)
        image = base.copy()
        draw = ImageDraw.Draw(image)
        local = (
            round((child[0] - x0) * sx), round((child[1] - y0) * sy),
            round((child[2] - x0) * sx), round((child[3] - y0) * sy),
        )
        draw.rectangle(local, outline="#ff2d55", width=8)
        question = (f"Is the center of {description} inside the red outlined region? "
                    "Judge the outlined region while using the rest of the screenshot as context.")
        examples.append((image, Example("Inspect the screenshot and red candidate region.",
                                         [Q(question, ["yes", "no"])])))
    probs, latency = infer(model, examples)
    return [row[0] for row in probs], latency


METHODS = {
    "binary_semantic_original": original_binary_crop_scores,
    "binary_visual": lambda m, s, r: binary_crop_scores(m, s, r, VISUAL),
    "grid_semantic": lambda m, s, r: grid_choice_scores(m, s, r, SEMANTIC),
    "grid_visual": lambda m, s, r: grid_choice_scores(m, s, r, VISUAL),
    "labelled_grid_visual": lambda m, s, r: grid_choice_scores(m, s, r, VISUAL, True),
    "highlighted_visual": lambda m, s, r: highlighted_candidate_scores(m, s, r, VISUAL),
}


def expected(rect):
    x0, y0, x1, y1 = rect
    return int(TARGET[0] >= (x0 + x1) // 2) + 2 * int(TARGET[1] >= (y0 + y1) // 2)


def run_path(model, screen, name, method):
    rect = START_RECT
    steps = []
    for level in range(3, 6):
        scores, latency = method(model, screen, rect)
        chosen = max(range(4), key=lambda i: scores[i])
        wanted = expected(rect)
        ordered = sorted(scores, reverse=True)
        steps.append({
            "pass": level,
            "parent": rect,
            "expected": OPTIONS[wanted],
            "chosen": OPTIONS[chosen],
            "correct": chosen == wanted,
            "scores": {OPTIONS[i]: round(scores[i], 6) for i in range(4)},
            "best_score": round(ordered[0], 6),
            "margin": round(ordered[0] - ordered[1], 6),
            "latency_ms": round(latency, 2),
        })
        rect = quadrant(rect, chosen)
    click = target_centre(rect)
    hit = TARGET_BOX[0] <= click[0] <= TARGET_BOX[2] and TARGET_BOX[1] <= click[1] <= TARGET_BOX[3]
    return {
        "steps": steps,
        "predicted_click": click,
        "click_hits_target": hit,
        "click_error_pixels": round(math.dist(click, TARGET), 2),
        "total_latency_ms": round(sum(step["latency_ms"] for step in steps), 2),
    }


def run_adaptive_hybrid(model, screen):
    """Use parallel crop scores while decisive, then preserve context with a grid."""
    rect = (0, 0, 1024, 768)
    steps = []
    grid_mode = False
    for level in range(1, 6):
        crop_scores = None
        crop_latency = 0.0
        ordered = None
        crop_margin = None
        if not grid_mode:
            crop_scores, crop_latency = original_binary_crop_scores(model, screen, rect)
            ordered = sorted(crop_scores, reverse=True)
            crop_margin = ordered[0] - ordered[1]
        if grid_mode or ordered[0] < 0.10 or crop_margin < 0.05:
            scores, grid_latency = grid_choice_scores(model, screen, rect, VISUAL)
            mode = "contextual_grid"
            latency = crop_latency + grid_latency
            grid_mode = True
        else:
            scores = crop_scores
            mode = "parallel_binary_crops"
            latency = crop_latency
        chosen = max(range(4), key=lambda i: scores[i])
        wanted = expected(rect)
        ranked = sorted(scores, reverse=True)
        steps.append({
            "pass": level,
            "mode": mode,
            "parent": rect,
            "expected": OPTIONS[wanted],
            "chosen": OPTIONS[chosen],
            "correct": chosen == wanted,
            "scores": {OPTIONS[i]: round(scores[i], 6) for i in range(4)},
            "best_score": round(ranked[0], 6),
            "margin": round(ranked[0] - ranked[1], 6),
            "binary_probe_best": None if ordered is None else round(ordered[0], 6),
            "binary_probe_margin": None if crop_margin is None else round(crop_margin, 6),
            "latency_ms": round(latency, 2),
        })
        rect = quadrant(rect, chosen)
    click = target_centre(rect)
    hit = TARGET_BOX[0] <= click[0] <= TARGET_BOX[2] and TARGET_BOX[1] <= click[1] <= TARGET_BOX[3]
    return {
        "policy": "Switch permanently to contextual grid when binary best < 0.10 or margin < 0.05",
        "steps": steps,
        "predicted_click": click,
        "click_hits_target": hit,
        "click_error_pixels": round(math.dist(click, TARGET), 2),
        "total_latency_ms": round(sum(step["latency_ms"] for step in steps), 2),
    }


def determinism_probe(model, screen, repetitions):
    all_scores = []
    latencies = []
    for _ in range(repetitions):
        scores, latency = original_binary_crop_scores(model, screen, START_RECT)
        all_scores.append(scores)
        latencies.append(latency)
    per_region = []
    for i, option in enumerate(OPTIONS):
        values = [row[i] for row in all_scores]
        per_region.append({
            "region": option,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "unique_rounded_10dp": len({round(value, 10) for value in values}),
        })
    return {
        "repetitions": repetitions,
        "chosen_regions": [OPTIONS[max(range(4), key=lambda i: row[i])] for row in all_scores],
        "scores": [[round(value, 10) for value in row] for row in all_scores],
        "per_region": per_region,
        "latency_median_ms": round(statistics.median(latencies), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    screen = mock_screen()
    screen.save(OUT / "mock_desktop.png")
    grid_image(screen, START_RECT, labelled=True).save(OUT / "labelled_step_3.png")

    started = time.perf_counter()
    model = VisionDecisionModel(str(ROOT / "model"), grad_ckpt=False).to("mps").eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started

    # Warm the exact failing path before measuring repeatability.
    original_binary_crop_scores(model, screen, START_RECT)
    results = {
        "model_load_seconds": round(load_seconds, 2),
        "start_rect": START_RECT,
        "target_box": TARGET_BOX,
        "target_centre": TARGET,
        "candidate_count_per_level": 4,
        "determinism": determinism_probe(model, screen, args.repetitions),
        "methods": {},
    }
    for name, method in METHODS.items():
        result = run_path(model, screen, name, method)
        results["methods"][name] = result
        print(f"{name}: hit={result['click_hits_target']} click={result['predicted_click']} "
              f"error={result['click_error_pixels']}px latency={result['total_latency_ms']}ms")
        for step in result["steps"]:
            print(f"  pass {step['pass']}: {step['chosen']} expected={step['expected']} "
                  f"best={step['best_score']:.3f} margin={step['margin']:.3f}")

    hybrid = run_adaptive_hybrid(model, screen)
    results["adaptive_hybrid"] = hybrid
    print(f"adaptive_hybrid: hit={hybrid['click_hits_target']} click={hybrid['predicted_click']} "
          f"error={hybrid['click_error_pixels']}px latency={hybrid['total_latency_ms']}ms")
    for step in hybrid["steps"]:
        print(f"  pass {step['pass']}: mode={step['mode']} {step['chosen']} "
              f"expected={step['expected']} best={step['best_score']:.3f} "
              f"margin={step['margin']:.3f}")

    (OUT / args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"determinism choices: {results['determinism']['chosen_regions']}")
    print(f"wrote {OUT / args.output}")


if __name__ == "__main__":
    main()
