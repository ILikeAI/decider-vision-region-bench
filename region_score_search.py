"""Score four zoomed candidate regions in one batched forward pass per search level."""

import json
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "region_results" / "parallel_region_scores"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "decider"))

from decider.infer import Example, Q
from decider.vision import VisionDecisionModel
from region_search import SIZE, TASKS, mock_screen, quadrant, target_centre

DESCRIPTIONS = {
    "close_window": "the small red circular window-close control in the dark title bar",
    "new_chart": "the blue New Chart button",
    "close_position": "the red Close Position button",
}
DEPTHS = {"close_window": 5, "new_chart": 4, "close_position": 4}


def resize_crop(screen, rect):
    crop = screen.crop(rect)
    return crop.resize((512, 384), Image.Resampling.NEAREST if crop.width < 128 else Image.Resampling.BICUBIC)


def run_task(model, screen, name):
    rect = (0, 0, *SIZE)
    box = TASKS[name]["target_box"]
    goal = target_centre(box)
    steps = []
    for depth in range(1, DEPTHS[name] + 1):
        started = time.perf_counter()
        children = [quadrant(rect, i) for i in range(4)]
        crops = [resize_crop(screen, child) for child in children]
        question = (f"Does this image show {DESCRIPTIONS[name]}? "
                    "Answer yes only if that specific control is visible in this crop.")
        examples = [(crop, Example("Inspect the screenshot crop.", [Q(question, ["yes", "no"])]))
                    for crop in crops]
        inputs = model.prepare(examples)
        with torch.inference_mode():
            logits = model.slot_logits(inputs)
            scores = torch.softmax(logits[:, :2], dim=-1)[:, 0].cpu().tolist()
        elapsed = (time.perf_counter() - started) * 1000
        best = max(range(4), key=lambda i: scores[i])
        step = {"pass": depth, "parent_crop": rect, "candidate_boxes": children,
                "yes_scores": [round(x, 4) for x in scores],
                "chosen_quadrant": best, "chosen_crop": children[best],
                "latency_ms": round(elapsed, 2)}
        steps.append(step)
        print(f"{name} pass {depth}: scores {[round(x, 3) for x in scores]}, "
              f"choose {best}, {elapsed:.0f} ms", flush=True)
        rect = children[best]

    click = target_centre(rect)
    hit = box[0] <= click[0] <= box[2] and box[1] <= click[1] <= box[3]
    visual = screen.copy()
    d = ImageDraw.Draw(visual)
    d.rectangle(box, outline="#20c55a", width=4)
    for step in steps:
        r = step["chosen_crop"]
        intersects = not (r[2] <= box[0] or r[0] >= box[2] or r[3] <= box[1] or r[1] >= box[3])
        d.rectangle(r, outline="#f0b823" if intersects else "#f44336", width=3)
    d.ellipse((click[0] - 6, click[1] - 6, click[0] + 6, click[1] + 6),
              fill="#e92b29", outline="white", width=2)
    visual.save(OUT / f"{name}_result.png")
    return {"task": TASKS[name]["instruction"], "target_box": box, "target_centre": goal,
            "predicted_click": click, "click_hits_target": hit,
            "click_error_pixels": round(math.dist(click, goal), 1),
            "total_latency_ms": round(sum(x["latency_ms"] for x in steps), 2), "steps": steps}


def main():
    screen = mock_screen()
    screen.save(OUT / "mock_desktop.png")
    started = time.perf_counter()
    model = VisionDecisionModel(str(ROOT / "model"), grad_ckpt=False).to("mps").eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"Model loaded in {load_seconds:.2f} s", flush=True)
    results = {"model_load_seconds": round(load_seconds, 2), "tasks": {}}
    for name in TASKS:
        results["tasks"][name] = run_task(model, screen, name)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
