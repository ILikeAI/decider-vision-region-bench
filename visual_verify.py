"""Small, fixed before/after GUI outcome probe for decider-2b-vision.

This uses authored mock screens. It never reads or controls the live desktop.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "verification_results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "decider"))

from decider.infer import Example, Q
from decider.vision import VisionDecisionModel
from region_search import mock_screen, font


OPTIONS_V1 = [
    "The requested outcome is visibly present",
    "A different outcome is visibly present",
    "The interface is still loading or processing",
    "The screenshots do not show enough to tell",
]
OPTIONS_V2 = [
    "Yes, the specified visible result has appeared",
    "No, a different change happened instead",
    "Not yet, a loading or progress state is shown",
    "Cannot determine from these screenshots",
]


def chart_opened(image):
    image = image.copy()
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((206, 112, 1006, 744), radius=10, fill="white", outline="#cbd7e4", width=3)
    d.text((231, 131), "New chart", font=font(28, True), fill="#203249")
    d.text((231, 177), "GBP / USD", font=font(22, True), fill="#203249")
    d.line([(250, 602), (360, 540), (460, 570), (560, 420), (670, 451),
            (770, 341), (889, 292)], fill="#1674d2", width=6)
    return image


def position_closed(image):
    image = image.copy()
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((211, 579, 1003, 742), radius=10, fill="white", outline="#d5dde7", width=2)
    d.text((237, 601), "Open positions", font=font(20, True), fill="#263448")
    d.text((237, 654), "No open positions", font=font(18), fill="#33465a")
    return image


def chart_loading(image):
    image = image.copy()
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((347, 288, 678, 481), radius=16, fill="white", outline="#b7c5d4", width=3)
    d.text((398, 343), "Opening chart...", font=font(23, True), fill="#203249")
    d.arc((472, 394, 526, 448), 15, 285, fill="#1674d2", width=5)
    return image


def window_closed(_image):
    image = Image.new("RGB", (1024, 768), "#638fba")
    d = ImageDraw.Draw(image)
    d.text((25, 20), "Desktop", font=font(19, True), fill="white")
    d.rounded_rectangle((285, 703, 739, 754), radius=17, fill="#b4c9dc")
    return image


def settings_screen(enabled=False):
    image = mock_screen()
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((273, 185, 752, 570), radius=14, fill="white", outline="#b9c8d8", width=3)
    d.text((305, 216), "Settings", font=font(27, True), fill="#203249")
    d.text((305, 315), "Alerts", font=font(21), fill="#203249")
    d.rounded_rectangle((615, 313, 715, 351), radius=18, fill="#1b9b70" if enabled else "#8491a0")
    d.text((632, 320), "On" if enabled else "Off", font=font(17, True), fill="white")
    d.rounded_rectangle((604, 484, 720, 528), radius=7, fill="#1674d2")
    d.text((632, 495), "Save", font=font(17, True), fill="white")
    return image


def pair(before, after):
    canvas = Image.new("RGB", (1280, 504), "#e2e9f1")
    d = ImageDraw.Draw(canvas)
    d.text((18, 7), "BEFORE", font=font(18, True), fill="#1e344c")
    d.text((658, 7), "AFTER", font=font(18, True), fill="#1e344c")
    canvas.paste(before.resize((640, 480), Image.Resampling.LANCZOS), (0, 24))
    canvas.paste(after.resize((640, 480), Image.Resampling.LANCZOS), (640, 24))
    return canvas


def cases():
    base = mock_screen()
    settings = settings_screen()
    return [
        ("chart_opened", "Open a new chart. Expected visible result: a new chart view appears.", base, chart_opened(base), 0),
        ("chart_no_change", "Open a new chart. Expected visible result: a new chart view appears.", base, base, 3),
        ("chart_loading", "Open a new chart. Expected visible result: a new chart view appears.", base, chart_loading(base), 2),
        ("chart_wrong_position_closed", "Open a new chart. Expected visible result: a new chart view appears.", base, position_closed(base), 1),
        ("window_closed", "Close the Market Desk application window. Expected visible result: its window disappears.", base, window_closed(base), 0),
        ("window_wrong_position_closed", "Close the Market Desk application window. Expected visible result: its window disappears.", base, position_closed(base), 1),
        ("position_closed", "Close the selected trading position. Expected visible result: the position row disappears.", base, position_closed(base), 0),
        ("save_no_feedback", "Save the settings. Expected visible result: a confirmation that changes were saved.", settings, settings, 3),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    options = OPTIONS_V1 if args.prompt_version == "v1" else OPTIONS_V2
    started = time.perf_counter()
    model = VisionDecisionModel(str(ROOT / "model"), grad_ckpt=False).to("mps").eval()
    torch.mps.synchronize()
    load_seconds = round(time.perf_counter() - started, 2)

    # Warm the model before measuring case latency.
    warm_image = pair(mock_screen(), chart_opened(mock_screen()))
    warm_ex = Example("The image compares BEFORE on the left with AFTER on the right.",
                      [Q("Which statement is best supported?", options)])
    with torch.inference_mode():
        model.slot_logits(model.prepare([(warm_image, warm_ex)]))
    torch.mps.synchronize()

    results = []
    for name, task, before, after, expected in cases():
        image = pair(before, after)
        image.save(OUT / f"{name}.png")
        ex = Example("Compare the BEFORE screenshot on the left with the AFTER screenshot on the right. "
                     "Judge only visible evidence; do not assume a click succeeded.",
                     [Q(f"Task: {task} Which statement is best supported by the screenshots?", options)])
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model.slot_logits(model.prepare([(image, ex)])), dim=-1)[0, :4].cpu().tolist()
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        predicted = max(range(4), key=lambda i: probabilities[i])
        result = {"case": name, "expected": options[expected], "predicted": options[predicted],
                  "correct": expected == predicted, "probabilities": dict(zip(options, [round(p, 4) for p in probabilities])),
                  "latency_ms": elapsed}
        print(f"{name}: expected={expected} predicted={predicted} p={probabilities[predicted]:.3f} {elapsed:.0f} ms", flush=True)
        results.append(result)

    output = {"model_load_seconds": load_seconds, "image_size": [1280, 504],
              "prompt_version": args.prompt_version, "options": options, "correct": sum(x["correct"] for x in results),
              "total": len(results), "median_latency_ms": round(statistics.median(x["latency_ms"] for x in results), 2),
              "results": results}
    (OUT / f"results_{args.prompt_version}.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
