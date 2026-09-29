"""Test iterative quadrant selection with decider-2b-vision on a mock desktop UI."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "region_results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "decider"))

from decider.infer import Example, Q
from decider.vision import VisionDecisionModel

SIZE = (1024, 768)
OPTIONS = ["top-left", "top-right", "bottom-left", "bottom-right", "not visible"]
TASKS = {
    "close_window": {
        "instruction": "Close the application window.",
        "target_description": "the small red circular window-close control in the title bar",
        "visual_description": "the red circle among the three coloured circles on the dark title bar",
        "target_box": (12, 11, 34, 33),
        "guided_depth": 6,
    },
    "new_chart": {
        "instruction": "Open a new chart.",
        "target_description": "the blue New Chart button",
        "target_box": (832, 57, 994, 94),
        "guided_depth": 4,
    },
    "close_position": {
        "instruction": "Close the selected trading position.",
        "target_description": "the red Close Position button",
        "target_box": (824, 680, 987, 721),
        "guided_depth": 4,
    },
}


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def mock_screen(show_close_x=False):
    im = Image.new("RGB", SIZE, "#eef2f6")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1024, 37), fill="#243246")
    for x, colour in ((23, "#ff605c"), (49, "#ffbd44"), (75, "#00ca4e")):
        d.ellipse((x - 10, 12, x + 10, 32), fill=colour)
    if show_close_x:
        d.line((19, 18, 27, 26), fill="white", width=2)
        d.line((27, 18, 19, 26), fill="white", width=2)
    d.text((450, 10), "Market Desk", font=font(16, True), fill="white")

    d.rectangle((0, 37, 1024, 106), fill="white")
    d.line((0, 105, 1024, 105), fill="#d6dee7", width=2)
    d.text((214, 60), "Portfolio overview", font=font(24, True), fill="#263448")
    d.rounded_rectangle((832, 57, 994, 94), radius=7, fill="#1674d2")
    d.text((849, 66), "+  New Chart", font=font(18, True), fill="white")

    d.rectangle((0, 106, 190, 768), fill="#e5ebf2")
    d.text((25, 131), "WORKSPACE", font=font(13, True), fill="#607286")
    for i, label in enumerate(("Overview", "Charts", "Watchlist", "Positions", "Reports")):
        y = 174 + i * 51
        if label == "Charts":
            d.rounded_rectangle((13, y - 7, 176, y + 32), radius=6, fill="#cbdff5")
        d.text((29, y), label, font=font(18), fill="#25374c")

    d.rounded_rectangle((211, 124, 788, 558), radius=10, fill="white", outline="#d5dde7", width=2)
    d.text((238, 144), "EUR / USD", font=font(21, True), fill="#203249")
    d.text((238, 177), "1.1842   +0.42% today", font=font(16), fill="#16846a")
    d.line((236, 480, 759, 480), fill="#c7d2df", width=2)
    d.line((236, 225, 236, 481), fill="#c7d2df", width=2)
    points = [(242, 429), (295, 402), (344, 416), (391, 340), (436, 359),
              (485, 310), (539, 329), (592, 274), (641, 292), (691, 239), (753, 259)]
    d.line(points, fill="#1674d2", width=5, joint="curve")

    d.rounded_rectangle((803, 124, 1003, 557), radius=10, fill="white", outline="#d5dde7", width=2)
    d.text((824, 145), "Chart settings", font=font(19, True), fill="#263448")
    d.text((976, 145), "×", font=font(22, True), fill="#6d7c8e")
    for i, label in enumerate(("Timeframe", "Indicators", "Chart type", "Appearance")):
        d.text((824, 202 + i * 62), label, font=font(16), fill="#46586c")
        d.line((824, 232 + i * 62, 982, 232 + i * 62), fill="#e0e6ec", width=1)

    d.rounded_rectangle((211, 579, 1003, 742), radius=10, fill="white", outline="#d5dde7", width=2)
    d.text((237, 601), "Open positions", font=font(20, True), fill="#263448")
    d.text((237, 654), "EUR / USD      Long      +2.4%", font=font(18), fill="#33465a")
    d.rounded_rectangle((824, 680, 987, 721), radius=7, fill="#c23b3d")
    d.text((840, 690), "Close Position", font=font(17, True), fill="white")
    return im


def quadrant(rect, choice):
    x0, y0, x1, y1 = rect
    xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
    return [
        (x0, y0, xm, ym),
        (xm, y0, x1, ym),
        (x0, ym, xm, y1),
        (xm, ym, x1, y1),
    ][choice]


def grid_crop(screen, rect, native=False):
    im = screen.crop(rect)
    if not native:
        im = im.resize((512, 384), Image.Resampling.NEAREST if im.width < 128 else Image.Resampling.BICUBIC)
    d = ImageDraw.Draw(im)
    d.line((im.width // 2, 0, im.width // 2, im.height), fill="#f4ce41", width=4)
    d.line((0, im.height // 2, im.width, im.height // 2), fill="#f4ce41", width=4)
    return im


def target_centre(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def expected_quadrant(rect, point):
    x0, y0, x1, y1 = rect
    if not (x0 <= point[0] < x1 and y0 <= point[1] < y1):
        return 4
    return int(point[0] >= (x0 + x1) // 2) + 2 * int(point[1] >= (y0 + y1) // 2)


def run_task(model, screen, name, spec, out_dir=OUT, guided=False, force_region=False,
             locator_only=False, fullres_first=False, oracle_first=False):
    rect = quadrant((0, 0, *SIZE), 0) if oracle_first else (0, 0, *SIZE)
    goal = target_centre(spec["target_box"])
    steps = []
    options = OPTIONS[:4] if force_region else OPTIONS
    max_depth = spec["guided_depth"] if guided else 6
    for depth in range(2 if oracle_first else 1, max_depth + 1):
        image = grid_crop(screen, rect, native=fullres_first and depth == 1)
        image.save(out_dir / f"{name}_pass_{depth}.png")
        target = (f" Target control: {spec['target_description']}." if guided else "")
        instruction = (f"Locate {spec['visual_description']}." if locator_only else
                       f"Task: {spec['instruction']}{target}")
        question = (f"{instruction} "
                    "The yellow lines divide this crop into four equal quarters. "
                    "Which quarter contains the center of the control to click?")
        if not force_region:
            question += " Choose not visible if that control is absent from this crop."
        example = Example("Look only at the displayed screen crop and locate the task's control.",
                          [Q(question, options)])
        torch.mps.synchronize()
        started = time.perf_counter()
        inp = model.prepare([(image, example)])
        with torch.inference_mode():
            probabilities = torch.softmax(model.slot_logits(inp), dim=-1)[0, :len(options)].cpu().tolist()
        ms = (time.perf_counter() - started) * 1000
        choice = max(range(len(options)), key=lambda i: probabilities[i])
        correct_choice = expected_quadrant(rect, goal)
        steps.append({
            "pass": depth,
            "crop": rect,
            "choice": options[choice],
            "expected": OPTIONS[correct_choice],
            "correct": choice == correct_choice,
            "probabilities": {option: round(probabilities[i], 4) for i, option in enumerate(options)},
            "latency_ms": round(ms, 2),
        })
        print(f"{name} pass {depth}: {options[choice]} ({probabilities[choice]:.3f}), "
              f"expected {OPTIONS[correct_choice]}, {ms:.0f} ms", flush=True)
        if choice == 4:
            break
        rect = quadrant(rect, choice)

    click = None if steps[-1]["choice"] == "not visible" else target_centre(rect)
    box = spec["target_box"]
    hit = click is not None and box[0] <= click[0] <= box[2] and box[1] <= click[1] <= box[3]
    distance = None if click is None else round(math.dist(click, goal), 1)
    visual = screen.copy()
    d = ImageDraw.Draw(visual)
    d.rectangle(box, outline="#20c55a", width=4)
    for step in steps:
        if step["choice"] == "not visible":
            continue
        path_rect = quadrant(step["crop"], OPTIONS.index(step["choice"]))
        d.rectangle(path_rect, outline="#f0b823" if step["correct"] else "#f44336", width=3)
    if click:
        x, y = click
        d.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#e92b29", outline="white", width=2)
    visual.save(out_dir / f"{name}_result.png")
    return {
        "task": spec["instruction"],
        "target_box": box,
        "target_centre": goal,
        "predicted_click": click,
        "click_hits_target": hit,
        "click_error_pixels": distance,
        "total_latency_ms": round(sum(s["latency_ms"] for s in steps), 2),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guided", action="store_true", help="Name the target control and stop at task-specific depth")
    parser.add_argument("--force-region", action="store_true", help="Offer only the four quadrants")
    parser.add_argument("--locator-only", action="store_true", help="Ask only for visual localization")
    parser.add_argument("--fullres-first", action="store_true", help="Keep the first screenshot at native resolution")
    parser.add_argument("--oracle-first", action="store_true", help="Start from the correct top-left quadrant")
    parser.add_argument("--show-close-x", action="store_true", help="Draw a visible X inside the red close control")
    parser.add_argument("--only", choices=list(TASKS), help="Run one task")
    args = parser.parse_args()
    out_dir = OUT / ("visible_x" if args.show_close_x else
                     "oracle_first" if args.oracle_first else
                     "fullres_first" if args.fullres_first else
                     "locator_only_four_way" if args.locator_only and args.force_region else
                     "guided_four_way" if args.guided and args.force_region else
                     "guided" if args.guided else "four_way" if args.force_region else "baseline")
    out_dir.mkdir(exist_ok=True)
    screen = mock_screen(show_close_x=args.show_close_x)
    screen.save(out_dir / "mock_desktop.png")
    started = time.perf_counter()
    model = VisionDecisionModel(str(ROOT / "model"), grad_ckpt=False).to("mps").eval()
    torch.mps.synchronize()
    load_s = time.perf_counter() - started
    print(f"Loaded model in {load_s:.2f} s", flush=True)
    results = {"model_load_seconds": round(load_s, 2), "guided": args.guided,
               "force_region": args.force_region, "locator_only": args.locator_only,
               "fullres_first": args.fullres_first, "oracle_first": args.oracle_first,
               "show_close_x": args.show_close_x,
               "tasks": {}}
    for name, spec in TASKS.items():
        if args.only and name != args.only:
            continue
        results["tasks"][name] = run_task(model, screen, name, spec, out_dir,
                                          args.guided, args.force_region, args.locator_only,
                                          args.fullres_first, args.oracle_first)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
