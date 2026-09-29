"""Render a compact visual summary of the guided four-way experiment."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "region_results"
DATA = json.loads((RESULTS / "guided_four_way" / "results.json").read_text())
SCREEN = Image.open(RESULTS / "mock_desktop.png").convert("RGB")
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 21)


def panel(name, caption):
    image = SCREEN.copy()
    if name:
        task = DATA["tasks"][name]
        box = task["target_box"]
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="#25c456", width=5)
        for step in task["steps"]:
            if step["choice"] == "not visible":
                continue
            x0, y0, x1, y1 = step["crop"]
            xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
            quarters = [(x0, y0, xm, ym), (xm, y0, x1, ym),
                        (x0, ym, xm, y1), (xm, ym, x1, y1)]
            r = quarters[["top-left", "top-right", "bottom-left", "bottom-right"].index(step["choice"])]
            overlaps = not (r[2] <= box[0] or r[0] >= box[2] or r[3] <= box[1] or r[1] >= box[3])
            draw.rectangle(r, outline="#edac26" if overlaps else "#ef3737", width=4)
        if task["predicted_click"]:
            x, y = task["predicted_click"]
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#ef3737", outline="white", width=3)
    image = image.resize((512, 384), Image.Resampling.LANCZOS)
    framed = Image.new("RGB", (512, 421), "#1e293b")
    framed.paste(image, (0, 37))
    ImageDraw.Draw(framed).text((12, 8), caption, font=FONT, fill="white")
    return framed


panels = [
    panel(None, "Controlled mock screen"),
    panel("new_chart", "New Chart: hit (928, 72)"),
    panel("close_position", "Close Position: hit (864, 696)"),
    panel("close_window", "Close window: miss (888, 702)"),
]
canvas = Image.new("RGB", (1024, 842), "#1e293b")
for panel_image, xy in zip(panels, [(0, 0), (512, 0), (0, 421), (512, 421)]):
    canvas.paste(panel_image, xy)
canvas.save(RESULTS / "summary.png")
