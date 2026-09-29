# Region search experiment

Run on 2026-09-29 with `Mapika/decider-2b-vision` on the local M3 Max (Apple Metal). This is a controlled demonstration on one generated 1024 × 768 trading-app mockup, not an accuracy estimate across real applications. No real UI clicks were made.

The model received a task, a screenshot crop split into four quadrants, and four fixed answers. Each chosen quadrant became the next crop. A second version named the target control explicitly and stopped after four passes for large buttons or six for the window-close icon. The predicted click was the center of the final region. All latency figures below exclude model loading and HTTP overhead.

| Task | Guided four-way click | Hit? | Four-way latency |
| --- | --- | --- | ---: |
| Open a new chart | (928, 72) | Yes | 1.06 s |
| Close the selected trading position | (864, 696) | Yes | 1.04 s |
| Close the application window | (888, 702) | No; lands on **Close Position** | 1.82 s |

The four-way results are in `guided_four_way/results.json`; `summary.png` shows the paths. Green outlines are target boxes, yellow regions still overlap the target, red regions do not, and red dots are predicted clicks.

## What changed the result

- With task wording alone and a `not visible` option, all three searches eventually failed. The two large buttons were found initially, but the model lost them when crops showed only part of their labels.
- Naming the intended control and stopping after four passes produced clicks inside both large buttons. Continuing to smaller crops caused the model to lose the buttons.
- The task-only and guided four-way runs routed “close the window” toward the prominent **Close Position** button. Keeping the first image at native resolution and drawing a visible × inside the title-bar close circle also failed to locate the correct control, though those variants took a different wrong path.
- When the correct top-left quarter was supplied manually, the model followed the small circle to a 32 × 24 region whose center would hit it. One more pass missed. This suggests that first-pass target selection and over-refinement are separate weaknesses.
- Scoring all four zoomed candidate crops in a batched model call still missed the small close control. It hit the two large buttons, but took 2.8–2.9 s for those tasks and 4.9 s for the failed close task (`parallel_region_scores/results.json`).

The practical design is to use the model for coarse choice among visible, labelled controls, stop at a click-sized region, and validate the selected control through UI accessibility data or another grounding method before acting. The high-confidence wrong route on the close-window task makes unattended clicks unsafe with this prototype.
