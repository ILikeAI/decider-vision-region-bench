# Close-window failure probe

This follow-up isolates the 256 × 192 parent region where the original parallel crop search became ambiguous. It checks repeatability and compares six ways of selecting the next quadrant.

## What “parallel binary crops” means

The full screen is divided into four quarter-size image tiles. Each tile is enlarged to the model's input size, so a small control occupies more visual tokens than it would in the full screenshot. The four enlarged images are placed in one model batch. Each image independently receives the same yes/no question: does this tile contain the requested control?

“Parallel” means one batched model forward pass over four images. It does not mean four processes, and the model is not shown four grid locations inside one image. The returned percentage is the model's softmax probability for `yes` versus `no` on that tile.

The tiling idea worked as intended during coarse search: the correct quarter was much stronger than the other three. The policy error was treating the largest percentage as a valid positive detection even when all four tiles were classified as `no`. A production policy should distinguish these outcomes:

1. One tile passes a calibrated detection threshold: descend into it.
2. Several tiles pass: retain the top candidates or their union.
3. No tile passes, but one is clearly stronger: treat it as a weak proposal and use a contextual comparison before descending.
4. No tile passes and the scores are similar: abstain, change the tiling, or escalate.

## Determinism

The exact original four-crop inference was repeated 20 times in one loaded process and 10 more times after loading the model in a fresh process. Every probability was identical to at least ten decimal places, and every repetition chose the same quadrant.

At the failing third pass, the original binary yes probabilities were:

| Candidate | Yes probability |
| --- | ---: |
| Top-left, correct | 0.0162774604 |
| Top-right | 0.0155436480 |
| Bottom-left | 0.0050998181 |
| Bottom-right | 0.0083804540 |

This was a deterministic policy failure rather than a sampling fluke. The model said “no” to every crop, while the search policy still selected the least-negative answer.

## Recovery attempts from the failing parent region

Each method refined the 256 × 192 parent through three additional levels to a 32 × 24 final cell.

| Method | Hit? | Error | Measured latency |
| --- | --- | ---: | ---: |
| Original four binary semantic crops | No | 38.64 px | 2.00 s |
| Four binary crops with visual wording | No | 26.93 px | 2.02 s |
| One four-way semantic grid | Yes | 12.21 px | 0.80 s |
| One four-way visual grid | Yes | 12.21 px | 0.69 s |
| Four-way visual grid with large labels | No | 15.65 px | 0.72 s |
| Four highlighted candidate images | Yes | 12.21 px | 2.11 s |

The large A/B/C/D labels eventually obscured the small target, demonstrating that annotations can change the visual evidence being measured.

## Adaptive full-screen result

The successful hybrid starts from the full 1024 × 768 screenshot:

1. Enlarge and batch four tiles while the leading yes score is at least 0.10 and leads the runner-up by at least 0.05.
2. When that evidence collapses, switch permanently to one contextual four-way grid image.
3. Continue until the region is click-sized.

The hybrid selected the correct quadrant on all five passes and clicked `(16, 12)`, inside the target, with 12.21 px centre error. Measured search latency was 2.71 seconds after model loading. This is one authored screen and establishes mechanism, not general accuracy.

Full outputs are in `results_final.json`. `results_run_2.json` contains ten exact repetitions from a separate model process.
