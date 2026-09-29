# Visual outcome verification pilot

Run on 2026-09-29 with local `Mapika/decider-2b-vision` on Apple M3 Max / MPS. The eight BEFORE/AFTER pairs are authored mock trading-app screens. They do not estimate accuracy on real applications, backend state, or delayed workflows. No live desktop action was taken.

The model answered one four-way question per pair: expected outcome visibly present, different change, still processing, or insufficient visual evidence. The pair was one 1280 × 504 image. The prompt and case definitions are in [`visual_verify.py`](../visual_verify.py); images and exact probabilities are in this directory. Model load and first warmup are excluded from the listed case latency.

| Case | Expected | Initial wording | Clearer wording |
| --- | --- | --- | --- |
| New chart appears | Visible success | Success | Success |
| New chart request, screen unchanged | Insufficient evidence | **False success** | Different change |
| New chart request, loading overlay | Processing | Processing | Processing |
| New chart request, position disappears instead | Different change | **False success** | Different change |
| Close window, desktop appears | Visible success | Different change | Different change |
| Close window, position disappears instead | Different change | Insufficient evidence | Different change |
| Close position, row disappears | Visible success | Different change | Success |
| Save settings, screen unchanged | Insufficient evidence | Insufficient evidence | Different change |

Initial wording: **3/8** correct. Clearer wording, chosen after seeing the initial errors: **5/8** correct. The clearer prompt removed false-success calls on these eight cases but still misclassified a genuinely closed window and both unchanged screenshots. Median measured inference was **585 ms** initially and **563 ms** with clearer wording. These are unreplicated tiny-sample observations; the second result is prompt tuning on the same cases, not a held-out improvement.

The useful role to test next is an optional visual evidence signal attached to a specific expected postcondition. It should be allowed to say “cannot tell.” A real benchmark should use stored, consented computer-use traces with independently verified outcomes, include no-op/wrong-target/stale/delayed/visually hidden effects, and measure **false-success rate** and abstention coverage before any completion gate uses it. DOM, AX, application state or a server receipt should remain the verifier when they can directly establish the outcome. A visible UI change may still be only an intermediate step, such as a trading position disappearing while backend settlement is not yet confirmed.
