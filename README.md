# Decider vision region benchmark

Local experiments with [`Mapika/decider-2b-vision`](https://huggingface.co/Mapika/decider-2b-vision) for fast visual decisions on screenshots.

The experiments cover:

- local inference latency on Apple Silicon;
- recursive screen region selection for tasks such as “open a new chart” and “close the window”;
- parallel scoring of four screen regions before zooming into the strongest candidate;
- visual verification of whether an attempted UI action succeeded.

## Results so far

Tests were run on an Apple M3 Max MacBook Pro with 40 GPU cores and 64 GB unified memory, using PyTorch 2.14.0, Transformers 5.17.0, and Apple Metal (MPS).

### Raw model latency

| Image | Questions | Median end to end | p95 end to end |
| --- | ---: | ---: | ---: |
| 256 × 240 | 1 | 125.78 ms | 131.04 ms |
| 1000 × 750 | 1 | 618.17 ms | 654.54 ms |
| 256 × 240 | 5 | 177.77 ms | 184.58 ms |

Model loading took 8.04 seconds. The first request after loading took 835 ms; an earlier cold run took 3.78 seconds while Metal kernels initialized. Peak process resident memory was 8.76 GiB. Full timing samples and output probabilities are in [`results.json`](results.json).

### Recursive screen region selection

The four way guided search correctly localized the large **New Chart** and **Close Position** controls in the generated desktop mockup in roughly 1.05 seconds. It did not reliably localize the much smaller window close button. The parallel region score experiment showed the same pattern: good discrimination for large, semantically distinct controls and weak discrimination for a small icon.

See [`region_results/README.md`](region_results/README.md) and [`region_results/summary.png`](region_results/summary.png) for the run details and visualizations.

### Visual outcome verification

On eight generated before and after pairs, the initial questions classified 3/8 correctly. Clearer state specific questions improved that to 5/8, with median inference latency of 563 ms. This is promising as a cheap verifier, but it is not reliable enough to be the sole success signal for UI automation.

See [`verification_results/README.md`](verification_results/README.md) for the cases and recommendations.

## Setup

The model weights and upstream inference repository are intentionally excluded from Git.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone https://github.com/Mapika/decider.git decider
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Mapika/decider-2b-vision", local_dir="model")'
```

## Run the experiments

```bash
# Raw latency
.venv/bin/python benchmark.py --reps 30 --warmup 5 --output results.json

# Recursive region search
.venv/bin/python region_search.py

# Parallel four region scoring
.venv/bin/python region_score_search.py

# Before and after visual verification
.venv/bin/python visual_verify.py

# Regenerate the region result montage
.venv/bin/python make_region_summary.py
```

Use the scripts as experimental benchmarks, not as a production click policy. Small controls, ambiguous prompts, and visually similar states remain failure modes. A practical computer use system should combine this model with accessibility or DOM targets when available and use explicit abstention thresholds before clicking.

## Exact model versions

- Model revision: `Mapika/decider-2b-vision@863e290863655f1d6b69324d77d09ac972d21609`
- Inference code: `Mapika/decider@23579f7a7e8f10e1045be492af3c1c05a005d67c`
