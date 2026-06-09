# Explanation Quality Validation

This folder contains quality benchmarks for explanation outputs. It is separate from
`../equivalence`, which checks old-vs-migrated value-function equivalence and coverage.

The first quality metric is AID:

```text
AID = area between least-important-first and most-important-first deletion curves
```

Higher AID is better. A good explanation removes important players first and makes the
most-important-first curve drop faster than the least-important-first curve.

This runner executes the full quality pipeline:

```text
image + text + model + segmenter + masker
-> build migrated game
-> generate InteractionValues with FIxLIP / ProxySHAP
-> cache InteractionValues in this result folder
-> compute AID curves
-> write CSV and plots
```

## Runner

The quality benchmark mirrors the equivalence benchmark structure:

| File | Purpose |
|---|---|
| `benchmark_aid.py` | CLI entry point, model/game construction, AID evaluation loop. |
| `aid_schema.py` | Shared constants, output fields, sample dataclass, default methods and strategies. |
| `aid_suite.py` | JSON config loading, manifest input expansion, model/strategy/method normalization. |
| `aid_outputs.py` | Resume keys, summary rows, curve rows, CSV append helpers. |
| `aid_plots.py` | Aggregated AID score, mean curve, and runtime plots. |

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json --dry-run
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json
```

Single-image CLI runs require image and text explicitly:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --input assets\dog_and_hydrant.jpg --text "dog" --model-preset clip-vit-b-32 --mode banzhaf/0.5 --order 2 --method smoke_banzhaf_p05_order2 --budget 512 --cuda
```

Batch runs read `manifest.csv` directly. The default config uses:

```text
data/input/wds_mscoco_captions_test_100
```

The benchmark uses `first_caption` as model text input to avoid CLIP-style 77-token issues.
The complete `caption` field is still saved in CSV as `text_full`.

## Preset

| Preset | Purpose | Planned runs |
|---|---|---:|
| `benchmark_suite.aid_clip_siglip_mscoco100.json` | AID quality over CLIP and SigLIP, 100 MS COCO samples, patch/SLIC strategies, and four explanation methods. | 1600 |

The runner writes generated `InteractionValues` to the result folder and reuses them on resume.
Use `--force` to regenerate explanations and recompute AID.

## Outputs

```text
experiments/validation/quality/results/<suite-name>/
  csv/
    aid_summary.csv
    aid_curves.csv
  interaction_values/
    *.json
  plots/
    quality_aid_score_mean_std.png
    quality_aid_quality_runtime_tradeoff.png
    quality_aid_coverage_table.png
    quality_aid_mean_deletion_curves_<model>_<strategy>.png
```

The `csv/` folder is ignored by Git. Selected `plots/` can be committed for reporting.
The `interaction_values/` folder is ignored by Git because these files are generated cache data.
Interrupted runs resume automatically from completed rows in `aid_summary.csv`. Use `--force`
only when the suite should be recomputed from scratch.

Important CSV fields:

| Field | Meaning |
|---|---|
| `aid_area_between_curves` | Main quality score; higher is better. |
| `aid_mean_gap` | Old teammate-compatible mean normalized LIF-MIF gap. |
| `mif_deletion_auc` | Area under most-important-first deletion curve; lower is better. |
| `lif_deletion_auc` | Area under least-important-first deletion curve; higher is better. |
| `baseline_aid_area_between_curves` | First-order baseline score for order-2 explanations. |
| `explanation_runtime_s` | Runtime for FIxLIP/ProxySHAP explanation generation. |
| `curve_evaluation_runtime_s` | Runtime for curve construction and value-function evaluation. |
| `interaction_cache_hit` | Whether the run reused cached `InteractionValues`. |

## Scope

- AID validates explanation quality, not pipeline equivalence.
- Equivalence and coverage remain in `../equivalence`.
- Faithfulness metrics in `src/evaluation.py` can be added later as a second quality runner.
