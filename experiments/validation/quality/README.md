# Explanation Quality Benchmark

This folder contains the explanation-quality benchmark for the migrated image-explanation
pipeline. It is separate from `../equivalence`, which checks old-vs-migrated value-function
equivalence and model or strategy coverage.

Use this benchmark to answer:

```text
Do the migrated explanations produce useful deletion-curve behavior?
```

The current quality metric is AID:

```text
AID = area between least-important-first deletion and most-important-first deletion curves
```

Higher AID is better. A good explanation removes important players first, so the
most-important-first curve should drop faster than the least-important-first curve.

## Validation Summary

This benchmark is the explanation-quality side of the validation work:

| Benchmark | Main question | Main metric |
|---|---|---|
| `../equivalence` | Does the migrated game output match or cover the old pipeline? | max absolute output difference, pass rate, runtime |
| `quality` | Are the generated explanations useful? | AID area between deletion curves |

The quality benchmark runs the migrated pipeline end to end:

```text
image + text + model + segmenter + masker
-> build migrated game
-> generate InteractionValues with FIxLIP / ProxySHAP
-> cache InteractionValues
-> compute AID deletion curves
-> write CSV and plots
```

## Files

| File | Purpose |
|---|---|
| `benchmark_aid.py` | CLI entry point and AID benchmark runner. |
| `aid_schema.py` | Shared constants, output fields, sample dataclass, default methods and strategies. |
| `aid_suite.py` | JSON config loading, manifest input expansion, and config normalization. |
| `aid_outputs.py` | Resume keys, CSV rows, `InteractionValues` paths, metadata, and append helpers. |
| `aid_plots.py` | AID score plots, mean curves, sample curve grids, runtime plots, and coverage table. |
| `plot_results.py` | Plot CLI for existing AID CSV folders. |
| `plot_fixlip_qualitative.py` | Paper-style qualitative image/text figure from one cached `InteractionValues` run. |

## Input Data

The default suite uses:

```text
data/input/wds_mscoco_captions_test_100
```

The folder must contain `manifest.csv`:

| Column | Use |
|---|---|
| `filename` | Image path relative to the input folder. |
| `first_caption` | Default model text input. |
| `caption` | Full caption preserved in result CSVs as `text_full`. |

The benchmark uses `first_caption` as model input to avoid CLIP-style 77-token limits.
The full `caption` is still saved for traceability.

## Preset Suites

### Preview Suite

Preset:

```text
benchmark_suite.aid_preview_assets.json
```

Planned run count:

```text
2 images * 2 models * 4 strategies * 2 methods = 32 runs
```

This suite is intended for fast visual inspection. It uses `assets/dog_and_hydrant.jpg` and
`assets/giraffe_drinking.jpg`, compares CLIP and SigLIP, and includes representative patch, SLIC,
and gradient-guided strategies. It does not replace the full coverage suite.

### Full Coverage Suite

Preset:

```text
benchmark_suite.aid_clip_siglip_mscoco100.json
```

Planned run count:

```text
100 images * 2 models * 18 strategies * 4 methods = 14400 runs
```

Full suite dimensions:

| Dimension | Values |
|---|---|
| Models | `clip-vit-b-32`, `siglip-base-p16-224` |
| Segmenters | `patch`, `slic`, `gradient_guided` |
| Maskers | `crossmodal_mean`, `crossmodal_blur`, `vision_mean`, `vision_blur`, `text_attn`, `attention` |
| Strategies | Full 3 segmenter * 6 masker Cartesian coverage, 18 combinations total |
| Methods | `kernelshap_shapley_order1`, `proxyshap_banzhaf_p03_order2`, `proxyshap_banzhaf_p05_order2`, `proxyshap_banzhaf_p07_order2` |
| Samples | 100 MS COCO caption samples |

Default runtime parameters:

| Parameter | Value |
|---|---:|
| `batch_size` | 64 |
| `curve_points` | 101 |
| `budget` | 4096 per method |
| `cuda` | true |
| `use_amp` | false |
| `text_column` | `first_caption` |

`random_state` remains supported internally and defaults to `0`, but it is intentionally
hidden from the main config.

## Run

Run from the repository root with the project environment activated.

Preview the 32-run plan without loading models:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_preview_assets.json --dry-run
```

Run the 32-run preview benchmark:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_preview_assets.json
```

Preview the 14400-run full plan without loading models:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json --dry-run
```

Run the full benchmark:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json
```

Normal runs print the planned run summary first, save it under `metadata/`, and then start
execution. Interrupted runs resume automatically from completed rows in `aid_summary.csv`.

Use `--force` only when the whole suite should be recomputed:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json --force
```

## Plot

The benchmark writes plots automatically at the end of a full run.

Regenerate plots from existing CSV files without rerunning explanations:

```powershell
python .\experiments\validation\quality\plot_results.py --input .\experiments\validation\quality\results\benchmark_aid_clip_siglip_mscoco100
```

Regenerate plots for the preview suite:

```powershell
python .\experiments\validation\quality\plot_results.py --input .\experiments\validation\quality\results\benchmark_aid_preview_assets
```

Generate one paper-style qualitative FIxLIP figure from a completed run:

```powershell
python .\experiments\validation\quality\plot_fixlip_qualitative.py --summary .\experiments\validation\quality\results\benchmark_aid_clip_siglip_mscoco100\csv\aid_summary.csv
```

This qualitative plot reuses `ImputerFactory.plot.plot_image_and_text_together`.

## Outputs

```text
experiments/validation/quality/results/benchmark_aid_clip_siglip_mscoco100/
  metadata/
    benchmark_plan.json
    suite_normalized.json
    cli_args.json
    environment.json
    config_used.json
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
    quality_aid_sample_curves_<model>_<strategy>_<method>.png
    qualitative/
      <run_id>_fixlip_paper_style.png
```

The `csv/` folder and `interaction_values/` cache are ignored by Git. Commit only selected
plots when they are useful for reporting.

## Important CSV Fields

| Field | Meaning |
|---|---|
| `aid_area_between_curves` | Main AID quality score; higher is better. |
| `aid_mean_gap` | Mean normalized LIF-MIF gap. |
| `mif_deletion_auc` | Area under most-important-first deletion curve; lower is better. |
| `lif_deletion_auc` | Area under least-important-first deletion curve; higher is better. |
| `baseline_aid_area_between_curves` | First-order baseline score for order-2 explanations. |
| `explanation_runtime_s` | Runtime for FIxLIP/ProxySHAP explanation generation. |
| `curve_evaluation_runtime_s` | Runtime for curve construction and value-function evaluation. |
| `interaction_cache_hit` | Whether cached `InteractionValues` were reused. |

## Plot Interpretation

| Plot | Meaning |
|---|---|
| `quality_aid_score_mean_std.png` | Mean AID score with standard deviation by model, strategy, and method. |
| `quality_aid_quality_runtime_tradeoff.png` | Mean AID quality versus explanation runtime. |
| `quality_aid_coverage_table.png` | Completed, failed, and success-rate summary. |
| `quality_aid_mean_deletion_curves_<model>_<strategy>.png` | Mean deletion curves grouped by model and strategy. |
| `quality_aid_sample_curves_<model>_<strategy>_<method>.png` | Teammate-style sample curve grid for direct visual inspection. |
| `qualitative/<run_id>_fixlip_paper_style.png` | FIxLIP paper-style image/text interaction visualization for one run. |

## Known Limitations

- AID measures deletion-curve explanation quality, not old-vs-new numerical equivalence.
- The 14400-run suite is large and is better suited for a server when all curves are required.
- Current quality coverage includes all benchmark-supported segmenters and maskers. If a specific
  combination fails, keep the failed row in the summary first; that failure is useful strategy
  compatibility evidence.
