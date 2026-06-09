# Explanation Quality Benchmark

This folder validates explanation quality for the migrated image-explanation pipeline.
It is separate from `../equivalence`, which checks old-vs-migrated value-function
equivalence and pipeline coverage.

The current quality metric is AID:

```text
AID = area between least-important-first and most-important-first deletion curves
```

Higher AID is better. A good explanation makes the most-important-first deletion
curve drop faster than the least-important-first deletion curve.

## Scope

This benchmark runs the migrated pipeline end to end:

```text
image + text + model + segmenter + masker
-> build migrated game
-> generate InteractionValues with FIxLIP / ProxySHAP
-> cache InteractionValues
-> compute AID curves
-> write CSV and plots
```

It evaluates explanation quality, not strict old-vs-new numerical equivalence.
Use `../equivalence` for A2/A3 equivalence and coverage results.

## Files

| File | Purpose |
|---|---|
| `benchmark_aid.py` | CLI entry point and AID benchmark runner. |
| `aid_schema.py` | Shared constants, output fields, sample dataclass, default methods and strategies. |
| `aid_suite.py` | JSON config loading, manifest input expansion, and config normalization. |
| `aid_outputs.py` | Resume keys, CSV rows, `InteractionValues` paths, and append helpers. |
| `aid_plots.py` | AID score plots, mean curves, sample curve grids, runtime plots, and coverage table. |
| `plot_fixlip_qualitative.py` | Paper-style qualitative image/text figure from one cached `InteractionValues` run. |

## Input Data

The 1600-run suite uses:

```text
data/input/wds_mscoco_captions_test_100
```

The folder must contain `manifest.csv` with these columns:

| Column | Use |
|---|---|
| `filename` | Image path relative to the input folder. |
| `first_caption` | Default model text input. |
| `caption` | Full caption preserved in result CSVs as `text_full`. |

The benchmark uses `first_caption` as model input to avoid CLIP-style 77-token
limits. The full `caption` is still saved for traceability.

## 1600-Run Suite

Preset:

```text
benchmark_suite.aid_clip_siglip_mscoco100.json
```

Planned run count:

```text
100 images * 2 models * 2 strategies * 4 methods = 1600 runs
```

Current suite dimensions:

| Dimension | Values |
|---|---|
| Models | `clip-vit-b-32`, `siglip-base-p16-224` |
| Strategies | `patch_crossmodal_mean`, `slic_crossmodal_mean` |
| Methods | `kernelshap_shapley_order1`, `proxyshap_banzhaf_p03_order2`, `proxyshap_banzhaf_p05_order2`, `proxyshap_banzhaf_p07_order2` |
| Samples | 100 MS COCO caption samples |

Default runtime parameters in the config:

| Parameter | Value |
|---|---:|
| `batch_size` | 64 |
| `curve_points` | 101 |
| `budget` | 4096 per method |
| `cuda` | true |
| `use_amp` | false |
| `text_column` | `first_caption` |

`random_state` is supported internally and defaults to `0`, but it is intentionally
not shown in the main config.

## Run

Dry run first. This checks the planned 1600 runs without loading models:

```powershell
D:\TTML\ttml\Scripts\python.exe .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json --dry-run
```

Run the full 1600-run benchmark:

```powershell
D:\TTML\ttml\Scripts\python.exe .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json
```

Interrupted runs resume automatically from completed rows in `aid_summary.csv`.
Use `--force` only when the whole suite should be recomputed:

```powershell
D:\TTML\ttml\Scripts\python.exe .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json --force
```

## Plot

The benchmark writes plots automatically at the end of the run.

To regenerate plots from existing CSV files without rerunning explanations:

```powershell
D:\TTML\ttml\Scripts\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('experiments/validation/quality').resolve())); from aid_plots import write_aid_plots; base = Path('experiments/validation/quality/results/benchmark_aid_clip_siglip_mscoco100'); print(write_aid_plots(base / 'csv' / 'aid_summary.csv', base / 'csv' / 'aid_curves.csv', base / 'plots'))"
```

To generate one paper-style qualitative FIxLIP figure from a completed run:

```powershell
D:\TTML\ttml\Scripts\python.exe .\experiments\validation\quality\plot_fixlip_qualitative.py --summary .\experiments\validation\quality\results\benchmark_aid_clip_siglip_mscoco100\csv\aid_summary.csv
```

This qualitative plot reuses `ImputerFactory.plot.plot_image_and_text_together`.

## Outputs

```text
experiments/validation/quality/results/benchmark_aid_clip_siglip_mscoco100/
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

The `csv/` folder and `interaction_values/` cache are ignored by Git. Commit only
selected plots when they are useful for reporting.

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
