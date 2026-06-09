# Unified Benchmark Harness

This folder contains the A2/A3 benchmark runner for validating migrated image-explanation games.

The benchmark compares the original HuggingFace-based game pipeline against the migrated
`ImageImputerFactory + Game.VisionLanguageGame` pipeline. It records run-level outputs,
runtime, model metadata, strategy metadata, and equivalence metrics.

## Input Data

Batch benchmarks use:

```text
data/input/wds_mscoco_captions_test_100
```

The input folder must contain `manifest.csv`. Each row must provide:

| Column | Use |
|---|---|
| `filename` | Image file path relative to the input folder. |
| `first_caption` | Default text input used by the benchmark. |
| `caption` | Full caption text preserved in result CSVs as `text_full`. |

The benchmark uses `first_caption` as the model text input to avoid CLIP-style 77-token limits.
The full caption is still saved for traceability.

For a single image, pass both image and text explicitly:

```powershell
python .\experiments\validation\benchmark_games.py --case insertion_deletion --input assets\dog_and_hydrant.jpg --text "dog" --num-coalitions 4 --batch-size 2 --tolerance 1e-4 --cuda
```

## Preset Suites

| Preset | Plot mode | Purpose | Runs |
|---|---|---|---:|
| `benchmark_suite.pipeline_strict.json` | `strict` | Strict original-vs-migrated numerical equivalence for `patch/crossmodal_mean`. | 200 |
| `benchmark_suite.pipeline_models.json` | `models` | Same pipeline over CLIP, SigLIP, and SigLIP2 model presets. | 800 |
| `benchmark_suite.pipeline_strategies.json` | `strategies` | Migrated pipeline coverage over segmenter/masker strategies. | 800 |
| `benchmark_suite.pipeline_crossmodal.json` | `crossmodal` | Crossmodal image-text player coverage. | 100 |

The staged full benchmark is:

```text
200 + 800 + 800 + 100 = 1900 runs
```

This is not the full Cartesian product. A full product over 8 cases, 100 images, 6 models,
and 4 strategies would be:

```text
8 * 100 * 6 * 4 = 19200 runs
```

## Run Benchmarks

Strict equivalence:

```powershell
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_strict.json --dry-run
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_strict.json
```

Model coverage:

```powershell
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_models.json --dry-run
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_models.json
```

Strategy coverage:

```powershell
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_strategies.json --dry-run
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_strategies.json
```

Crossmodal coverage:

```powershell
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_crossmodal.json --dry-run
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_crossmodal.json
```

Interrupted runs resume automatically. Existing valid `csv/*_comparison.csv` files are skipped.
Use `--force` only when a preset should be recomputed:

```powershell
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_strict.json --force
```

## Runtime Parameters

Important CLI parameters:

| Parameter | Meaning |
|---|---|
| `--config` | JSON preset for batch benchmarks. |
| `--dry-run` | Print planned runs without loading models. |
| `--force` | Recompute existing run CSVs. |
| `--case` | Single migrated game name. |
| `--input` | Single image or manifest-backed input folder. |
| `--text` | Required only for single-image input. |
| `--run-mode` | `compare` for old-vs-migrated, `original` for original-only. |
| `--model-preset` | Standard model preset. |
| `--model-name` | Custom HuggingFace model id. |
| `--segmenter-strategy` | Explicit segmenter strategy, such as `patch` or `slic`. |
| `--masker-strategy` | Explicit masker strategy, such as `crossmodal_mean`, `vision_mean`, or `text_attn`. |
| `--num-coalitions` | Number of sampled coalitions per run. |
| `--batch-size` | Model batch size for coalition evaluation. |
| `--tolerance` | Numerical equivalence threshold. |
| `--cuda` | Run on CUDA. |
| `--use-amp` | Use autocast in the migrated pipeline. |

`random_state` remains supported internally and defaults to `0`, but it is intentionally hidden
from the main preset configs.

## Results

Each preset writes to its own result directory:

```text
experiments/validation/results/<suite-name>/
  csv/
    summary.csv
    *_coverage_table.csv
    *_comparison.csv
  plots/
    *.png
```

The `csv/` folders are ignored by Git. They can contain thousands of large run-level CSVs.
The `plots/` folders are separate and can be committed when needed.

Run CSV filenames are intentionally short:

```text
<case>_<image_stem>_model_<model>_seg_<segmenter>_mask_<masker>_<short_hash>_comparison.csv
```

Full text, full caption, model parameters, strategy parameters, sampling parameters, outputs,
differences, and runtime fields are stored inside the CSV.

## Generate Plots

Plots are generated from existing CSVs. The plot mode is explicit and must match the suite.

Strict:

```powershell
python .\experiments\validation\summarize_results.py --input .\experiments\validation\results\benchmark_pipeline_strict_mscoco100_clip_b32\csv --mode strict
```

Models:

```powershell
python .\experiments\validation\summarize_results.py --input .\experiments\validation\results\benchmark_pipeline_models_mscoco100\csv --mode models
```

Strategies:

```powershell
python .\experiments\validation\summarize_results.py --input .\experiments\validation\results\benchmark_pipeline_strategies_mscoco100_clip_b32\csv --mode strategies
```

Crossmodal:

```powershell
python .\experiments\validation\summarize_results.py --input .\experiments\validation\results\benchmark_pipeline_crossmodal_mscoco100_clip_b32\csv --mode crossmodal
```

## Plot Interpretation

### Strict

Use this suite to answer:

```text
Does the migrated pipeline reproduce the original pipeline outputs?
```

Main outputs:

- `strict_coverage_table.png`: pass count, max output difference, and runtime by validation case.
- `strict_max_output_diff_distribution.png`: run-level max output difference distribution.
- `strict_runtime_by_case.png`: original baseline runtime vs migrated pipeline runtime.

### Models

Use this suite to answer:

```text
Does the pipeline work across multiple vision-language model backbones?
```

Main outputs:

- `models_coverage_table.png`: run count, pass count, max output difference, and runtime by model.
- `models_runtime_by_model.png`: original baseline runtime vs migrated pipeline runtime.
- `models_max_output_diff_by_model.png`: worst output difference per model.
- `models_pass_rate.png`: pass rate per model.

### Strategies

Use this suite to answer:

```text
Do migrated segmenter/masker strategies run successfully, and what runtime cost do they add?
```

Do not interpret non-standard strategies as strict-equivalence failures. Only `patch/crossmodal_mean`
preserves the original patch-based player layout. Strategies such as `vision_mean`, `text_attn`,
and `slic` intentionally change masking or player layout, so old-vs-new coalition equivalence is
not always meaningful.

Main outputs:

- `strategies_coverage_table.png`: completed runs, strict-equivalent runs, baseline-comparable runs, and runtime.
- `strategies_baseline_deviation_by_strategy.png`: deviation from baseline only where baseline comparison is meaningful.
- `strategies_migrated_runtime_by_strategy.png`: migrated pipeline runtime by strategy.
- `strategies_runtime_case_heatmap.png`: migrated runtime by case and strategy.

### Crossmodal

Use this suite to answer:

```text
Does the benchmark cover image-text coalition games?
```

Main outputs:

- `crossmodal_coverage_table.png`: crossmodal pass and runtime summary.
- `crossmodal_max_output_diff_distribution.png`: crossmodal output-difference distribution.
- `crossmodal_original_vs_migrated_runtime.png`: original-vs-migrated runtime scatter.
- `crossmodal_runtime_by_sample.png`: original and migrated runtime by image-text sample.

## Notes

- `patch/crossmodal_mean` is the strict-equivalence baseline.
- Strategy benchmarks primarily measure coverage and runtime, not explanation quality.
- These plots validate output equivalence and pipeline coverage; they do not evaluate explanation quality.
- Large CSV outputs should remain local. Commit selected plots only when they are needed for reporting.
