# Unified Benchmark Harness

This benchmark validates the full image-explanation pipeline by comparing the original pipeline against the migrated `ImageImputerFactory + Game.VisionLanguageGame` pipeline.

Commands below assume they are run from the project root with the benchmark environment activated:

```powershell
cd D:\TTML\Image-Explanations-with-shapiq
```

## Quick Single-Image Check

Run one image with the default strategy suite:

```powershell
python .\experiments\validation\benchmark_games.py --case insertion_deletion --input assets\dog_and_hydrant.jpg --text "dog" --num-coalitions 4 --batch-size 2 --tolerance 1e-4 --cuda
```

Run only the strict-equivalence strategy:

```powershell
python .\experiments\validation\benchmark_games.py --case insertion_deletion --input assets\dog_and_hydrant.jpg --text "dog" --num-coalitions 1000 --batch-size 16 --tolerance 1e-4 --cuda --segmenter-strategy patch --masker-strategy crossmodal_mean
```

## Pipeline Presets

The preset JSON files use `data/input/wds_mscoco_captions_test_100`. Input directories must contain `manifest.csv`; the benchmark reads each row's `filename` and uses its `caption` column as text input.

| Preset | Purpose | Planned runs |
|---|---|---:|
| `benchmark_suite.pipeline_strict.json` | Strict original-vs-migrated equivalence for the standard `patch/crossmodal_mean` setup. | 200 |
| `benchmark_suite.pipeline_models.json` | Model coverage for base CLIP/SigLIP/SigLIP2 models. | 800 |
| `benchmark_suite.pipeline_strategies.json` | Strategy coverage for the migrated pipeline. | 800 |
| `benchmark_suite.pipeline_crossmodal.json` | Crossmodal value-function coverage with a smaller coalition count. | 100 |

Running all four current presets completes:

```text
200 + 800 + 800 + 100 = 1900 planned runs
```

This is staged full-pipeline validation. It is not an exhaustive Cartesian-product benchmark. A naive exhaustive run over all current cases, 100 images, 6 model presets, and 4 strategies would be:

```text
8 cases * 100 images * 6 models * 4 strategies = 19200 run-level combinations
```

`pipeline_crossmodal` is kept separate because each run expands internally across image and text coalitions.

## Run Presets

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

## Resume And Force

Interrupted runs can be resumed by running the same command again. Existing valid `runs/*_comparison.csv` files are reused and skipped.

Use `--force` to recompute an existing preset:

```powershell
python .\experiments\validation\benchmark_games.py --config .\experiments\validation\benchmark_suite.pipeline_strict.json --force
```

## Main Parameters

- `--config`: JSON batch preset. This is the preferred path for folder-level benchmark runs.
- `--dry-run`: print the expanded benchmark plan without loading models.
- `--force`: recompute runs even when result CSVs already exist.
- `--case`: migrated Python file name without `.py`.
- `--input`: image file or manifest-backed image folder. If omitted, `data/input/wds_mscoco_captions_test_100` is used.
- `--text`: required text input for single-image input. Batch directory inputs read `manifest.csv`.
- `--run-mode`: `compare` runs original-vs-migrated benchmarks; `original` runs only the original pipeline.
- `--model-preset`: one of `clip-vit-b-32`, `clip-vit-b-16`, `clip-vit-l-14`, `siglip-base-p16-224`, `siglip2-base-p32-256`, `siglip2-so400m-p14-384`.
- `--model-name`: custom HuggingFace model id.
- `--segmenter-strategy`: one explicit segmenter, such as `patch`, `slic`, or `gradient_guided`.
- `--masker-strategy`: one explicit masker, such as `crossmodal_mean`, `vision_mean`, or `text_attn`.
- `--num-coalitions`, `--batch-size`, `--tolerance`: benchmark sampling and validation controls.
- `--cuda`: run on CUDA.
- `--use-amp`: enable autocast in the migrated pipeline.

If `--segmenter-strategy` and `--masker-strategy` are omitted, the default stable strategy suite is used. For batch runs, edit one of the `benchmark_suite.pipeline_*.json` presets instead of changing code.

## Result Files

Each preset writes to its own result directory:

```text
experiments/validation/results/<preset-name>/
  summary.csv
  plots/
    max_output_diff_distribution.png
    top_max_output_diff.png
    mean_max_output_diff_heatmap.png
    mean_runtime_heatmap.png
  runs/
    *_comparison.csv
```

The CSV records the full run context, model metadata, segmenter/masker strategy metadata, original pipeline outputs, migrated pipeline outputs, output differences when player layouts match, and runtime fields.

Strict numerical equivalence should use `patch/crossmodal_mean`, because it preserves the original patch-based player layout. Strategies such as `slic` or `gradient_guided` are benchmark strategies; their player layout differs from the original pipeline, so coalition-by-coalition equivalence is not directly meaningful.

## Summaries And Plots

Summarize existing CSV results:

```powershell
python .\experiments\validation\summarize_results.py
```

The summary script recursively scans all preset folders under `experiments/validation/results/`.

Recommended plot interpretation:

- `max_output_diff_distribution.png`: distribution of run-level max output differences.
- `top_max_output_diff.png`: the largest-difference runs for quick debugging.
- `mean_max_output_diff_heatmap.png`: aggregated mean max difference by case and model.
- `mean_runtime_heatmap.png`: aggregated migrated runtime by case and strategy.
- `summary.csv`: preferred source for additional grouped plots by case, model, strategy, and input.

The benchmark does not generate one bar per run by default. That format is not readable for the current 1900-run staged benchmark.
