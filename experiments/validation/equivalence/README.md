# Equivalence and Coverage Benchmark

This folder contains the A2/A3 benchmark runner for migrated image-explanation games.

The benchmark compares the original FIxLIP-style HuggingFace game pipeline against the migrated
`ImageImputerFactory + ImageImputer + Game.VisionLanguageGame` pipeline. It records run-level
outputs, runtime, model metadata, strategy metadata, and equivalence or baseline-deviation metrics.

This benchmark validates **value-function behavior** and **pipeline coverage**. It does **not**
evaluate explanation quality. Explanation quality validation belongs in `../quality/` and should
use AID, insertion-deletion, and faithfulness-style quality metrics.

## Scope

This benchmark is used to answer three questions:

1. Does the migrated pipeline reproduce the original FIxLIP-style pipeline for the standard setup?
2. Does the migrated pipeline run across the planned CLIP / SigLIP / SigLIP2 model presets?
3. Do the migrated segmenter / masker strategies run successfully and expose useful runtime or deviation information?

It is not meant to be an exhaustive Cartesian-product benchmark. The preset suites are staged so
that we can validate the current pipeline without running every possible combination.

## Equivalence vs Coverage

The benchmark separates **strict equivalence** from broader **coverage testing**.

### Strict equivalence

Strict equivalence is only expected for the standard setup:

    segmenter = patch
    masker = crossmodal_mean

This setup preserves the original patch-based player layout and masking behavior. Therefore, the
original FIxLIP-style pipeline and the migrated pipeline should produce matching value-function
outputs under the same image, text, model, coalitions, and random seed.

For this setup, output differences should remain within the configured tolerance.

### Coverage and baseline deviation

Other strategies may intentionally change the player layout or masking behavior, for example:

    segmenter = slic
    masker = vision_mean
    masker = text_attn

These strategies are not strict old-vs-migrated equivalence targets. They are used for coverage,
runtime, and baseline-deviation analysis. Do not interpret non-standard strategies as strict
equivalence failures.

In short:

    patch/crossmodal_mean  -> strict old-vs-migrated equivalence
    other strategies       -> coverage, runtime, and baseline-deviation checks

## Structure

| File | Purpose |
|---|---|
| `benchmark_equivalence.py` | CLI entry point, model/game construction, equivalence evaluation loop. |
| `benchmark_schema.py` | Shared constants, cases, model presets, output fields, and naming helpers. |
| `benchmark_suite.py` | JSON config loading, manifest input expansion, and case/model/strategy normalization. |
| `benchmark_outputs.py` | Resume keys, summary rows, comparison CSVs, and output path helpers. |
| `benchmark_plots.py` | Aggregated strict/models/strategies/crossmodal plots. |
| `plot_results.py` | Plot CLI for existing CSV folders. |

## Input Data

Batch benchmarks use:

    data/input/wds_mscoco_captions_test_100

The input folder must contain `manifest.csv`. Each row must provide:

| Column | Use |
|---|---|
| `filename` | Image file path relative to the input folder. |
| `first_caption` | Default text input used by the benchmark. |
| `caption` | Full caption text preserved in result CSVs as `text_full`. |

The benchmark uses `first_caption` as the model text input to avoid CLIP-style 77-token limits.
The complete `caption` field is still saved for traceability.

For a single image, pass both image and text explicitly:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --case insertion_deletion --input assets\dog_and_hydrant.jpg --text "dog" --num-coalitions 4 --batch-size 2 --tolerance 1e-4 --cuda

## Preset Suites

| Preset | Plot mode | Purpose | Planned runs |
|---|---|---|---:|
| `benchmark_suite.equivalence_strict.json` | `strict` | Strict original-vs-migrated numerical equivalence for `patch/crossmodal_mean`. | 200 |
| `benchmark_suite.equivalence_models.json` | `models` | Model coverage for CLIP, SigLIP, and SigLIP2 model presets. | 800 |
| `benchmark_suite.equivalence_strategies.json` | `strategies` | Migrated strategy coverage plus baseline-deviation comparison. | 800 |
| `benchmark_suite.equivalence_crossmodal.json` | `crossmodal` | Crossmodal image-text player value-function coverage. | 100 |

Running all four current presets gives:

    200 + 800 + 800 + 100 = 1900 planned runs

This is staged full-pipeline validation, not an exhaustive Cartesian-product benchmark. A naive
full product over 8 cases, 100 images, 6 model presets, and 4 strategies would require:

    8 * 100 * 6 * 4 = 19200 run-level combinations

## Recommended First Check

For a first local check, run the strict suite in dry-run mode:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strict.json --dry-run

Then run the strict suite before moving to the larger model and strategy suites:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strict.json

This verifies the standard `patch/crossmodal_mean` path before broader model or strategy coverage.

## Run Benchmarks

Strict equivalence:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strict.json --dry-run
    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strict.json

Model coverage:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_models.json --dry-run
    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_models.json

Strategy coverage:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strategies.json --dry-run
    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strategies.json

Crossmodal coverage:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_crossmodal.json --dry-run
    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_crossmodal.json

Interrupted runs resume automatically. Existing valid `csv/*_comparison.csv` files are skipped.
Use `--force` only when a preset should be recomputed from scratch:

    python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strict.json --force

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

    experiments/validation/equivalence/results/<suite-name>/
      csv/
        summary.csv
        *_coverage_table.csv
        *_comparison.csv
      plots/
        *.png

Generated CSV files can contain many large run-level outputs and should usually remain local.
Commit selected plots only when they are useful for reporting.

Run CSV filenames are intentionally short:

    <case>_<image_stem>_model_<model>_seg_<segmenter>_mask_<masker>_<short_hash>_comparison.csv

Full text, full caption, model parameters, strategy parameters, sampling parameters, outputs,
differences, and runtime fields are stored inside the CSV.

Summary CSVs include comparison-scope fields:

| Field | Meaning |
|---|---|
| `comparison_scope` | `strict_equivalence`, `baseline_deviation`, or `anchor_compatibility`. |
| `reference_name` | Baseline being compared against. |
| `candidate_name` | Migrated pipeline and strategy being evaluated. |
| `equivalence_expected` | Whether exact old-vs-new equivalence is expected for this run. |
| `metric_family` | High-level metric category, such as `output_equivalence` or `baseline_deviation`. |

## Generate Plots

Plots are generated from existing CSVs. The plot mode must match the suite.

Strict:

    python .\experiments\validation\equivalence\plot_results.py --input .\experiments\validation\equivalence\results\benchmark_equivalence_strict_mscoco100_clip_b32\csv --mode strict

Models:

    python .\experiments\validation\equivalence\plot_results.py --input .\experiments\validation\equivalence\results\benchmark_equivalence_models_mscoco100\csv --mode models

Strategies:

    python .\experiments\validation\equivalence\plot_results.py --input .\experiments\validation\equivalence\results\benchmark_equivalence_strategies_mscoco100_clip_b32\csv --mode strategies

Crossmodal:

    python .\experiments\validation\equivalence\plot_results.py --input .\experiments\validation\equivalence\results\benchmark_equivalence_crossmodal_mscoco100_clip_b32\csv --mode crossmodal

## Plot Interpretation

### Strict

Use this suite to answer:

    Does the migrated pipeline reproduce the original pipeline outputs?

Main outputs:

- `equivalence_strict_coverage_table.png`: pass count, max output difference, and runtime by validation case.
- `equivalence_strict_max_output_diff_distribution.png`: run-level max output difference distribution.
- `equivalence_strict_runtime_by_case.png`: original baseline runtime vs migrated pipeline runtime.

### Models

Use this suite to answer:

    Does the pipeline work across multiple vision-language model backbones?

Main outputs:

- `equivalence_models_coverage_table.png`: run count, pass count, max output difference, and runtime by model.
- `equivalence_models_runtime_by_model.png`: original baseline runtime vs migrated pipeline runtime.
- `equivalence_models_max_output_diff_by_model.png`: worst output difference per model.
- `equivalence_models_pass_rate.png`: pass rate per model.

### Strategies

Use this suite to answer:

    Do migrated segmenter/masker strategies run successfully, and what runtime cost do they add?

Do not interpret non-standard strategies as strict-equivalence failures. Only `patch/crossmodal_mean`
preserves the original patch-based player layout. Strategies such as `vision_mean`, `text_attn`,
and `slic` intentionally change masking or player layout, so exact old-vs-new coalition equivalence
is not always meaningful.

Main outputs:

- `equivalence_strategies_coverage_table.png`: completed runs, strict-equivalent runs, baseline-comparable runs, and runtime.
- `equivalence_strategies_baseline_deviation_by_strategy.png`: deviation from baseline only where baseline comparison is meaningful.
- `equivalence_strategies_migrated_runtime_by_strategy.png`: migrated pipeline runtime by strategy.
- `equivalence_strategies_runtime_case_heatmap.png`: migrated runtime by case and strategy.

### Crossmodal

Use this suite to answer:

    Does the benchmark cover image-text coalition games?

Main outputs:

- `equivalence_crossmodal_coverage_table.png`: crossmodal pass and runtime summary.
- `equivalence_crossmodal_max_output_diff_distribution.png`: crossmodal output-difference distribution.
- `equivalence_crossmodal_original_vs_migrated_runtime.png`: original-vs-migrated runtime scatter.
- `equivalence_crossmodal_runtime_by_sample.png`: original and migrated runtime by image-text sample.

## Known Limitations

- `patch/crossmodal_mean` is the strict-equivalence baseline.
- Strategy benchmarks primarily measure coverage and runtime, not explanation quality.
- These plots validate output equivalence and pipeline coverage; they do not evaluate explanation quality.
- CLIP-based original pipelines fail when the model text input exceeds 77 tokens. Manifest-backed runs use `first_caption` as model input and keep the full caption only for traceability.
- Large CSV outputs and generated caches should remain local. Commit selected plots only when they are needed for reporting.