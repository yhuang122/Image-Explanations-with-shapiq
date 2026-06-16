# Validation Benchmarks

This folder contains two benchmark tracks for the migrated image-explanation pipeline.

| Folder | Purpose | Main question | Main outputs |
|---|---|---|---|
| `equivalence/` | A2/A3 numerical equivalence and coverage | Does the migrated game reproduce or cover the old FIxLIP-style game behavior? | max absolute output difference, pass rate, runtime, coverage tables |
| `quality/` | Explanation quality evaluation | Are the generated explanations useful under deletion-curve evaluation? | AID score, deletion curves, runtime tradeoff, qualitative FIxLIP-style figures |

## Overview

The validation work is split into two layers:

```text
Layer 1: Equivalence and coverage
same input -> old pipeline output vs migrated pipeline output

Layer 2: Explanation quality
migrated pipeline -> InteractionValues -> AID deletion curves
```

Use `equivalence/` for A2/A3 numerical validation. Use `quality/` when evaluating whether the
explanations themselves are good, not only whether the pipeline runs.

## Common Commands

Run strict A2 equivalence:

```powershell
python .\experiments\validation\equivalence\benchmark_equivalence.py --config .\experiments\validation\equivalence\benchmark_suite.equivalence_strict.json
```

Run the AID quality preview suite:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_preview_assets.json
```

Run the full AID quality suite:

```powershell
python .\experiments\validation\quality\benchmark_aid.py --config .\experiments\validation\quality\benchmark_suite.aid_clip_siglip_mscoco100.json
```

Regenerate equivalence plots from existing CSVs:

```powershell
python .\experiments\validation\equivalence\plot_results.py --input .\experiments\validation\equivalence\results\benchmark_equivalence_strict_mscoco100_clip_b32\csv --mode strict
```

Regenerate AID quality plots from existing CSVs:

```powershell
python .\experiments\validation\quality\plot_results.py --input .\experiments\validation\quality\results\benchmark_aid_clip_siglip_mscoco100
```

## Run Counts

| Benchmark | Planned runs |
|---|---:|
| `equivalence_strict` | 800 |
| `equivalence_models` | 1000 |
| `equivalence_strategies` | 800 |
| `equivalence_crossmodal` | 100 |
| Core equivalence total | 2700 |
| Optional insertion-deletion CLIP + SigLIP strategies part1 + part2 | 2000 |
| AID quality preview suite | 32 |
| AID quality CLIP + SigLIP full segmenter/masker suite | 14400 |

The equivalence suites validate value-function behavior and coverage. The AID suite validates
explanation quality and is usually the better result to show when discussing whether the full
pipeline gives meaningful explanations.
