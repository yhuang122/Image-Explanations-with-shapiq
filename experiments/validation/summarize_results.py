from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark_outputs import CSV_DIRNAME, PLOT_MODES, PLOTS_DIRNAME, write_benchmark_plots
from benchmark_schema import PROJECT_ROOT, SUMMARY_FIELDS as BENCHMARK_FIELDS

UNIFIED_REQUIRED_FIELDS = {
    "case",
    "row_type",
    "original_pipeline_coalition_output",
    "migrated_pipeline_coalition_output",
    "abs_coalition_output_diff",
}
SUMMARY_FIELDS = ("result_name", *BENCHMARK_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot benchmark CSV files.")
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing benchmark comparison CSV files. A suite folder or its csv folder both work.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=PLOT_MODES,
        help="Plot mode: strict, models, strategies, or crossmodal.",
    )
    parser.add_argument(
        "--output",
        help="Summary CSV path. Defaults to <suite-folder>/csv/summary.csv.",
    )
    parser.add_argument(
        "--plot",
        help="Plot directory. Defaults to <suite-folder>/plots.",
    )
    return parser.parse_args()


def result_name(path: Path) -> str:
    return path.stem.removesuffix("_comparison")


def read_summary_row(path: Path) -> dict:
    name = result_name(path)
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    values = {field: row.get(field, "") for field in SUMMARY_FIELDS if field != "result_name"}
    values["benchmark_metric_max_output_diff"] = (
        values["benchmark_metric_max_output_diff"]
        or values["coalition_max_abs_output_diff"]
        or values["empty_full_anchor_max_abs_output_diff"]
    )
    values["benchmark_metric_mean_output_diff"] = (
        values["benchmark_metric_mean_output_diff"]
        or values["coalition_mean_abs_output_diff"]
        or values["empty_full_anchor_mean_abs_output_diff"]
    )
    return {"result_name": name, **values}


def is_unified_benchmark_csv(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as file:
        fieldnames = csv.DictReader(file).fieldnames or []
    return UNIFIED_REQUIRED_FIELDS.issubset(fieldnames)


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def suite_root(input_path: Path) -> Path:
    return input_path.parent if input_path.name in {"runs", CSV_DIRNAME} else input_path


def benchmark_csv_paths(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(input_path.rglob("*_comparison.csv"))
    return [path for path in paths if is_unified_benchmark_csv(path)]


def output_paths(args: argparse.Namespace, input_path: Path) -> tuple[Path, Path]:
    root = suite_root(input_path)
    summary_path = resolve_path(args.output) if args.output else root / CSV_DIRNAME / "summary.csv"
    plot_path = resolve_path(args.plot) if args.plot else root / PLOTS_DIRNAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)
    return summary_path, plot_path


def write_summary(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    paths = benchmark_csv_paths(input_path)
    if not paths:
        raise FileNotFoundError(f"No benchmark comparison CSV files found in {input_path}")

    rows = [read_summary_row(path) for path in paths]
    summary_path, plot_path = output_paths(args, input_path)
    write_summary(rows, summary_path)
    root = suite_root(input_path)
    plot_outputs = write_benchmark_plots(root, rows, args.mode, plots_dir=plot_path, csv_dir=summary_path.parent)

    print(summary_path)
    print(plot_path)
    print(f"rows={len(rows)}")
    print(f"mode={plot_outputs['plot_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
