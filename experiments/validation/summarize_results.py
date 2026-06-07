from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from benchmark_outputs import (
    group_mean,
    write_heatmap,
    write_histogram,
    write_top_bar_plot,
)
from benchmark_schema import PROJECT_ROOT, RESULTS_DIR, SUMMARY_FIELDS as BENCHMARK_FIELDS

UNIFIED_REQUIRED_FIELDS = {
    "case",
    "row_type",
    "original_pipeline_coalition_output",
    "migrated_pipeline_coalition_output",
    "abs_coalition_output_diff",
}
SUMMARY_FIELDS = ("result_name", *BENCHMARK_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validation comparison CSV files.")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "summary.csv"),
        help="Summary CSV path. Defaults to experiments/validation/results/summary.csv.",
    )
    parser.add_argument(
        "--plot",
        default=str(RESULTS_DIR / "summary_plots"),
        help="Summary plot directory. Defaults to experiments/validation/results/summary_plots.",
    )
    return parser.parse_args()


def result_name(path: Path) -> str:
    return path.stem.removesuffix("_comparison")


def text_from_result_name(name: str) -> str:
    if "_text_" not in name:
        return ""
    text_part = name.split("_text_", 1)[1]
    text_part = text_part.split("_seg_", 1)[0]
    return text_part.rsplit("_rs", 1)[0].replace("_", " ")


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
    values["text"] = values["text"] or text_from_result_name(name)
    return {
        "result_name": name,
        **values,
    }


def is_unified_benchmark_csv(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as file:
        fieldnames = csv.DictReader(file).fieldnames or []
    return UNIFIED_REQUIRED_FIELDS.issubset(fieldnames)


def resolve_output_path(path: str) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_summary(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(rows: list[dict], output_dir: Path) -> None:
    plot_rows = [
        (
            row,
            parse_float(
                row.get("benchmark_metric_max_output_diff", "")
                or row.get("coalition_max_abs_output_diff", "")
                or row.get("empty_full_anchor_max_abs_output_diff", "")
                or row.get("max_abs_diff", "")
            ),
        )
        for row in rows
    ]
    plot_rows = [(row, value) for row, value in plot_rows if value is not None]
    if not plot_rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_histogram(
        output_dir / "max_output_diff_distribution.png",
        [value for _, value in plot_rows],
        "max output diff: original vs migrated",
        "All-suite max output diff distribution",
        tolerance=max(float(row["tolerance"]) for row, _ in plot_rows),
    )

    top_rows = sorted(plot_rows, key=lambda item: item[1], reverse=True)[: min(20, len(plot_rows))]
    write_top_bar_plot(
        output_dir / "top_max_output_diff.png",
        [plot_label(row) for row, _ in top_rows],
        [value for _, value in top_rows],
        "max output diff: original vs migrated",
        "All-suite top max output diff runs",
        tolerance=max(float(row["tolerance"]) for row, _ in plot_rows),
    )

    case_model_rows, case_model_cols, case_model_matrix = group_mean(
        rows, "case", "model_preset", "benchmark_metric_max_output_diff"
    )
    write_heatmap(
        output_dir / "mean_max_output_diff_by_case_model.png",
        case_model_rows,
        case_model_cols,
        case_model_matrix,
        "Mean max output diff by case and model",
        "mean max output diff",
    )

    case_strategy_rows, case_strategy_cols, case_strategy_matrix = group_mean(
        rows, "case", "strategy_name", "migrated_pipeline_runtime_s"
    )
    write_heatmap(
        output_dir / "mean_runtime_by_case_strategy.png",
        case_strategy_rows,
        case_strategy_cols,
        case_strategy_matrix,
        "Mean migrated runtime by case and strategy",
        "mean runtime (s)",
    )


def plot_label(row: dict) -> str:
    name = row["result_name"]
    if row.get("text"):
        name = name.split("_text_", 1)[0]
        strategy = strategy_label(row)
        return f"{name}\ntext={row['text']}{strategy}"
    return name


def strategy_label(row: dict) -> str:
    strategy_name = row.get("strategy_name")
    if strategy_name:
        return f"\nstrategy={strategy_name}"
    segmenter = row.get("segmenter_strategy")
    masker = row.get("masker_strategy")
    if not segmenter and not masker:
        return ""
    if segmenter == "patch" and masker == "crossmodal_mean":
        return ""
    return f"\nseg={segmenter or '?'} mask={masker or '?'}"


def parse_float(value: str):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> int:
    args = parse_args()
    paths = [path for path in sorted(RESULTS_DIR.rglob("*_comparison.csv")) if is_unified_benchmark_csv(path)]
    if not paths:
        raise FileNotFoundError(f"No unified benchmark CSV files found in {RESULTS_DIR}")

    rows = [read_summary_row(path) for path in paths]
    output_path = resolve_output_path(args.output)
    plot_path = resolve_output_path(args.plot)
    write_summary(rows, output_path)
    write_plots(rows, plot_path)

    print(output_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
