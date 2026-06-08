"""Output helpers for validation benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark_schema import (
    BENCHMARK_SUMMARY_FIELDS,
    RESULTS_DIR,
    ROW_FIELDS,
    SUMMARY_FIELDS,
    slug,
)


PLOTS_DIRNAME = "plots"
RUNS_DIRNAME = "runs"


def active_params(config) -> dict:
    params = config.active_params
    return asdict(params) if params is not None else {}


def compact_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_report_float(report: dict, field: str):
    value = report.get(field)
    if value in (None, ""):
        return None
    return float(value)


def max_output_diff_metric(report: dict) -> float:
    coalition_diff = parse_report_float(report, "coalition_max_abs_output_diff")
    if coalition_diff is not None:
        return coalition_diff
    return float(report["empty_full_anchor_max_abs_output_diff"])


def mean_output_diff_metric(report: dict) -> float:
    coalition_diff = parse_report_float(report, "coalition_mean_abs_output_diff")
    if coalition_diff is not None:
        return coalition_diff
    return float(report["empty_full_anchor_mean_abs_output_diff"])


def segmenter_token(config) -> str:
    if config.strategy == "slic":
        params = config.slic
        return f"slic{params.n_segments}_c{params.compactness:g}_s{params.sigma:g}"
    if config.strategy == "gradient_guided":
        n_segments = config.gradient_guided.n_segments
        return f"gradient_guided{n_segments if n_segments is not None else 'auto'}"
    return config.strategy


def model_token(case: dict) -> str:
    if case["model_preset"] == "custom":
        return slug(case["model_name"], max_length=64)
    return slug(case["model_preset"])


def run_hash(case: dict, length: int = 12) -> str:
    payload = {
        "case": case["case"],
        "input_path": str(case["input_path"]),
        "text": case["text"],
        "model_preset": case["model_preset"],
        "model_name": case["model_name"],
        "segmenter": case["segmenter_config"].strategy,
        "segmenter_params": active_params(case["segmenter_config"]),
        "masker": case["masker_config"].strategy,
        "masker_params": active_params(case["masker_config"]),
        "random_state": case["random_state"],
        "num_coalitions": case["num_coalitions"],
        "batch_size": case["batch_size"],
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def run_name(case: dict) -> str:
    return (
        f"{case['case']}_{case['input_path'].stem}_"
        f"model_{model_token(case)}_"
        f"seg_{slug(segmenter_token(case['segmenter_config']))}_"
        f"mask_{slug(case['masker_config'].strategy)}_"
        f"{run_hash(case)}"
    )


def suite_output_dir(name: str) -> Path:
    return RESULTS_DIR / name


def comparison_csv_path(output_dir: Path, case: dict) -> Path:
    return output_dir / RUNS_DIRNAME / f"{run_name(case)}_comparison.csv"


def read_existing_report(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if not set(SUMMARY_FIELDS).issubset(fieldnames):
            return None
        row = next(reader, None)
    if not row:
        return None

    report = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    report["result_paths"] = {"csv": str(csv_path)}
    report["_resumed"] = True
    return report


def maybe_float(value) -> float | str:
    if value is None:
        return ""
    return float(value)


def value_row(row_type: str, index, old_output, new_output: float, diff, **coalition_fields) -> dict:
    row = {"row_type": row_type, "coalition_index": index, **coalition_fields}
    row.update(
        original_pipeline_coalition_output=maybe_float(old_output),
        migrated_pipeline_coalition_output=float(new_output),
        abs_coalition_output_diff=maybe_float(diff),
    )
    return row


def iter_anchor_rows(report: dict):
    yield value_row(
        "anchor_empty",
        "empty",
        report["original_pipeline_empty_coalition_output"],
        report["migrated_pipeline_empty_coalition_output"],
        report["abs_empty_coalition_output_diff"],
        image_coalition_index="",
        text_coalition_index="",
        coalition_size=0,
        image_coalition_size=0,
        text_coalition_size=0,
    )
    yield value_row(
        "anchor_full",
        "full",
        report["original_pipeline_full_coalition_output"],
        report["migrated_pipeline_full_coalition_output"],
        report["abs_full_coalition_output_diff"],
        image_coalition_index="",
        text_coalition_index="",
        coalition_size=report["n_players"],
        image_coalition_size=report["n_players_image"],
        text_coalition_size=report["n_players_text"],
    )


def iter_result_rows(inputs: dict, old_pipeline_values: np.ndarray | None, new_pipeline_values: np.ndarray):
    diffs = None if old_pipeline_values is None else np.abs(old_pipeline_values - new_pipeline_values)
    if inputs["comparison_type"] == "crossmodal":
        image_coalitions = inputs["image_coalitions"]
        text_coalitions = inputs["text_coalitions"]
        for index, (image_index, text_index) in enumerate(np.ndindex(len(image_coalitions), len(text_coalitions))):
            image_coalition = image_coalitions[image_index]
            text_coalition = text_coalitions[text_index]
            yield value_row(
                "coalition",
                index,
                None if old_pipeline_values is None else old_pipeline_values[index],
                new_pipeline_values[index],
                None if diffs is None else diffs[index],
                image_coalition_index=image_index,
                text_coalition_index=text_index,
                coalition_size=int(np.sum(image_coalition) + np.sum(text_coalition)),
                image_coalition_size=int(np.sum(image_coalition)),
                text_coalition_size=int(np.sum(text_coalition)),
            )
        return

    old_outputs = [None] * len(new_pipeline_values) if old_pipeline_values is None else old_pipeline_values
    differences = [None] * len(new_pipeline_values) if diffs is None else diffs
    for index, (coalition, old_output, new_output, diff) in enumerate(
        zip(inputs["coalitions"], old_outputs, new_pipeline_values, differences)
    ):
        yield value_row(
            "coalition",
            index,
            old_output,
            new_output,
            diff,
            image_coalition_index="",
            text_coalition_index="",
            coalition_size=int(np.sum(coalition)),
            image_coalition_size="",
            text_coalition_size="",
        )


def write_results(
    output_dir: Path,
    case: dict,
    report: dict,
    inputs: dict,
    old_pipeline_values: np.ndarray | None,
    new_pipeline_values: np.ndarray,
) -> dict:
    csv_path = comparison_csv_path(output_dir, case)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS + ROW_FIELDS)
        writer.writeheader()
        for anchor_row in iter_anchor_rows(report):
            row = {field: report[field] for field in SUMMARY_FIELDS}
            row.update(anchor_row)
            writer.writerow(row)
        for result_row in iter_result_rows(inputs, old_pipeline_values, new_pipeline_values):
            row = {field: report[field] for field in SUMMARY_FIELDS}
            row.update(result_row)
            writer.writerow(row)

    tmp_path.replace(csv_path)
    return {"csv": str(csv_path)}


def write_top_bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    tolerance=None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    upper_candidates = list(values)
    if tolerance is not None:
        upper_candidates.append(float(tolerance))
    upper_limit = max(upper_candidates or [1.0]) * 1.2
    if upper_limit == 0:
        upper_limit = 1.0
    lower_limit = -max(upper_limit * 0.08, 1e-6)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5.0))
    ax.bar(labels, values)
    if tolerance is not None:
        ax.axhline(float(tolerance), color="red", linestyle="--", linewidth=1, label="tolerance")
        ax.legend()
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_histogram(path: Path, values: list[float], xlabel: str, title: str, tolerance=None) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values) or 1)))))
    if tolerance is not None:
        ax.axvline(float(tolerance), color="red", linestyle="--", linewidth=1, label="tolerance")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("run count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def group_mean(reports: list[dict], row_field: str, col_field: str, value_field: str) -> tuple[list[str], list[str], np.ndarray]:
    grouped = defaultdict(list)
    rows = []
    cols = []
    for report in reports:
        value = parse_report_float(report, value_field)
        if value is None:
            continue
        row = str(report.get(row_field) or "unknown")
        col = str(report.get(col_field) or "unknown")
        grouped[(row, col)].append(value)
        if row not in rows:
            rows.append(row)
        if col not in cols:
            cols.append(col)

    matrix = np.full((len(rows), len(cols)), np.nan)
    for row_index, row in enumerate(rows):
        for col_index, col in enumerate(cols):
            values = grouped.get((row, col))
            if values:
                matrix[row_index, col_index] = float(np.mean(values))
    return rows, cols, matrix


def write_heatmap(path: Path, rows: list[str], cols: list[str], matrix: np.ndarray, title: str, colorbar_label: str) -> None:
    if not rows or not cols:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.3), max(4, len(rows) * 0.7)))
    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked_matrix, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    for row_index in range(len(rows)):
        for col_index in range(len(cols)):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f"{value:.3g}", ha="center", va="center", color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report_top_label(report: dict) -> str:
    return (
        f"{Path(report['input_path']).stem}\n"
        f"{report['case']} | {report['model_preset']}\n"
        f"{report['strategy_name']}"
    )


def write_benchmark_summary(output_dir: Path, reports: list[dict]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    plots_dir = output_dir / PLOTS_DIRNAME
    histogram_path = plots_dir / "max_output_diff_distribution.png"
    top_diff_path = plots_dir / "top_max_output_diff.png"
    diff_heatmap_path = plots_dir / "mean_max_output_diff_heatmap.png"
    runtime_heatmap_path = plots_dir / "mean_runtime_heatmap.png"
    tmp_summary_path = summary_path.with_suffix(f"{summary_path.suffix}.tmp")

    with tmp_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BENCHMARK_SUMMARY_FIELDS)
        writer.writeheader()
        for report in reports:
            row = {field: report.get(field, "") for field in SUMMARY_FIELDS}
            row["result_csv"] = report["result_paths"]["csv"]
            writer.writerow(row)
    tmp_summary_path.replace(summary_path)

    diff_values = [max_output_diff_metric(report) for report in reports]
    write_histogram(
        histogram_path,
        diff_values,
        "max output diff: original vs migrated",
        "Max output diff distribution",
        tolerance=max(float(report["tolerance"]) for report in reports),
    )

    top_reports = sorted(reports, key=max_output_diff_metric, reverse=True)[: min(20, len(reports))]
    write_top_bar_plot(
        top_diff_path,
        [report_top_label(report) for report in top_reports],
        [max_output_diff_metric(report) for report in top_reports],
        "max output diff: original vs migrated",
        "Top max output diff runs",
        tolerance=max(float(report["tolerance"]) for report in reports),
    )

    rows, cols, matrix = group_mean(reports, "case", "model_preset", "benchmark_metric_max_output_diff")
    write_heatmap(
        diff_heatmap_path,
        rows,
        cols,
        matrix,
        "Mean max output diff by case and model",
        "mean max output diff",
    )
    rows, cols, matrix = group_mean(reports, "case", "strategy_name", "migrated_pipeline_runtime_s")
    write_heatmap(
        runtime_heatmap_path,
        rows,
        cols,
        matrix,
        "Mean migrated runtime by case and strategy",
        "mean runtime (s)",
    )
    return {
        "summary_csv": str(summary_path),
        "max_output_diff_distribution_plot": str(histogram_path),
        "top_max_output_diff_plot": str(top_diff_path),
        "mean_max_output_diff_heatmap": str(diff_heatmap_path),
        "mean_runtime_heatmap": str(runtime_heatmap_path),
    }


def write_original_summary(output_dir: Path, reports: list[dict]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "original_pipeline.csv"
    tmp_summary_path = summary_path.with_suffix(f"{summary_path.suffix}.tmp")
    fieldnames = (
        "case", "input_path", "model_preset", "model_name", "text", "text_full", "text_source", "comparison_type",
        "n_players", "n_players_image", "n_players_text", "num_coalitions", "batch_size",
        "random_state", "model_load_runtime_s", "original_game_build_runtime_s",
        "original_anchor_runtime_s", "original_pipeline_runtime_s", "total_runtime_s",
        "empty_coalition_output", "full_coalition_output", "original_pipeline_outputs",
    )
    with tmp_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({field: report.get(field, "") for field in fieldnames})
    tmp_summary_path.replace(summary_path)
    return {"original_pipeline_csv": str(summary_path)}
