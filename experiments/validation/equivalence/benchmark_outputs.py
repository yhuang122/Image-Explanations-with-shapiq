"""Output helpers for validation benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark_schema import (
    BENCHMARK_SUMMARY_FIELDS,
    COMPARISON_SCOPE_FIELDS,
    RESULTS_DIR,
    ROW_FIELDS,
    SUMMARY_FIELDS,
    slug,
)
from benchmark_plots import CSV_DIRNAME, PLOT_MODES, PLOTS_DIRNAME, write_benchmark_plots


RUNS_DIRNAME = CSV_DIRNAME


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


def true_text(value) -> bool:
    return str(value).strip().lower() == "true"


def fill_legacy_scope_fields(report: dict) -> None:
    candidate_name = (
        f"{report.get('migrated_pipeline', '')}:"
        f"{report.get('segmenter_strategy', '')}/{report.get('masker_strategy', '')}"
    )
    if report.get("comparison_scope"):
        return
    if true_text(report.get("strict_equivalence")):
        report.update(
            comparison_scope="strict_equivalence",
            reference_name=report.get("original_pipeline", ""),
            candidate_name=candidate_name,
            equivalence_expected=True,
            metric_family="output_equivalence",
        )
    elif true_text(report.get("coalition_comparison_available")):
        report.update(
            comparison_scope="baseline_deviation",
            reference_name=report.get("original_pipeline", ""),
            candidate_name=candidate_name,
            equivalence_expected=False,
            metric_family="baseline_deviation",
        )
    else:
        report.update(
            comparison_scope="anchor_compatibility",
            reference_name=f"{report.get('original_pipeline', '')}:empty_full_anchors",
            candidate_name=candidate_name,
            equivalence_expected=False,
            metric_family="anchor_consistency",
        )


def read_existing_report(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        required_fields = set(SUMMARY_FIELDS) - set(COMPARISON_SCOPE_FIELDS)
        if not required_fields.issubset(fieldnames):
            return None
        row = next(reader, None)
    if not row:
        return None

    report = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    fill_legacy_scope_fields(report)
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

def write_benchmark_summary(output_dir: Path, reports: list[dict], plot_mode: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / CSV_DIRNAME
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_path = csv_dir / "summary.csv"
    tmp_summary_path = summary_path.with_suffix(f"{summary_path.suffix}.tmp")

    with tmp_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BENCHMARK_SUMMARY_FIELDS)
        writer.writeheader()
        for report in reports:
            row = {field: report.get(field, "") for field in SUMMARY_FIELDS}
            row["result_csv"] = report["result_paths"]["csv"]
            writer.writerow(row)
    tmp_summary_path.replace(summary_path)

    result_paths = {"summary_csv": str(summary_path)}
    if plot_mode is not None:
        result_paths.update(write_benchmark_plots(output_dir, reports, plot_mode))
    return result_paths


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
