"""CSV output helpers for AID quality benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from aid_schema import CURVE_FIELDS, RUN_KEY_FIELDS, SUMMARY_FIELDS, Sample, slug


def run_id(sample: Sample, model_case: dict[str, Any], strategy: dict[str, Any], method: dict[str, Any]) -> str:
    token = "_".join(
        [
            sample.sample_id,
            model_case["model_preset"],
            strategy["strategy_name"],
            method["method_name"],
            f"order{method['order']}",
            f"budget{method['budget']}",
            method["approximation_type"],
        ]
    )
    return slug(token, max_length=96)


def interaction_value_path(output_dir: Path, sample: Sample, model_case: dict[str, Any], strategy: dict[str, Any], method: dict[str, Any]) -> Path:
    context = "|".join(
        str(value)
        for value in (
            sample.sample_id,
            sample.path,
            model_case["model_name"],
            strategy["strategy_name"],
            method["method_name"],
            method["mode"],
            method["order"],
            method["budget"],
            method["approximation_type"],
        )
    )
    short_hash = hashlib.sha1(context.encode("utf-8")).hexdigest()[:10]
    filename = f"{run_id(sample, model_case, strategy, method)}_{short_hash}.json"
    return output_dir / "interaction_values" / filename


def run_key_from_values(values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(values.get(field, "")) for field in RUN_KEY_FIELDS)


def existing_run_keys(summary_path: Path) -> set[tuple[str, ...]]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as file:
        return {run_key_from_values(row) for row in csv.DictReader(file) if row.get("status") == "completed"}


def append_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def base_summary_row(
    sample: Sample,
    model_case: dict[str, Any],
    strategy: dict[str, Any],
    method: dict[str, Any],
    interaction_path: Path,
    device,
    defaults: dict[str, Any],
    model_load_runtime_s: float,
    game_build_runtime_s: float | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id(sample, model_case, strategy, method),
        "status": "completed",
        "passed": True,
        "error_message": "",
        "quality_scope": "explanation_quality",
        "quality_metric": "aid_area_between_insertion_deletion_curves",
        "sample_id": sample.sample_id,
        "sample_index": sample.sample_index,
        "source_dataset": sample.source_dataset,
        "source_key": sample.source_key,
        "input_path": str(sample.path),
        "text": sample.text,
        "text_full": sample.text_full,
        "text_source": sample.text_source,
        "model_preset": model_case["model_preset"],
        "model_name": model_case["model_name"],
        "strategy_name": strategy["strategy_name"],
        "segmenter_strategy": strategy["segmenter_strategy"],
        "segmenter_params": compact_json(
            {
                "slic_n_segments": strategy["slic_n_segments"],
                "slic_compactness": strategy["slic_compactness"],
                "slic_sigma": strategy["slic_sigma"],
                "gradient_guided_n_segments": strategy["gradient_guided_n_segments"],
            }
        ),
        "masker_strategy": strategy["masker_strategy"],
        "masker_params": compact_json({"strategy": strategy["masker_strategy"]}),
        "explainer_name": method["explainer_name"],
        "method_name": method["method_name"],
        "mode": method["mode"],
        "sampler_name": method["sampler_name"],
        "sampler_p": method["sampler_p"],
        "order": method["order"],
        "approximation_type": method["approximation_type"],
        "explanation_budget": method["budget"],
        "interaction_value_path": str(interaction_path),
        "device": str(device),
        "use_amp": defaults["use_amp"],
        "batch_size": defaults["batch_size"],
        "curve_points": defaults["curve_points"],
        "random_state": defaults["random_state"],
        "model_load_runtime_s": model_load_runtime_s,
        "game_build_runtime_s": game_build_runtime_s,
    }


def summarize_failure(row: dict[str, Any], error: Exception, total_runtime_s: float) -> dict[str, Any]:
    row.update(
        status="failed",
        passed=False,
        error_message=f"{type(error).__name__}: {error}",
        total_runtime_s=total_runtime_s,
    )
    return row


def add_curve_context(row: dict[str, Any], curve: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "quality_scope": row["quality_scope"],
        "sample_id": row["sample_id"],
        "sample_index": row["sample_index"],
        "input_path": row["input_path"],
        "text": row["text"],
        "model_preset": row["model_preset"],
        "model_name": row["model_name"],
        "strategy_name": row["strategy_name"],
        "method_name": row["method_name"],
        "mode": row["mode"],
        "order": row["order"],
        **curve,
    }


__all__ = (
    "CURVE_FIELDS",
    "SUMMARY_FIELDS",
    "add_curve_context",
    "append_rows",
    "base_summary_row",
    "existing_run_keys",
    "interaction_value_path",
    "run_key_from_values",
    "summarize_failure",
)
