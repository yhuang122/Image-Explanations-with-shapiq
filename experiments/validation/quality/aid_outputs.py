"""CSV output helpers for AID quality benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

from aid_schema import PROJECT_ROOT, RUN_KEY_FIELDS, SUMMARY_FIELDS, Sample, slug

METADATA_DIRNAME = "metadata"
CSV_DIRNAME = "csv"
PLOTS_DIRNAME = "plots"
INTERACTION_VALUES_DIRNAME = "interaction_values"
SUMMARY_FILENAME = "aid_summary.csv"
CURVES_FILENAME = "aid_curves.csv"


def output_paths(output_dir: Path) -> dict[str, Path]:
    metadata_dir = output_dir / METADATA_DIRNAME
    csv_dir = output_dir / CSV_DIRNAME
    plots_dir = output_dir / PLOTS_DIRNAME
    interaction_values_dir = output_dir / INTERACTION_VALUES_DIRNAME
    return {
        "metadata_dir": metadata_dir,
        "csv_dir": csv_dir,
        "plots_dir": plots_dir,
        "interaction_values_dir": interaction_values_dir,
        "summary_csv": csv_dir / SUMMARY_FILENAME,
        "curves_csv": csv_dir / CURVES_FILENAME,
    }


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
    return output_paths(output_dir)["interaction_values_dir"] / filename


def run_key_from_values(values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(values.get(field, "")) for field in RUN_KEY_FIELDS)


def existing_run_keys(summary_path: Path) -> set[tuple[str, ...]]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as file:
        return {run_key_from_values(row) for row in csv.DictReader(file) if row.get("status") == "completed"}


def keep_completed_summary_rows(summary_path: Path) -> None:
    """Drop stale failed rows before resuming a suite."""
    if not summary_path.exists():
        return
    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    completed = [row for row in rows if row.get("status") == "completed"]
    if len(completed) == len(rows):
        return
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def package_version(package_name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "shapiq": package_version("shapiq"),
        },
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as error:
        info["torch_error"] = f"{type(error).__name__}: {error}"
    return info


def write_run_metadata(output_dir: Path, args, suite: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    metadata_dir = output_paths(output_dir)["metadata_dir"]
    paths = {
        "benchmark_plan": metadata_dir / "benchmark_plan.json",
        "suite_normalized": metadata_dir / "suite_normalized.json",
        "cli_args": metadata_dir / "cli_args.json",
        "environment": metadata_dir / "environment.json",
    }
    write_json(paths["benchmark_plan"], plan)
    write_json(paths["suite_normalized"], suite)
    write_json(paths["cli_args"], vars(args))
    write_json(paths["environment"], environment_info())

    config_path = None
    if args.config:
        raw_config_path = Path(args.config)
        config_path = raw_config_path if raw_config_path.is_absolute() else PROJECT_ROOT / raw_config_path
        config_path = config_path.resolve()
    if config_path and config_path.exists():
        paths["config_used"] = metadata_dir / "config_used.json"
        shutil.copyfile(config_path, paths["config_used"])
    return {name: str(path) for name, path in paths.items()}


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def segmenter_params(strategy: dict[str, Any]) -> dict[str, Any]:
    if strategy["segmenter_strategy"] == "slic":
        return {
            "n_segments": strategy["slic_n_segments"],
            "compactness": strategy["slic_compactness"],
            "sigma": strategy["slic_sigma"],
        }
    if strategy["segmenter_strategy"] == "gradient_guided":
        return {"n_segments": strategy["gradient_guided_n_segments"]}
    return {}


def masker_params(strategy: dict[str, Any]) -> dict[str, Any]:
    if strategy["masker_strategy"] == "vision_blur":
        return {"sigma": strategy["vision_blur_sigma"]}
    if strategy["masker_strategy"] == "crossmodal_blur":
        return {"sigma": strategy["crossmodal_blur_sigma"]}
    return {}


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
        "segmenter_params": compact_json(segmenter_params(strategy)),
        "masker_strategy": strategy["masker_strategy"],
        "masker_params": compact_json(masker_params(strategy)),
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
    "CURVES_FILENAME",
    "CSV_DIRNAME",
    "INTERACTION_VALUES_DIRNAME",
    "METADATA_DIRNAME",
    "PLOTS_DIRNAME",
    "SUMMARY_FILENAME",
    "add_curve_context",
    "append_rows",
    "base_summary_row",
    "existing_run_keys",
    "interaction_value_path",
    "keep_completed_summary_rows",
    "output_paths",
    "run_key_from_values",
    "summarize_failure",
    "write_run_metadata",
)
