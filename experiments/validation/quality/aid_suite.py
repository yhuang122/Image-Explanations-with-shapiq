"""JSON preset and input expansion for AID quality benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from aid_schema import (
    DEFAULT_INPUT_DIR,
    DEFAULT_METHODS,
    DEFAULT_MODEL_PRESET,
    DEFAULT_RANDOM_STATE,
    DEFAULT_STRATEGIES,
    IMAGE_SUFFIXES,
    MANIFEST_FILENAME,
    MASKER_CHOICES,
    MODEL_PRESETS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEGMENTER_CHOICES,
    Sample,
    slug,
)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path_value: str) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    if path.suffix.lower() != ".json":
        raise ValueError("Only JSON benchmark configs are supported.")
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest_rows(input_dir: Path) -> list[dict[str, str]]:
    manifest_path = input_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch AID input requires manifest.csv: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def sample_from_manifest_row(input_dir: Path, row: dict[str, str], text_column: str) -> Sample:
    filename = row.get("filename", "").strip()
    if not filename:
        raise ValueError("Manifest rows must include a filename column.")
    image_path = input_dir / filename
    if not image_path.exists():
        raise FileNotFoundError(f"Manifest image file not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Manifest file is not a supported image: {image_path}")

    text = row.get(text_column, "").strip()
    text_full = row.get("caption", "").strip()
    if not text:
        raise ValueError(f"Manifest row has no text in column '{text_column}'.")

    raw_index = row.get("index", "").strip()
    sample_index = int(raw_index) if raw_index.isdigit() else 0
    source_key = row.get("source_key", "").strip()
    return Sample(
        sample_id=source_key or image_path.stem,
        sample_index=sample_index,
        path=image_path,
        text=text,
        text_full=text_full or text,
        text_source=text_column,
        source_dataset=row.get("source_dataset", "").strip(),
        source_key=source_key,
    )


def resolve_samples(input_value: str | dict[str, Any] | None, text: str | None, text_column: str) -> list[Sample]:
    input_text = text
    if isinstance(input_value, dict):
        input_text = input_value.get("text", text)
        input_value = input_value.get("path")
    input_path = DEFAULT_INPUT_DIR if input_value is None else resolve_project_path(input_value)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        if not input_text:
            raise ValueError("Single-image AID input requires --text.")
        return [
            Sample(
                sample_id=input_path.stem,
                sample_index=0,
                path=input_path,
                text=input_text,
                text_full=input_text,
                text_source="config_text" if isinstance(input_value, str) and text is None else "cli_text",
            )
        ]
    if input_text:
        raise ValueError("Batch AID input reads manifest.csv; --text is only for single-image input.")
    return [sample_from_manifest_row(input_path, row, text_column) for row in read_manifest_rows(input_path)]


def normalize_defaults(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    defaults = {
        "batch_size": 64,
        "curve_points": 101,
        "cuda": False,
        "use_amp": False,
        "random_state": DEFAULT_RANDOM_STATE,
        "text_column": "first_caption",
    }
    defaults.update(config.get("defaults", {}))
    for key in ("batch_size", "curve_points", "random_state"):
        cli_value = getattr(args, key)
        if cli_value is not None:
            defaults[key] = cli_value
    for key in ("cuda", "use_amp"):
        cli_value = getattr(args, key)
        if cli_value is not None:
            defaults[key] = bool(cli_value)
    if args.text_column is not None:
        defaults["text_column"] = args.text_column
    defaults["batch_size"] = int(defaults["batch_size"])
    defaults["curve_points"] = int(defaults["curve_points"])
    defaults["random_state"] = int(defaults["random_state"])
    defaults["cuda"] = bool(defaults["cuda"])
    defaults["use_amp"] = bool(defaults["use_amp"])
    if defaults["text_column"] not in ("first_caption", "caption"):
        raise ValueError("defaults.text_column must be first_caption or caption.")
    if defaults["curve_points"] < 2:
        raise ValueError("curve_points must be at least 2.")
    return defaults


def normalize_models(raw_models: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.model_name:
        return [{"model_name": args.model_name}]
    if args.model_preset:
        return [{"model_preset": args.model_preset}]
    if not raw_models:
        return [{"model_preset": DEFAULT_MODEL_PRESET}]
    models = []
    for model in raw_models:
        if isinstance(model, str):
            models.append({"model_preset": model})
        elif isinstance(model, dict):
            models.append(dict(model))
        else:
            raise TypeError("Config models must be strings or objects.")
    return models


def resolve_model(model_entry: dict[str, Any]) -> dict[str, Any]:
    if model_entry.get("model_name"):
        model_name = str(model_entry["model_name"])
        preset = str(model_entry.get("model_preset", "custom"))
    else:
        preset = str(model_entry.get("preset") or model_entry.get("model_preset") or DEFAULT_MODEL_PRESET)
        if preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset '{preset}'. Valid presets: {sorted(MODEL_PRESETS)}")
        model_name = MODEL_PRESETS[preset]
    return {**model_entry, "model_preset": preset, "model_name": model_name}


def normalize_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    segmenter = strategy.get("segmenter") or strategy.get("segmenter_strategy") or "patch"
    masker = strategy.get("masker") or strategy.get("masker_strategy") or "crossmodal_mean"
    if segmenter not in SEGMENTER_CHOICES:
        raise ValueError(f"Unknown segmenter strategy '{segmenter}'.")
    if masker not in MASKER_CHOICES:
        raise ValueError(f"Unknown masker strategy '{masker}'.")
    slic = strategy.get("slic", {})
    gradient_guided = strategy.get("gradient_guided", {})
    gradient_segments = gradient_guided.get("n_segments")
    return {
        "strategy_name": strategy.get("name") or f"{segmenter}_{masker}",
        "segmenter_strategy": segmenter,
        "masker_strategy": masker,
        "slic_n_segments": int(slic.get("n_segments", strategy.get("slic_n_segments", 49))),
        "slic_compactness": float(slic.get("compactness", strategy.get("slic_compactness", 10.0))),
        "slic_sigma": float(slic.get("sigma", strategy.get("slic_sigma", 0.0))),
        "gradient_guided_n_segments": (
            None if gradient_segments is None else int(gradient_segments)
        ),
    }


def strategy_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.segmenter_strategy is None and args.masker_strategy is None:
        return None
    segmenter = args.segmenter_strategy or "patch"
    masker = args.masker_strategy or "crossmodal_mean"
    return normalize_strategy(
        {
            "name": f"{segmenter}_{masker}",
            "segmenter": segmenter,
            "masker": masker,
            "slic": {
                "n_segments": args.slic_n_segments,
                "compactness": args.slic_compactness,
                "sigma": args.slic_sigma,
            },
            "gradient_guided": {"n_segments": args.gradient_guided_n_segments},
        }
    )


def normalize_strategies(raw_strategies: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    cli_strategy = strategy_from_args(args)
    if cli_strategy is not None:
        return [cli_strategy]
    return [normalize_strategy(strategy) for strategy in (raw_strategies or DEFAULT_STRATEGIES)]


def sampler_name(mode: str) -> str:
    return mode.split("/", 1)[0]


def sampler_p(mode: str) -> float | None:
    parts = mode.split("/", 1)
    if len(parts) == 1:
        return 0.5
    return float(parts[1])


def normalize_method(method: dict[str, Any]) -> dict[str, Any]:
    mode = str(method.get("mode", "")).strip()
    if not mode:
        raise ValueError("Each method must define mode.")
    order = int(method.get("order", 1))
    if order not in (1, 2):
        raise ValueError("AID benchmark supports order 1 or 2.")
    approximation_type = str(method.get("approximation_type", "original")).strip().lower()
    if approximation_type not in ("original", "proxyshap"):
        raise ValueError("approximation_type must be original or proxyshap.")
    name = method.get("name") or f"{slug(mode)}_order{order}"
    return {
        "method_name": name,
        "mode": mode,
        "order": order,
        "explainer_name": method.get("explainer_name", name),
        "sampler_name": method.get("sampler_name", sampler_name(mode)),
        "sampler_p": method.get("sampler_p", sampler_p(mode)),
        "approximation_type": approximation_type,
        "budget": int(4096 if method.get("budget") is None else method["budget"]),
        "sparse_regression": bool(method.get("sparse_regression", False)),
        "proxy_params": dict(method.get("proxy_params", {})),
    }


def method_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.method is None and args.mode is None and args.order is None:
        return None
    if args.mode is None or args.order is None:
        raise ValueError("--mode and --order are required for a single CLI method.")
    return normalize_method(
        {
            "name": args.method,
            "mode": args.mode,
            "order": args.order,
            "budget": args.budget,
            "approximation_type": args.approximation_type,
        }
    )


def normalize_methods(raw_methods: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    cli_method = method_from_args(args)
    if cli_method is not None:
        return [cli_method]
    return [normalize_method(method) for method in (raw_methods or DEFAULT_METHODS)]


def build_suite(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config) if args.config else {}
    defaults = normalize_defaults(config, args)
    inputs = [args.input] if args.input else config.get("inputs", [str(DEFAULT_INPUT_DIR.relative_to(PROJECT_ROOT))])
    return {
        "name": args.output_name or config.get("name") or "aid_quality",
        "inputs": inputs,
        "single_text": args.text,
        "defaults": defaults,
        "models": normalize_models(config.get("models"), args),
        "strategies": normalize_strategies(config.get("strategies"), args),
        "methods": normalize_methods(config.get("methods"), args),
    }


def suite_output_dir(suite: dict[str, Any]) -> Path:
    return RESULTS_DIR / f"benchmark_{slug(suite['name'], max_length=80)}"


def describe_suite(suite: dict[str, Any]) -> dict[str, Any]:
    samples = [
        sample
        for input_value in suite["inputs"]
        for sample in resolve_samples(input_value, suite["single_text"], suite["defaults"]["text_column"])
    ]
    return {
        "name": suite_output_dir(suite).name,
        "inputs": suite["inputs"],
        "input_files": len(samples),
        "models": suite["models"],
        "strategies": [
            {
                "strategy_name": strategy["strategy_name"],
                "segmenter_strategy": strategy["segmenter_strategy"],
                "masker_strategy": strategy["masker_strategy"],
            }
            for strategy in suite["strategies"]
        ],
        "methods": suite["methods"],
        "defaults": suite["defaults"],
        "planned_runs": len(samples) * len(suite["models"]) * len(suite["strategies"]) * len(suite["methods"]),
    }
