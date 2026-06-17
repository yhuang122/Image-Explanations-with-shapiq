"""Minimal JSON preset support for validation benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark_schema import (
    CASES,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PRESET,
    DEFAULT_STRATEGY_SPECS,
    IMAGE_SUFFIXES,
    MASKER_CHOICES,
    MODEL_PRESET_BY_NAME,
    MODEL_PRESETS,
    PROJECT_ROOT,
    SEGMENTER_CHOICES,
    slug,
)


DEFAULT_RANDOM_STATE = 0
REQUIRED_DEFAULTS = ("num_coalitions", "batch_size")
MANIFEST_FILENAME = "manifest.csv"
DEFAULT_VISION_BLUR_SIGMA = 3.0


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_manifest_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch input requires manifest.csv: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def sample_from_manifest_row(directory: Path, row: dict) -> dict:
    filename = row.get("filename")
    if not filename:
        raise ValueError("Manifest rows must include a filename column.")
    first_caption = row.get("first_caption", "").strip()
    caption = row.get("caption", "").strip()
    text = first_caption or caption
    if not text:
        raise ValueError("Manifest rows must include a non-empty first_caption or caption column.")
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Manifest image file not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Manifest file is not a supported image: {path}")
    return {"path": path, "text": text, "text_full": caption, "text_source": "first_caption" if first_caption else "caption"}


def resolve_input_samples(input_path: str | None, text: str | None = None) -> list[dict]:
    path = DEFAULT_INPUT_DIR if input_path is None else resolve_project_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Input file is not a supported image: {path}")
        if not text:
            raise ValueError("Single-image input requires --text.")
        return [{"path": path, "text": text, "text_full": text, "text_source": "cli_text"}]

    if text:
        raise ValueError("Batch directory input reads manifest.csv; --text is only for single-image input.")
    return [sample_from_manifest_row(path, row) for row in read_manifest_rows(path / MANIFEST_FILENAME)]


def require_cli_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--case", args.case),
            ("--num-coalitions", args.num_coalitions),
            ("--batch-size", args.batch_size),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required argument(s) without --config: {', '.join(missing)}")


def load_json_config(path_value: str | Path) -> dict:
    path = path_value if isinstance(path_value, Path) else resolve_project_path(path_value)
    if path.suffix.lower() != ".json":
        raise ValueError("Only JSON benchmark presets are supported.")
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def config_json_paths(path_value: str) -> list[Path]:
    path = resolve_project_path(path_value)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Benchmark config path not found: {path}")
    paths = sorted(path.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Benchmark config directory contains no JSON files: {path}")
    return paths


def suite_group_output_name(path_value: str) -> str:
    path = resolve_project_path(path_value)
    name = slug(path.name if path.is_dir() else path.stem, max_length=80)
    return name if name.startswith("benchmark_") else f"benchmark_{name}"


def merged_defaults(config: dict, args: argparse.Namespace) -> dict:
    defaults = dict(config.get("defaults", {}))
    for key in REQUIRED_DEFAULTS:
        cli_value = getattr(args, key)
        if cli_value is not None:
            defaults[key] = cli_value
        if defaults.get(key) is None:
            raise ValueError(f"Benchmark config is missing defaults.{key}")

    for key, default_value in (
        ("random_state", DEFAULT_RANDOM_STATE),
        ("tolerance", 1e-4),
        ("cuda", False),
        ("use_amp", False),
    ):
        cli_value = getattr(args, key)
        defaults[key] = default_value if cli_value is None and key not in defaults else defaults.get(key)
        if cli_value is not None:
            defaults[key] = cli_value

    return {
        "random_state": int(defaults["random_state"]),
        "num_coalitions": int(defaults["num_coalitions"]),
        "batch_size": int(defaults["batch_size"]),
        "tolerance": float(defaults["tolerance"]),
        "cuda": bool(defaults["cuda"]),
        "use_amp": bool(defaults["use_amp"]),
    }


def runtime_args(
    case_name: str,
    text: str | None,
    defaults: dict,
) -> argparse.Namespace:
    return argparse.Namespace(
        case=case_name,
        text=text,
        random_state=defaults["random_state"],
        num_coalitions=defaults["num_coalitions"],
        batch_size=defaults["batch_size"],
        tolerance=defaults["tolerance"],
        cuda=defaults["cuda"],
        use_amp=defaults["use_amp"],
    )


def normalize_cases(raw_cases, cli_case: str | None) -> list[dict]:
    case_names = [cli_case] if cli_case else raw_cases
    if not case_names:
        raise ValueError("Benchmark config must define 'cases' or use --case.")
    cases = []
    for case_name in case_names:
        if case_name not in CASES:
            raise ValueError(f"Unknown case '{case_name}'. Valid cases: {sorted(CASES)}")
        cases.append({"name": case_name})
    return cases


def normalize_inputs(raw_inputs, args: argparse.Namespace) -> list[str | None]:
    if args.text and not args.input:
        raise ValueError("--text requires --input.")
    if args.input:
        return [args.input]
    if not raw_inputs:
        return [None]
    normalized = []
    for item in raw_inputs:
        if isinstance(item, str):
            normalized.append(item)
        else:
            raise TypeError("Each config input must be a path string.")
    return normalized


def normalize_models(raw_models, args: argparse.Namespace) -> list[dict]:
    if args.model_name:
        return [{"model_name": args.model_name}]
    if args.model_preset:
        return [{"model_preset": args.model_preset}]
    if not raw_models:
        return [{}]
    models = []
    for preset in raw_models:
        if not isinstance(preset, str):
            raise TypeError("Config models must be model preset strings.")
        models.append({"model_preset": preset})
    return models


def normalize_strategy(strategy: dict) -> dict:
    if not isinstance(strategy, dict):
        raise TypeError("Each config strategy must be an object.")
    segmenter = strategy.get("segmenter", "patch")
    masker = strategy.get("masker", "crossmodal_mean")
    if segmenter not in SEGMENTER_CHOICES:
        raise ValueError(f"Unknown segmenter strategy '{segmenter}'.")
    if masker not in MASKER_CHOICES:
        raise ValueError(f"Unknown masker strategy '{masker}'.")

    slic = strategy.get("slic", {})
    gradient_guided = strategy.get("gradient_guided", {})
    vision_blur = strategy.get("vision_blur", {})
    crossmodal_blur = strategy.get("crossmodal_blur", vision_blur)
    gradient_guided_n_segments = gradient_guided.get("n_segments")
    return {
        "strategy_name": strategy.get("name") or f"{segmenter}_{masker}",
        "segmenter_strategy": segmenter,
        "masker_strategy": masker,
        "slic_n_segments": int(slic.get("n_segments", 49)),
        "slic_compactness": float(slic.get("compactness", 10.0)),
        "slic_sigma": float(slic.get("sigma", 0.0)),
        "gradient_guided_n_segments": (
            None if gradient_guided_n_segments is None else int(gradient_guided_n_segments)
        ),
        "vision_blur_sigma": float(vision_blur.get("sigma", DEFAULT_VISION_BLUR_SIGMA)),
        "crossmodal_blur_sigma": float(crossmodal_blur.get("sigma", DEFAULT_VISION_BLUR_SIGMA)),
    }


def resolve_strategy_specs(args: argparse.Namespace) -> list[dict]:
    single_requested = args.segmenter_strategy is not None or args.masker_strategy is not None
    if single_requested:
        segmenter = args.segmenter_strategy or "patch"
        masker = args.masker_strategy or "crossmodal_mean"
        return [
            {
                "strategy_name": f"{segmenter}_{masker}",
                "segmenter_strategy": segmenter,
                "masker_strategy": masker,
                "slic_n_segments": args.slic_n_segments,
                "slic_compactness": args.slic_compactness,
                "slic_sigma": args.slic_sigma,
                "gradient_guided_n_segments": args.gradient_guided_n_segments,
                "vision_blur_sigma": args.vision_blur_sigma,
                "crossmodal_blur_sigma": args.vision_blur_sigma,
            }
        ]

    specs = []
    for spec in DEFAULT_STRATEGY_SPECS:
        specs.append(
            {
                **spec,
                "slic_n_segments": args.slic_n_segments,
                "slic_compactness": args.slic_compactness,
                "slic_sigma": args.slic_sigma,
                "gradient_guided_n_segments": args.gradient_guided_n_segments,
                "vision_blur_sigma": args.vision_blur_sigma,
                "crossmodal_blur_sigma": args.vision_blur_sigma,
            }
        )
    return specs


def normalize_strategies(raw_strategies, args: argparse.Namespace) -> list[dict]:
    if args.segmenter_strategy or args.masker_strategy:
        return resolve_strategy_specs(args)
    if not raw_strategies:
        return resolve_strategy_specs(args)
    return [normalize_strategy(strategy) for strategy in raw_strategies]


def build_suite_from_config(config: dict, args: argparse.Namespace, config_path: Path | None = None) -> dict:
    name = config.get("name")
    if name is None and config_path is not None:
        name = config_path.stem
    elif name is None:
        name = Path(args.config).stem
    return {
        "name": name,
        "run_mode": args.run_mode,
        "single_text": args.text,
        "defaults": merged_defaults(config, args),
        "cases": normalize_cases(config.get("cases"), args.case),
        "inputs": normalize_inputs(config.get("inputs"), args),
        "models": normalize_models(config.get("models"), args),
        "strategies": normalize_strategies(config.get("strategies"), args),
    }


def build_suite(args: argparse.Namespace) -> dict:
    if args.config is None:
        require_cli_args(args)
        defaults = {
            "random_state": DEFAULT_RANDOM_STATE if args.random_state is None else int(args.random_state),
            "num_coalitions": int(args.num_coalitions),
            "batch_size": int(args.batch_size),
            "tolerance": 1e-4 if args.tolerance is None else float(args.tolerance),
            "cuda": bool(args.cuda),
            "use_amp": bool(args.use_amp),
        }
        return {
            "name": None,
            "run_mode": args.run_mode,
            "single_text": args.text,
            "defaults": defaults,
            "cases": [{"name": args.case}],
            "inputs": [args.input],
            "models": normalize_models([], args),
            "strategies": resolve_strategy_specs(args),
        }

    config = load_json_config(args.config)
    return build_suite_from_config(config, args)


def build_suites(args: argparse.Namespace) -> tuple[list[dict], str]:
    if args.config is None:
        suite = build_suite(args)
        return [suite], suite_output_name(suite)

    config_path = resolve_project_path(args.config)
    if config_path.is_dir():
        suites = [
            build_suite_from_config(load_json_config(path), args, path)
            for path in config_json_paths(args.config)
        ]
        return suites, suite_group_output_name(args.config)

    suite = build_suite(args)
    return [suite], suite_output_name(suite)


def resolve_model_selection(case: dict, args: argparse.Namespace) -> None:
    if args.model_name:
        case["model_preset"] = "custom"
        case["model_name"] = args.model_name
        return

    model_preset = args.model_preset or case.get("model_preset")
    if model_preset:
        if model_preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset '{model_preset}'. Valid presets: {sorted(MODEL_PRESETS)}")
        case["model_preset"] = model_preset
        case["model_name"] = MODEL_PRESETS[model_preset]
        return

    case.setdefault("model_name", MODEL_PRESETS[DEFAULT_MODEL_PRESET])
    case["model_preset"] = MODEL_PRESET_BY_NAME.get(case["model_name"], "custom")


def resolve_model_case(case_entry: dict, model_entry: dict) -> dict:
    case = dict(CASES[case_entry["name"]])
    resolve_model_selection(
        case,
        argparse.Namespace(
            model_name=model_entry.get("model_name"),
            model_preset=model_entry.get("model_preset"),
        ),
    )
    return case


def suite_output_name(suite: dict) -> str:
    if suite["name"]:
        return f"benchmark_{slug(suite['name'], max_length=80)}"
    case_token = suite["cases"][0]["name"] if len(suite["cases"]) == 1 else f"{len(suite['cases'])}_cases"
    input_token = "default_inputs"
    if len(suite["inputs"]) == 1 and suite["inputs"][0]:
        input_token = Path(suite["inputs"][0]).stem
    elif len(suite["inputs"]) > 1:
        input_token = f"{len(suite['inputs'])}_inputs"
    model_token = "case_default_models" if suite["models"] == [{}] else f"{len(suite['models'])}_models"
    defaults = suite["defaults"]
    return f"benchmark_{slug(case_token)}_{slug(input_token)}_{model_token}_rs{defaults['random_state']}_n{defaults['num_coalitions']}"


def describe_suite(suite: dict) -> dict:
    samples = [
        sample
        for input_path in suite["inputs"]
        for sample in resolve_input_samples(input_path, suite.get("single_text"))
    ]
    strategy_count = 1 if suite["run_mode"] == "original" else len(suite["strategies"])
    return {
        "name": suite_output_name(suite),
        "run_mode": suite["run_mode"],
        "cases": [entry["name"] for entry in suite["cases"]],
        "inputs": suite["inputs"],
        "input_files": len(samples),
        "models": suite["models"],
        "strategies": [] if suite["run_mode"] == "original" else [
            strategy_plan(spec)
            for spec in suite["strategies"]
        ],
        "defaults": suite["defaults"],
        "planned_runs": len(suite["cases"]) * len(samples) * len(suite["models"]) * strategy_count,
    }


def strategy_plan(strategy_spec: dict) -> dict:
    segmenter_params = {}
    if strategy_spec["segmenter_strategy"] == "slic":
        segmenter_params = {
            "n_segments": strategy_spec["slic_n_segments"],
            "compactness": strategy_spec["slic_compactness"],
            "sigma": strategy_spec["slic_sigma"],
        }
    elif strategy_spec["segmenter_strategy"] == "gradient_guided":
        segmenter_params = {"n_segments": strategy_spec["gradient_guided_n_segments"]}

    masker_params = {}
    if strategy_spec["masker_strategy"] == "vision_blur":
        masker_params = {"sigma": strategy_spec["vision_blur_sigma"]}
    elif strategy_spec["masker_strategy"] == "crossmodal_blur":
        masker_params = {"sigma": strategy_spec["crossmodal_blur_sigma"]}

    return {
        "strategy_name": strategy_spec["strategy_name"],
        "segmenter_strategy": strategy_spec["segmenter_strategy"],
        "segmenter_params": segmenter_params,
        "masker_strategy": strategy_spec["masker_strategy"],
        "masker_params": masker_params,
    }
