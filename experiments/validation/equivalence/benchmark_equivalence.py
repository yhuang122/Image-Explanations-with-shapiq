"""Pipeline equivalence benchmark for original and migrated VLM games."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from benchmark_outputs import (
    active_params,
    compact_json,
    comparison_csv_path,
    max_output_diff_metric,
    mean_output_diff_metric,
    read_existing_report,
    suite_output_dir,
    write_benchmark_summary,
    write_failure_result,
    write_original_summary,
    write_run_metadata,
    write_results,
)
from benchmark_schema import (
    CASES,
    MASKER_CHOICES,
    MIGRATED_PIPELINE,
    MODEL_PRESETS,
    ORIGINAL_PIPELINE,
    PROJECT_ROOT,
    SEGMENTER_CHOICES,
)
from benchmark_suite import (
    DEFAULT_VISION_BLUR_SIGMA,
    build_suites,
    describe_suite,
    resolve_input_samples,
    resolve_model_case,
    runtime_args,
)


if TYPE_CHECKING:
    import torch

    from ImputerFactory import MaskerConfig, SegmenterConfig


sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_SRC_PACKAGE = "_benchmark_equivalence_local_src"
LOCAL_SRC_DIR = PROJECT_ROOT / "src"


def local_src_module(module_name: str):
    """Load selected src modules without executing src/__init__.py."""
    if LOCAL_SRC_PACKAGE not in sys.modules:
        package = types.ModuleType(LOCAL_SRC_PACKAGE)
        package.__path__ = [str(LOCAL_SRC_DIR)]
        sys.modules[LOCAL_SRC_PACKAGE] = package

    full_name = f"{LOCAL_SRC_PACKAGE}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = LOCAL_SRC_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local src module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline equivalence and coverage benchmarks.")
    parser.add_argument(
        "--config",
        help=(
            "JSON benchmark suite config file or a folder of JSON sub-suites. "
            "A folder writes all sub-suite results into one shared results folder."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded benchmark plan and exit. Normal runs already print this plan before execution.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute runs even when result CSVs already exist.")
    parser.add_argument("--case", choices=sorted(CASES), help="Experiment case name.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        help="Use a standard model preset. Covers the A3 cross-model adoption models.",
    )
    model_group.add_argument("--model-name", help="Override with an arbitrary HuggingFace model id.")
    parser.add_argument(
        "--input",
        help=(
            "Single image file or manifest-backed image directory. Relative paths use project root. "
            "Defaults to data/input/wds_mscoco_captions_test_100."
        ),
    )
    parser.add_argument("--text", help="Required text input for single-image input.")
    parser.add_argument(
        "--run-mode",
        choices=("compare", "original"),
        default="compare",
        help="Use 'compare' for original-vs-migrated benchmark, or 'original' for original pipeline only.",
    )
    for name in ("random-state", "num-coalitions", "batch-size"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--cuda", action="store_true", default=None, help="Run comparison on CUDA. Defaults to CPU.")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=None,
        help="Use torch autocast in the migrated pipeline.",
    )
    parser.add_argument(
        "--segmenter-strategy",
        choices=SEGMENTER_CHOICES,
        help="Run one explicit segmenter strategy instead of the default benchmark suite.",
    )
    parser.add_argument("--slic-n-segments", type=int, default=49)
    parser.add_argument("--slic-compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=0.0)
    parser.add_argument("--gradient-guided-n-segments", type=int)
    parser.add_argument(
        "--vision-blur-sigma",
        type=float,
        default=DEFAULT_VISION_BLUR_SIGMA,
        help="Gaussian sigma for vision_blur and crossmodal_blur maskers.",
    )
    parser.add_argument(
        "--masker-strategy",
        choices=MASKER_CHOICES,
        help="Run one explicit masker strategy instead of the default benchmark suite.",
    )
    return parser.parse_args()


def resolve_device(use_cuda: bool) -> torch.device:
    import torch

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --cuda but is not available.")
    return torch.device("cuda" if use_cuda else "cpu")


def generate_coalitions(n_players: int, num_coalitions: int, random_state: int) -> np.ndarray:
    if num_coalitions < 2:
        raise ValueError("--num-coalitions must be at least 2.")
    rng = np.random.default_rng(random_state)
    coalitions = rng.random((num_coalitions, n_players)) >= 0.5
    coalitions[0, :] = False
    coalitions[1, :] = True
    return coalitions


def build_segmenter_config(strategy_spec: dict) -> SegmenterConfig:
    from ImputerFactory import GradientGuidedParams, SegmenterConfig, SlicParams

    return SegmenterConfig(
        strategy=strategy_spec["segmenter_strategy"],
        slic=SlicParams(
            n_segments=strategy_spec["slic_n_segments"],
            compactness=strategy_spec["slic_compactness"],
            sigma=strategy_spec["slic_sigma"],
        ),
        gradient_guided=GradientGuidedParams(n_segments=strategy_spec["gradient_guided_n_segments"]),
    )


def build_masker_config(strategy_spec: dict) -> MaskerConfig:
    from ImputerFactory import MaskerConfig

    config = MaskerConfig(strategy=strategy_spec["masker_strategy"])
    if strategy_spec["masker_strategy"] == "crossmodal_blur" and "crossmodal_blur_sigma" in strategy_spec:
        config.vision_blur.sigma = strategy_spec["crossmodal_blur_sigma"]
    elif "vision_blur_sigma" in strategy_spec:
        config.vision_blur.sigma = strategy_spec["vision_blur_sigma"]
    return config


def load_model_bundle(case: dict, device: torch.device) -> dict:
    from transformers import AutoModel, AutoProcessor

    start = perf_counter()
    model = AutoModel.from_pretrained(case["model_name"])
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(case["model_name"])
    return {
        "model": model,
        "processor": processor,
        "model_load_runtime_s": perf_counter() - start,
    }


def build_original_game(model_bundle: dict, image: Image.Image, text: str, batch_size: int):
    game_huggingface = local_src_module("game_huggingface")
    start = perf_counter()
    old_game = game_huggingface.VisionLanguageGame(
        model_bundle["model"],
        model_bundle["processor"],
        input_image=image,
        input_text=text,
        batch_size=batch_size,
    )
    return old_game, perf_counter() - start


def build_migrated_game(case: dict, model_bundle: dict, image: Image.Image):
    from Game import VisionLanguageGame
    from ImputerFactory import ImageImputerFactory

    start = perf_counter()
    imputer = ImageImputerFactory().build(
        model_bundle["model"],
        model_bundle["processor"],
        image,
        case["text"],
        segmenter_config=case["segmenter_config"],
        masker_config=case["masker_config"],
        use_amp=case["use_amp"],
    )
    new_game = VisionLanguageGame(imputer, batch_size=case["batch_size"])
    return new_game, imputer, perf_counter() - start


def build_original_context(model_bundle: dict, input_path: Path, text: str, batch_size: int) -> dict:
    image = Image.open(input_path).convert("RGB")
    old_game, original_game_build_runtime_s = build_original_game(model_bundle, image, text, batch_size)
    original_anchor_start = perf_counter()
    original_anchor_values = evaluate_anchor_coalitions(old_game)
    original_anchor_runtime_s = perf_counter() - original_anchor_start
    return {
        "image": image,
        "old_game": old_game,
        "original_game_build_runtime_s": original_game_build_runtime_s,
        "original_anchor_values": original_anchor_values,
        "original_anchor_runtime_s": original_anchor_runtime_s,
    }


def has_matching_player_layout(old_game, new_game) -> bool:
    fields = ("n_players", "n_players_image", "n_players_text")
    return all(getattr(old_game, field) == getattr(new_game, field) for field in fields)


def evaluate_anchor_coalitions(game) -> np.ndarray:
    coalitions = np.zeros((2, game.n_players), dtype=bool)
    coalitions[1, :] = True
    return game.value_function(coalitions)


def build_coalition_inputs(game, case: dict) -> dict:
    if case["comparison_type"] == "crossmodal":
        return {
            "comparison_type": "crossmodal",
            "image_coalitions": generate_coalitions(
                game.n_players_image, case["num_coalitions"], case["random_state"]
            ),
            "text_coalitions": generate_coalitions(
                game.n_players_text, case["num_coalitions"], case["random_state"] + 1
            ),
        }
    return {
        "comparison_type": "1d",
        "coalitions": generate_coalitions(game.n_players, case["num_coalitions"], case["random_state"]),
    }


def evaluate_game_outputs(game, inputs: dict) -> np.ndarray:
    if inputs["comparison_type"] == "crossmodal":
        values = game.value_function_crossmodal(
            inputs["image_coalitions"],
            inputs["text_coalitions"],
        )
        return values.reshape(-1)
    return game.value_function(inputs["coalitions"])


def is_strict_equivalence_run(case: dict, coalition_comparison_available: bool) -> bool:
    return (
        case["segmenter_config"].strategy == "patch"
        and case["masker_config"].strategy == "crossmodal_mean"
        and coalition_comparison_available
    )


def comparison_scope(case: dict, coalition_comparison_available: bool) -> dict:
    candidate_name = (
        f"{MIGRATED_PIPELINE}:"
        f"{case['segmenter_config'].strategy}/{case['masker_config'].strategy}"
    )
    if is_strict_equivalence_run(case, coalition_comparison_available):
        return {
            "comparison_scope": "strict_equivalence",
            "reference_name": ORIGINAL_PIPELINE,
            "candidate_name": candidate_name,
            "equivalence_expected": True,
            "metric_family": "output_equivalence",
        }
    if coalition_comparison_available:
        return {
            "comparison_scope": "baseline_deviation",
            "reference_name": ORIGINAL_PIPELINE,
            "candidate_name": candidate_name,
            "equivalence_expected": False,
            "metric_family": "baseline_deviation",
        }
    return {
        "comparison_scope": "anchor_compatibility",
        "reference_name": f"{ORIGINAL_PIPELINE}:empty_full_anchors",
        "candidate_name": candidate_name,
        "equivalence_expected": False,
        "metric_family": "anchor_consistency",
    }


def summarize_coalition_differences(
    old_pipeline_values: np.ndarray | None,
    new_pipeline_values: np.ndarray,
    tolerance: float,
) -> dict:
    if old_pipeline_values is None:
        return {
            "coalition_max_abs_output_diff": None,
            "coalition_mean_abs_output_diff": None,
            "equivalence_passed": None,
        }

    diff = np.abs(old_pipeline_values - new_pipeline_values)
    return {
        "coalition_max_abs_output_diff": float(np.max(diff)),
        "coalition_mean_abs_output_diff": float(np.mean(diff)),
        "equivalence_passed": bool(np.max(diff) <= tolerance),
    }


def build_benchmark_report(
    *,
    run_args: argparse.Namespace,
    run_config: dict,
    device,
    model_bundle: dict,
    input_context: dict,
    imputer,
    old_game,
    new_game,
    coalition_comparison_available: bool,
    comparison_mode: str,
    original_pipeline_runtime_s,
    migrated_pipeline_runtime_s: float,
    migrated_game_build_runtime_s: float,
    original_anchor_values: np.ndarray,
    new_anchor_values: np.ndarray,
    new_anchor_runtime_s: float,
    new_pipeline_values: np.ndarray,
    total_start: float,
    diff_result: dict,
) -> dict:
    anchor_diff = np.abs(original_anchor_values - new_anchor_values)
    strict_equivalence = is_strict_equivalence_run(run_config, coalition_comparison_available)
    report = {
        "case": run_args.case,
        "strategy_name": run_config["strategy_name"],
        "original_pipeline": ORIGINAL_PIPELINE,
        "migrated_pipeline": MIGRATED_PIPELINE,
        "comparison_type": run_config["comparison_type"],
        "comparison_mode": comparison_mode,
        **comparison_scope(run_config, coalition_comparison_available),
        "input_path": str(run_config["input_path"]),
        "model_preset": run_config["model_preset"],
        "model_name": run_config["model_name"],
        "model_type": imputer.model_type,
        "text": run_config["text"],
        "text_full": run_config["text_full"],
        "text_source": run_config["text_source"],
        "device": str(device),
        "use_amp": bool(run_config["use_amp"]),
        "segmenter_strategy": run_config["segmenter_config"].strategy,
        "segmenter_params": compact_json(active_params(run_config["segmenter_config"])),
        "masker_strategy": run_config["masker_config"].strategy,
        "masker_params": compact_json(active_params(run_config["masker_config"])),
        "image_size": int(imputer.image_size),
        "patch_size": int(imputer.patch_size),
        "grid_size": int(imputer.grid_size),
        "text_total_length": int(imputer.layout.text_total_length),
        "n_players": int(new_game.n_players),
        "n_players_image": int(new_game.n_players_image),
        "n_players_text": int(new_game.n_players_text),
        "original_n_players": int(old_game.n_players),
        "original_n_players_image": int(old_game.n_players_image),
        "original_n_players_text": int(old_game.n_players_text),
        "coalition_comparison_available": bool(coalition_comparison_available),
        "num_coalitions": int(run_config["num_coalitions"]),
        "batch_size": int(run_config["batch_size"]),
        "random_state": int(run_config["random_state"]),
        "tolerance": run_args.tolerance,
        "model_load_runtime_s": model_bundle["model_load_runtime_s"],
        "original_game_build_runtime_s": input_context["original_game_build_runtime_s"],
        "migrated_game_build_runtime_s": migrated_game_build_runtime_s,
        "original_anchor_runtime_s": input_context["original_anchor_runtime_s"],
        "migrated_anchor_runtime_s": new_anchor_runtime_s,
        "build_runtime_s": input_context["original_game_build_runtime_s"] + migrated_game_build_runtime_s,
        "original_pipeline_runtime_s": original_pipeline_runtime_s,
        "migrated_pipeline_runtime_s": migrated_pipeline_runtime_s,
        "total_runtime_s": perf_counter() - total_start,
        "original_pipeline_empty_coalition_output": float(original_anchor_values[0]),
        "migrated_pipeline_empty_coalition_output": float(new_anchor_values[0]),
        "abs_empty_coalition_output_diff": float(anchor_diff[0]),
        "original_pipeline_full_coalition_output": float(original_anchor_values[1]),
        "migrated_pipeline_full_coalition_output": float(new_anchor_values[1]),
        "abs_full_coalition_output_diff": float(anchor_diff[1]),
        "empty_full_anchor_max_abs_output_diff": float(np.max(anchor_diff)),
        "empty_full_anchor_mean_abs_output_diff": float(np.mean(anchor_diff)),
        "empty_full_anchor_passed": bool(np.max(anchor_diff) <= run_args.tolerance),
        **diff_result,
        "strict_equivalence": bool(strict_equivalence),
    }
    report["benchmark_metric_max_output_diff"] = max_output_diff_metric(report)
    report["benchmark_metric_mean_output_diff"] = mean_output_diff_metric(report)
    report["passed"] = report["equivalence_passed"] if strict_equivalence else None
    return report


def infer_model_type(model_name: str) -> str:
    name = model_name.lower()
    if "siglip2" in name:
        return "siglip2"
    if "siglip" in name:
        return "siglip"
    if "clip" in name:
        return "clip"
    return "unknown"


def model_image_layout(model_bundle: dict) -> dict:
    embeddings = model_bundle["model"].vision_model.embeddings
    image_size = int(embeddings.image_size)
    patch_size = int(embeddings.patch_size)
    grid_size = image_size // patch_size
    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "n_players_image": grid_size**2,
    }


def build_failure_report(
    *,
    run_args: argparse.Namespace,
    run_config: dict,
    device,
    model_bundle: dict,
    total_start: float,
    error: Exception,
) -> dict:
    layout = model_image_layout(model_bundle)
    candidate_name = (
        f"{MIGRATED_PIPELINE}:"
        f"{run_config['segmenter_config'].strategy}/{run_config['masker_config'].strategy}"
    )
    return {
        "case": run_args.case,
        "strategy_name": run_config["strategy_name"],
        "original_pipeline": ORIGINAL_PIPELINE,
        "migrated_pipeline": MIGRATED_PIPELINE,
        "comparison_type": run_config["comparison_type"],
        "comparison_mode": "setup_failure",
        "comparison_scope": "original_pipeline_incompatible",
        "reference_name": ORIGINAL_PIPELINE,
        "candidate_name": candidate_name,
        "equivalence_expected": False,
        "metric_family": f"setup_failure:{type(error).__name__}",
        "input_path": str(run_config["input_path"]),
        "model_preset": run_config["model_preset"],
        "model_name": run_config["model_name"],
        "model_type": infer_model_type(run_config["model_name"]),
        "text": run_config["text"],
        "text_full": run_config["text_full"],
        "text_source": run_config["text_source"],
        "device": str(device),
        "use_amp": bool(run_config["use_amp"]),
        "segmenter_strategy": run_config["segmenter_config"].strategy,
        "segmenter_params": compact_json(active_params(run_config["segmenter_config"])),
        "masker_strategy": run_config["masker_config"].strategy,
        "masker_params": compact_json(active_params(run_config["masker_config"])),
        "image_size": layout["image_size"],
        "patch_size": layout["patch_size"],
        "grid_size": layout["grid_size"],
        "text_total_length": "",
        "n_players": "",
        "n_players_image": "",
        "n_players_text": "",
        "original_n_players": "",
        "original_n_players_image": "",
        "original_n_players_text": "",
        "coalition_comparison_available": False,
        "num_coalitions": int(run_config["num_coalitions"]),
        "batch_size": int(run_config["batch_size"]),
        "random_state": int(run_config["random_state"]),
        "tolerance": run_args.tolerance,
        "model_load_runtime_s": model_bundle["model_load_runtime_s"],
        "original_game_build_runtime_s": "",
        "migrated_game_build_runtime_s": "",
        "original_anchor_runtime_s": "",
        "migrated_anchor_runtime_s": "",
        "build_runtime_s": "",
        "original_pipeline_runtime_s": "",
        "migrated_pipeline_runtime_s": "",
        "total_runtime_s": perf_counter() - total_start,
        "original_pipeline_empty_coalition_output": "",
        "migrated_pipeline_empty_coalition_output": "",
        "abs_empty_coalition_output_diff": "",
        "original_pipeline_full_coalition_output": "",
        "migrated_pipeline_full_coalition_output": "",
        "abs_full_coalition_output_diff": "",
        "empty_full_anchor_max_abs_output_diff": "",
        "empty_full_anchor_mean_abs_output_diff": "",
        "empty_full_anchor_passed": False,
        "coalition_max_abs_output_diff": "",
        "coalition_mean_abs_output_diff": "",
        "benchmark_metric_max_output_diff": "",
        "benchmark_metric_mean_output_diff": "",
        "equivalence_passed": False,
        "passed": False,
        "strict_equivalence": False,
    }


def run_strategy_comparison(
    run_config: dict,
    run_args: argparse.Namespace,
    device: torch.device,
    model_bundle: dict,
    input_context: dict,
    output_dir: Path,
) -> dict:
    utils = local_src_module("utils")
    total_start = perf_counter()
    utils.set_seed(run_config["random_state"])
    old_game = input_context["old_game"]
    new_game, imputer, migrated_game_build_runtime_s = build_migrated_game(
        run_config, model_bundle, input_context["image"]
    )
    coalition_comparison_available = has_matching_player_layout(old_game, new_game)

    original_anchor_values = input_context["original_anchor_values"]
    new_anchor_start = perf_counter()
    new_anchor_values = evaluate_anchor_coalitions(new_game)
    new_anchor_runtime_s = perf_counter() - new_anchor_start

    if coalition_comparison_available:
        cache_key = (run_config["comparison_type"], run_config["num_coalitions"], run_config["random_state"])
        original_pipeline_cache = input_context.setdefault("original_pipeline_cache", {})
        if cache_key not in original_pipeline_cache:
            inputs = build_coalition_inputs(old_game, run_config)
            original_pipeline_start = perf_counter()
            original_pipeline_cache[cache_key] = {
                "inputs": inputs,
                "values": evaluate_game_outputs(old_game, inputs),
                "runtime_s": perf_counter() - original_pipeline_start,
            }
        cached = original_pipeline_cache[cache_key]
        inputs = cached["inputs"]
        old_pipeline_values = cached["values"]
        original_pipeline_runtime_s = cached["runtime_s"]
        comparison_mode = "original_pipeline_vs_migrated_pipeline"
    else:
        inputs = build_coalition_inputs(new_game, run_config)
        original_pipeline_runtime_s = None
        old_pipeline_values = None
        comparison_mode = "original_anchor_vs_migrated_pipeline"

    migrated_pipeline_start = perf_counter()
    new_pipeline_values = evaluate_game_outputs(new_game, inputs)
    migrated_pipeline_runtime_s = perf_counter() - migrated_pipeline_start

    report = build_benchmark_report(
        run_args=run_args,
        run_config=run_config,
        device=device,
        model_bundle=model_bundle,
        input_context=input_context,
        imputer=imputer,
        old_game=old_game,
        new_game=new_game,
        coalition_comparison_available=coalition_comparison_available,
        comparison_mode=comparison_mode,
        original_pipeline_runtime_s=original_pipeline_runtime_s,
        migrated_pipeline_runtime_s=migrated_pipeline_runtime_s,
        migrated_game_build_runtime_s=migrated_game_build_runtime_s,
        original_anchor_values=original_anchor_values,
        new_anchor_values=new_anchor_values,
        new_anchor_runtime_s=new_anchor_runtime_s,
        new_pipeline_values=new_pipeline_values,
        total_start=total_start,
        diff_result=summarize_coalition_differences(old_pipeline_values, new_pipeline_values, run_args.tolerance),
    )
    report["result_paths"] = write_results(
        output_dir,
        run_config,
        report,
        inputs,
        old_pipeline_values,
        new_pipeline_values,
    )
    return report


def run_original_pipeline(
    run_config: dict,
    model_bundle: dict,
    input_path: Path,
    text: str,
    text_full: str,
    text_source: str,
) -> dict:
    utils = local_src_module("utils")
    total_start = perf_counter()
    utils.set_seed(run_config["random_state"])
    input_context = build_original_context(model_bundle, input_path, text, run_config["batch_size"])
    old_game = input_context["old_game"]
    inputs = build_coalition_inputs(old_game, run_config)
    original_pipeline_start = perf_counter()
    values = evaluate_game_outputs(old_game, inputs)
    original_pipeline_runtime_s = perf_counter() - original_pipeline_start
    return {
        "case": run_config["case"],
        "input_path": str(input_path),
        "model_preset": run_config["model_preset"],
        "model_name": run_config["model_name"],
        "text": text,
        "text_full": text_full,
        "text_source": text_source,
        "comparison_type": run_config["comparison_type"],
        "n_players": int(old_game.n_players),
        "n_players_image": int(old_game.n_players_image),
        "n_players_text": int(old_game.n_players_text),
        "num_coalitions": int(values.shape[0]),
        "batch_size": int(run_config["batch_size"]),
        "random_state": int(run_config["random_state"]),
        "model_load_runtime_s": model_bundle["model_load_runtime_s"],
        "original_game_build_runtime_s": input_context["original_game_build_runtime_s"],
        "original_anchor_runtime_s": input_context["original_anchor_runtime_s"],
        "original_pipeline_runtime_s": original_pipeline_runtime_s,
        "total_runtime_s": perf_counter() - total_start,
        "empty_coalition_output": float(input_context["original_anchor_values"][0]),
        "full_coalition_output": float(input_context["original_anchor_values"][1]),
        "original_pipeline_outputs": json.dumps([float(value) for value in values]),
    }


def build_strategy_run_config(
    run_args: argparse.Namespace,
    model_config: dict,
    input_path: Path,
    strategy_spec: dict,
    text_full: str,
    text_source: str,
) -> dict:
    run_config = dict(model_config)
    run_config.update(
        case=run_args.case,
        input_path=input_path,
        text=run_args.text,
        text_full=text_full,
        text_source=text_source,
        random_state=run_args.random_state,
        num_coalitions=run_args.num_coalitions,
        batch_size=run_args.batch_size,
        tolerance=run_args.tolerance,
        use_amp=run_args.use_amp,
        strategy_name=strategy_spec["strategy_name"],
        segmenter_config=build_segmenter_config(strategy_spec),
        masker_config=build_masker_config(strategy_spec),
    )
    return run_config


def build_original_pipeline_run_config(run_args: argparse.Namespace, model_config: dict) -> dict:
    run_config = dict(model_config)
    run_config.update(
        case=run_args.case,
        random_state=run_args.random_state,
        num_coalitions=run_args.num_coalitions,
        batch_size=run_args.batch_size,
    )
    return run_config


def is_failed_strict_report(report: dict) -> bool:
    return str(report.get("strict_equivalence")).lower() == "true" and str(report.get("passed")).lower() == "false"


def run_suite(args: argparse.Namespace, suite: dict, output_dir: Path) -> list[dict]:
    device = resolve_device(bool(suite["defaults"]["cuda"]))
    model_cache = {}
    input_context_cache = {}
    input_context_error_cache = {}
    reports = []

    for case_entry in suite["cases"]:
        case_name = case_entry["name"]
        for model_entry in suite["models"]:
            model_config = resolve_model_case(case_entry, model_entry)
            model_key = model_config["model_name"]

            for input_entry in suite["inputs"]:
                for sample in resolve_input_samples(input_entry, args.text):
                    input_path = sample["path"]
                    text = sample["text"]
                    text_full = sample["text_full"]
                    text_source = sample["text_source"]
                    run_args = runtime_args(
                        case_name=case_name,
                        text=text,
                        defaults=suite["defaults"],
                    )
                    if args.run_mode == "original":
                        if model_key not in model_cache:
                            model_cache[model_key] = load_model_bundle(model_config, device)
                        model_bundle = model_cache[model_key]
                        reports.append(
                            run_original_pipeline(
                                build_original_pipeline_run_config(run_args, model_config),
                                model_bundle,
                                input_path,
                                text,
                                text_full,
                                text_source,
                            )
                        )
                    else:
                        for strategy_spec in suite["strategies"]:
                            strategy_run_config = build_strategy_run_config(
                                run_args,
                                model_config,
                                input_path,
                                strategy_spec,
                                text_full,
                                text_source,
                            )
                            existing_report = None if args.force else read_existing_report(
                                comparison_csv_path(output_dir, strategy_run_config)
                            )
                            if existing_report is not None:
                                reports.append(existing_report)
                                continue

                            if model_key not in model_cache:
                                model_cache[model_key] = load_model_bundle(model_config, device)
                            model_bundle = model_cache[model_key]
                            context_key = (model_key, str(input_path), text, run_args.batch_size)
                            total_start = perf_counter()
                            try:
                                if context_key in input_context_error_cache:
                                    raise input_context_error_cache[context_key]
                                if context_key not in input_context_cache:
                                    input_context_cache[context_key] = build_original_context(
                                        model_bundle,
                                        input_path,
                                        text,
                                        run_args.batch_size,
                                    )
                                input_context = input_context_cache[context_key]
                                reports.append(
                                    run_strategy_comparison(
                                        strategy_run_config,
                                        run_args,
                                        device,
                                        model_bundle,
                                        input_context,
                                        output_dir,
                                    )
                                )
                            except Exception as error:
                                if context_key not in input_context_cache:
                                    input_context_error_cache[context_key] = error
                                failure_report = build_failure_report(
                                    run_args=run_args,
                                    run_config=strategy_run_config,
                                    device=device,
                                    model_bundle=model_bundle,
                                    total_start=total_start,
                                    error=error,
                                )
                                failure_report["result_paths"] = write_failure_result(
                                    output_dir,
                                    strategy_run_config,
                                    failure_report,
                                )
                                reports.append(failure_report)

                        if device.type == "cuda":
                            import torch

                            torch.cuda.empty_cache()
    return reports


def main() -> int:
    args = parse_args()
    suites, output_name = build_suites(args)
    output_dir = suite_output_dir(output_name)
    suite_plans = [describe_suite(suite) for suite in suites]
    plan = suite_plans[0] if len(suite_plans) == 1 else {
        "name": output_name,
        "config_path": args.config,
        "suite_count": len(suite_plans),
        "planned_runs": sum(suite_plan["planned_runs"] for suite_plan in suite_plans),
        "suites": suite_plans,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    print(json.dumps({"benchmark_plan": plan}, indent=2), flush=True)
    metadata_suite = suites[0] if len(suites) == 1 else {"name": output_name, "suites": suites}
    metadata_paths = write_run_metadata(output_dir, args, metadata_suite, plan)
    reports = []
    for suite in suites:
        reports.extend(run_suite(args, suite, output_dir))

    if args.run_mode == "original":
        result_paths = write_original_summary(output_dir, reports)
        print(
            json.dumps(
                {
                    "original_pipeline_runs": len(reports),
                    "result_paths": result_paths,
                    "metadata_paths": metadata_paths,
                },
                indent=2,
            )
        )
        return 0
    elif len(reports) == 1:
        reports[0]["metadata_paths"] = metadata_paths
        print(json.dumps(reports[0], indent=2))
    else:
        result_paths = write_benchmark_summary(output_dir, reports)
        print(
            json.dumps(
                {
                    "benchmark_runs": len(reports),
                    "resumed_runs": sum(1 for report in reports if report.get("_resumed")),
                    "failed_strict_runs": sum(1 for report in reports if is_failed_strict_report(report)),
                    "result_paths": result_paths,
                    "metadata_paths": metadata_paths,
                },
                indent=2,
            )
        )

    return 1 if any(is_failed_strict_report(report) for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
