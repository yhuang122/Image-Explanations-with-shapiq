"""AID explanation-quality benchmark for migrated image-explanation games."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aid_outputs import (
    add_curve_context,
    append_rows,
    base_summary_row,
    existing_run_keys,
    interaction_value_path,
    keep_completed_summary_rows,
    output_paths,
    run_key_from_values,
    summarize_failure,
    write_run_metadata,
)
from aid_schema import (
    CURVE_FIELDS,
    MASKER_CHOICES,
    MODEL_PRESETS,
    SEGMENTER_CHOICES,
    SUMMARY_FIELDS,
)
from aid_suite import (
    build_suite,
    describe_suite,
    resolve_model,
    resolve_samples,
    suite_output_dir,
)

LOCAL_SRC_PACKAGE = "_benchmark_local_src"
LOCAL_SRC_DIR = Path(__file__).resolve().parents[3] / "src"


def configure_warnings() -> None:
    # pandas is pulled in indirectly by sklearn during ProxySHAP imports.
    # In this Windows environment, importing native pyarrow can crash the
    # process, and the benchmark does not use pandas' optional Arrow backend.
    sys.modules.setdefault("pyarrow", None)
    warnings.filterwarnings("ignore", message="Using a slow image processor.*")
    warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
    warnings.filterwarnings("ignore", message="Index FWBII is not a valid index.*")


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
    parser = argparse.ArgumentParser(description="Run AID explanation-quality benchmarks.")
    parser.add_argument("--config", help="JSON benchmark suite config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded benchmark plan and exit. Normal runs already print this plan before execution.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute runs even when result CSVs already exist.")
    parser.add_argument("--quiet", action="store_true", help="Suppress final JSON output.")
    parser.add_argument(
        "--input",
        help="Single image file or manifest-backed image directory. Relative paths use project root.",
    )
    parser.add_argument("--text", help="Required text input for single-image input.")
    parser.add_argument("--text-column", choices=("first_caption", "caption"), help="Manifest text column.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-preset", choices=sorted(MODEL_PRESETS))
    model_group.add_argument("--model-name", help="Custom HuggingFace model id.")
    parser.add_argument("--output-name", help="Override result folder name.")
    parser.add_argument("--method", help="Single method name for CLI runs.")
    parser.add_argument("--mode", help="Single explanation mode, such as shapley or banzhaf/0.5.")
    parser.add_argument("--order", type=int, choices=(1, 2), help="Single interaction order.")
    parser.add_argument("--budget", type=int, help="Explanation budget for a single CLI method.")
    parser.add_argument(
        "--approximation-type",
        choices=("original", "proxyshap"),
        default="original",
        help="FIxLIP approximation backend for a single CLI method.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--curve-points", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--cuda", action="store_true", default=None)
    parser.add_argument("--use-amp", action="store_true", default=None)
    parser.add_argument("--segmenter-strategy", choices=SEGMENTER_CHOICES)
    parser.add_argument("--slic-n-segments", type=int, default=49)
    parser.add_argument("--slic-compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=0.0)
    parser.add_argument("--gradient-guided-n-segments", type=int)
    parser.add_argument("--masker-strategy", choices=MASKER_CHOICES)
    parser.add_argument("--vision-blur-sigma", type=float, default=3.0)
    return parser.parse_args()


def resolve_device(use_cuda: bool):
    import torch

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --cuda but is not available.")
    return torch.device("cuda" if use_cuda else "cpu")


class ProcessorWithAttentionMask:
    """Delegate to a HF processor and fill missing attention masks for SigLIP-style outputs."""

    def __init__(self, processor):
        self._processor = processor

    def __call__(self, *args, **kwargs):
        outputs = self._processor(*args, **kwargs)
        if "attention_mask" not in outputs and "input_ids" in outputs:
            outputs["attention_mask"] = (outputs["input_ids"] != 1).long()
        return outputs

    def __getattr__(self, name: str):
        return getattr(self._processor, name)


def load_model_bundle(model_case: dict[str, Any], device) -> dict[str, Any]:
    from transformers import AutoModel, AutoProcessor

    start = perf_counter()
    model = AutoModel.from_pretrained(model_case["model_name"])
    model.to(device)
    model.eval()
    processor = ProcessorWithAttentionMask(AutoProcessor.from_pretrained(model_case["model_name"]))
    return {"model": model, "processor": processor, "model_load_runtime_s": perf_counter() - start}


def build_segmenter_config(strategy: dict[str, Any]):
    from shapiq.imputer.vision import GradientGuidedParams, SegmenterConfig, SlicParams

    return SegmenterConfig(
        strategy=strategy["segmenter_strategy"],
        slic=SlicParams(
            n_segments=strategy["slic_n_segments"],
            compactness=strategy["slic_compactness"],
            sigma=strategy["slic_sigma"],
        ),
        gradient_guided=GradientGuidedParams(n_segments=strategy["gradient_guided_n_segments"]),
    )


def build_masker_config(strategy: dict[str, Any]):
    from shapiq.imputer.vision import MaskerConfig

    config = MaskerConfig(strategy=strategy["masker_strategy"])
    if strategy["masker_strategy"] == "crossmodal_blur":
        config.vision_blur.sigma = strategy["crossmodal_blur_sigma"]
    elif strategy["masker_strategy"] == "vision_blur":
        config.vision_blur.sigma = strategy["vision_blur_sigma"]
    return config


def build_game(
    model_bundle: dict[str, Any],
    sample,
    strategy: dict[str, Any],
    batch_size: int,
    use_amp: bool,
):
    from shapiq.imputer.vision import VisionLanguageGame
    from shapiq.imputer.vision import VisionImputerFactory

    start = perf_counter()
    image = Image.open(sample.path).convert("RGB")
    imputer = VisionImputerFactory().build(
        model_bundle["model"],
        model_bundle["processor"],
        image,
        sample.text,
        segmenter_config=build_segmenter_config(strategy),
        masker_config=build_masker_config(strategy),
        use_amp=use_amp,
    )
    game = VisionLanguageGame(imputer, batch_size=batch_size)
    return game, perf_counter() - start


def should_use_crossmodal(game, method: dict[str, Any]) -> bool:
    return (
        method["sampler_name"].lower() == "banzhaf"
        and getattr(game, "n_players_image", 0) > 0
        and getattr(game, "n_players_text", 0) > 0
    )


def run_fixlip_explanation(game, method: dict[str, Any], random_state: int):
    fixlip = local_src_module("fixlip")
    kwargs = dict(method["proxy_params"])
    start = perf_counter()
    if should_use_crossmodal(game, method):
        approximator = fixlip.FIxLIP(
            n_players_image=game.n_players_image,
            n_players_text=game.n_players_text,
            mode="banzhaf",
            max_order=method["order"],
            p=float(method["sampler_p"]),
            random_state=random_state,
            sparse_regression=method["sparse_regression"],
        )
        interaction_values = approximator.approximate_crossmodal(
            game=game,
            budget=method["budget"],
            time_game=True,
            approximation_type=method["approximation_type"],
            **kwargs,
        )
    else:
        approximator = fixlip.FIxLIP(
            n_players=game.n_players,
            mode=method["sampler_name"],
            max_order=method["order"],
            p=0.5 if method["sampler_p"] is None else float(method["sampler_p"]),
            random_state=random_state,
            sparse_regression=method["sparse_regression"],
        )
        interaction_values = approximator.approximate(
            game=game,
            budget=method["budget"],
            time_game=True,
            approximation_type=method["approximation_type"],
            **kwargs,
        )
    game_runtime = None
    if hasattr(approximator, "time_game_start") and hasattr(approximator, "time_game_end"):
        game_runtime = approximator.time_game_end - approximator.time_game_start
    return interaction_values, perf_counter() - start, game_runtime


def first_order_attribution(iv, method: dict[str, Any]) -> np.ndarray:
    if method["order"] == 1:
        return np.asarray(iv.get_n_order(1).values, dtype=float)

    utils = local_src_module("utils")
    p_sampler = 0.5 if method["sampler_p"] is None else float(method["sampler_p"])
    first_order = utils.convert_iv_to_first_order(iv, p_sampler=p_sampler)
    return np.asarray(first_order.get_n_order(1).values, dtype=float)


def sorted_deletion_curves(game, attribution_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(attribution_values)
    empty = np.asarray(game.empty_coalition, dtype=bool)
    mif_coalitions = np.stack([attribution_values <= value for value in sorted_values[::-1]] + [empty])
    lif_coalitions = np.stack([attribution_values >= value for value in sorted_values] + [empty])
    return game.value_function(mif_coalitions), game.value_function(lif_coalitions)


def interaction_deletion_curves(game, iv, attribution_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clique = local_src_module("clique")
    start_players = None
    if game.n_players > 100:
        start_players = clique.get_interesting_starting_players(
            attribution_values=attribution_values,
            first_order_values=iv.get_n_order(1).values,
            k=min(19, game.n_players),
        )
    mif_coalitions, lif_coalitions = clique.get_cliques_greedy_mif_lif(
        iv=iv,
        start_players=start_players,
        verbose=False,
    )
    empty = np.asarray([game.empty_coalition], dtype=bool)
    mif_inputs = np.concatenate((np.asarray(mif_coalitions, dtype=bool), empty), axis=0)
    lif_inputs = np.concatenate((np.asarray(lif_coalitions, dtype=bool), empty), axis=0)
    return game.value_function(mif_inputs), game.value_function(lif_inputs)


def normalize_outputs(values: np.ndarray, empty_output: float, full_output: float) -> np.ndarray:
    denominator = full_output - empty_output
    if abs(denominator) < 1e-12:
        raise ValueError("Cannot normalize AID curves because full and empty outputs are equal.")
    return (values - empty_output) / denominator


def curve_stats(mif_norm: np.ndarray, lif_norm: np.ndarray) -> dict[str, float]:
    x = np.linspace(0.0, 1.0, num=len(mif_norm))
    mif_auc = float(np.trapezoid(mif_norm, x))
    lif_auc = float(np.trapezoid(lif_norm, x))
    return {
        "aid_area_between_curves": lif_auc - mif_auc,
        "aid_mean_gap": float(np.mean(lif_norm - mif_norm)),
        "mif_deletion_auc": mif_auc,
        "lif_deletion_auc": lif_auc,
    }


def interpolate_curve(raw: np.ndarray, normalized: np.ndarray, curve_points: int) -> list[dict[str, float]]:
    old_x = np.linspace(0.0, 1.0, num=len(normalized))
    new_x = np.linspace(0.0, 1.0, num=curve_points)
    norm_interp = np.interp(new_x, old_x, normalized)
    raw_interp = np.interp(new_x, old_x, raw)
    return [
        {
            "curve_step": step,
            "removed_fraction": float(x_value),
            "normalized_output": float(norm_value),
            "raw_output": float(raw_value),
        }
        for step, (x_value, norm_value, raw_value) in enumerate(zip(new_x, norm_interp, raw_interp))
    ]


def aid_curve_rows(
    mif_raw: np.ndarray,
    lif_raw: np.ndarray,
    baseline_mif: np.ndarray,
    baseline_lif: np.ndarray,
    empty_output: float,
    full_output: float,
    curve_points: int,
) -> list[dict[str, Any]]:
    curves = []
    for curve_source, curve_name, raw in (
        ("interaction", "most_important_first_deletion", mif_raw),
        ("interaction", "least_important_first_deletion", lif_raw),
        ("first_order_baseline", "most_important_first_deletion", baseline_mif),
        ("first_order_baseline", "least_important_first_deletion", baseline_lif),
    ):
        normalized = normalize_outputs(raw, empty_output, full_output)
        for row in interpolate_curve(raw, normalized, curve_points):
            curves.append({"curve_source": curve_source, "curve_name": curve_name, **row})
    return curves


def evaluate_aid(game, iv, method: dict[str, Any], curve_points: int) -> dict[str, Any]:
    if iv.n_players != game.n_players:
        raise ValueError(f"InteractionValues players ({iv.n_players}) do not match game players ({game.n_players}).")

    start = perf_counter()
    attribution_values = first_order_attribution(iv, method)
    baseline_mif, baseline_lif = sorted_deletion_curves(game, attribution_values)
    if method["order"] == 1:
        mif_raw, lif_raw = baseline_mif, baseline_lif
    else:
        mif_raw, lif_raw = interaction_deletion_curves(game, iv, attribution_values)

    if not np.isclose(mif_raw[0], lif_raw[0]) or not np.isclose(mif_raw[-1], lif_raw[-1]):
        raise ValueError("MIF and LIF curves have inconsistent full or empty anchors.")

    full_output = float(mif_raw[0])
    empty_output = float(mif_raw[-1])
    mif_norm = normalize_outputs(mif_raw, empty_output, full_output)
    lif_norm = normalize_outputs(lif_raw, empty_output, full_output)
    baseline_mif_norm = normalize_outputs(baseline_mif, empty_output, full_output)
    baseline_lif_norm = normalize_outputs(baseline_lif, empty_output, full_output)

    actual_stats = curve_stats(mif_norm, lif_norm)
    baseline_stats = curve_stats(baseline_mif_norm, baseline_lif_norm)
    return {
        **actual_stats,
        "baseline_aid_area_between_curves": baseline_stats["aid_area_between_curves"],
        "baseline_aid_mean_gap": baseline_stats["aid_mean_gap"],
        "baseline_mif_deletion_auc": baseline_stats["mif_deletion_auc"],
        "baseline_lif_deletion_auc": baseline_stats["lif_deletion_auc"],
        "empty_coalition_output": empty_output,
        "full_coalition_output": full_output,
        "normalization_denominator": full_output - empty_output,
        "curve_evaluation_runtime_s": perf_counter() - start,
        "curves": aid_curve_rows(
            mif_raw,
            lif_raw,
            baseline_mif,
            baseline_lif,
            empty_output,
            full_output,
            curve_points,
        ),
    }


def pending_methods_for_run(
    completed_keys: set[tuple[str, ...]],
    sample,
    model_case: dict[str, Any],
    strategy: dict[str, Any],
    methods: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    pending = []
    resumed = 0
    for method in methods:
        key_values = {
            "sample_id": sample.sample_id,
            "model_preset": model_case["model_preset"],
            "model_name": model_case["model_name"],
            "strategy_name": strategy["strategy_name"],
            "method_name": method["method_name"],
            "mode": method["mode"],
            "order": method["order"],
            "approximation_type": method["approximation_type"],
            "explanation_budget": method["budget"],
        }
        if run_key_from_values(key_values) in completed_keys:
            resumed += 1
        else:
            pending.append(method)
    return pending, resumed


def update_success_row(row: dict[str, Any], game, iv, result: dict[str, Any], interaction_load_runtime_s: float) -> None:
    metrics = (
        "empty_coalition_output",
        "full_coalition_output",
        "normalization_denominator",
        "aid_area_between_curves",
        "aid_mean_gap",
        "mif_deletion_auc",
        "lif_deletion_auc",
        "baseline_aid_area_between_curves",
        "baseline_aid_mean_gap",
        "baseline_mif_deletion_auc",
        "baseline_lif_deletion_auc",
        "curve_evaluation_runtime_s",
    )
    row.update({name: result[name] for name in metrics})
    row.update(
        n_players=int(game.n_players),
        n_players_image=int(getattr(game, "n_players_image", 0)),
        n_players_text=int(getattr(game, "n_players_text", 0)),
        estimation_budget=getattr(iv, "estimation_budget", None),
        interaction_value_load_runtime_s=interaction_load_runtime_s,
        total_runtime_s=row.get("total_runtime_s"),
    )


def run_method(
    suite: dict[str, Any],
    sample,
    model_case: dict[str, Any],
    model_bundle: dict[str, Any],
    strategy: dict[str, Any],
    method: dict[str, Any],
    game,
    game_build_runtime_s: float | None,
    device,
    output_dir,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import shapiq

    total_start = perf_counter()
    interaction_path = interaction_value_path(output_dir, sample, model_case, strategy, method)
    row = base_summary_row(
        sample,
        model_case,
        strategy,
        method,
        interaction_path,
        device,
        suite["defaults"],
        model_bundle.get("model_load_runtime_s"),
        game_build_runtime_s,
    )
    cache_hit = interaction_path.exists() and not force
    if cache_hit:
        interaction_start = perf_counter()
        iv = shapiq.InteractionValues.load(interaction_path)
        interaction_load_runtime_s = perf_counter() - interaction_start
        explanation_runtime_s = 0.0
        explanation_game_runtime_s = 0.0
    else:
        iv, explanation_runtime_s, explanation_game_runtime_s = run_fixlip_explanation(
            game,
            method,
            suite["defaults"]["random_state"],
        )
        interaction_path.parent.mkdir(parents=True, exist_ok=True)
        iv.save(interaction_path)
        interaction_load_runtime_s = 0.0
    result = evaluate_aid(game, iv, method, suite["defaults"]["curve_points"])
    row["total_runtime_s"] = perf_counter() - total_start
    row["interaction_cache_hit"] = cache_hit
    row["explanation_runtime_s"] = explanation_runtime_s
    row["explanation_game_runtime_s"] = explanation_game_runtime_s
    update_success_row(row, game, iv, result, interaction_load_runtime_s)
    curve_rows = [add_curve_context(row, curve) for curve in result["curves"]]
    return row, curve_rows


def failure_row(
    suite: dict[str, Any],
    sample,
    model_case: dict[str, Any],
    model_bundle: dict[str, Any],
    strategy: dict[str, Any],
    method: dict[str, Any],
    game_build_runtime_s: float | None,
    device,
    output_dir,
    error: Exception,
    total_runtime_s: float,
) -> dict[str, Any]:
    interaction_path = interaction_value_path(output_dir, sample, model_case, strategy, method)
    row = base_summary_row(
        sample,
        model_case,
        strategy,
        method,
        interaction_path,
        device,
        suite["defaults"],
        model_bundle.get("model_load_runtime_s"),
        game_build_runtime_s,
    )
    return summarize_failure(row, error, total_runtime_s)


def run_suite(args: argparse.Namespace, suite: dict[str, Any], output_dir) -> dict[str, Any]:
    utils = local_src_module("utils")
    utils.set_seed(suite["defaults"]["random_state"])
    device = resolve_device(suite["defaults"]["cuda"])
    paths = output_paths(output_dir)
    plots_dir = paths["plots_dir"]
    summary_path = paths["summary_csv"]
    curves_path = paths["curves_csv"]
    if args.force:
        for path in (summary_path, curves_path):
            if path.exists():
                path.unlink()
    else:
        keep_completed_summary_rows(summary_path)

    completed_keys = existing_run_keys(summary_path)
    model_cache: dict[str, dict[str, Any]] = {}
    run_count = resumed_count = failed_count = 0

    for raw_model in suite["models"]:
        model_case = resolve_model(raw_model)
        model_key = model_case["model_name"]
        for input_value in suite["inputs"]:
            samples = resolve_samples(input_value, suite["single_text"], suite["defaults"]["text_column"])
            for sample in samples:
                for strategy in suite["strategies"]:
                    pending_methods, resumed = pending_methods_for_run(
                        completed_keys,
                        sample,
                        model_case,
                        strategy,
                        suite["methods"],
                    )
                    resumed_count += resumed
                    if not pending_methods:
                        continue

                    game = None
                    game_build_runtime_s = None
                    game_error = None
                    try:
                        if model_key not in model_cache:
                            model_cache[model_key] = load_model_bundle(model_case, device)
                        model_bundle = model_cache[model_key]
                        game, game_build_runtime_s = build_game(
                            model_bundle,
                            sample,
                            strategy,
                            suite["defaults"]["batch_size"],
                            suite["defaults"]["use_amp"],
                        )
                    except Exception as error:
                        model_bundle = model_cache.get(model_key, {"model_load_runtime_s": None})
                        game_error = error

                    for method in pending_methods:
                        total_start = perf_counter()
                        curve_rows = []
                        try:
                            if game_error is not None:
                                raise game_error
                            row, curve_rows = run_method(
                                suite,
                                sample,
                                model_case,
                                model_bundle,
                                strategy,
                                method,
                                game,
                                game_build_runtime_s,
                                device,
                                output_dir,
                                args.force,
                            )
                        except Exception as error:
                            failed_count += 1
                            row = failure_row(
                                suite,
                                sample,
                                model_case,
                                model_bundle,
                                strategy,
                                method,
                                game_build_runtime_s,
                                device,
                                output_dir,
                                error,
                                perf_counter() - total_start,
                            )

                        append_rows(summary_path, SUMMARY_FIELDS, [row])
                        append_rows(curves_path, CURVE_FIELDS, curve_rows)
                        run_count += 1

                    del game

                if device.type == "cuda":
                    import torch

                    torch.cuda.empty_cache()

    from aid_plots import write_aid_plots

    plot_paths = write_aid_plots(summary_path, curves_path, plots_dir)
    return {
        "aid_quality_runs": run_count,
        "resumed_runs": resumed_count,
        "failed_runs": failed_count,
        "result_paths": {
            "summary_csv": str(summary_path),
            "curves_csv": str(curves_path),
            **plot_paths,
        },
    }


def main() -> int:
    configure_warnings()
    args = parse_args()
    suite = build_suite(args)
    output_dir = suite_output_dir(suite)
    plan = describe_suite(suite)
    if args.dry_run:
        if not args.quiet:
            print(json.dumps(plan, indent=2))
        return 0

    if not args.quiet:
        print(json.dumps({"benchmark_plan": plan}, indent=2), flush=True)
    metadata_paths = write_run_metadata(output_dir, args, suite, plan)
    result = run_suite(args, suite, output_dir)
    result["metadata_paths"] = metadata_paths
    if not args.quiet:
        print(json.dumps(result, indent=2))
    return 1 if result["failed_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
