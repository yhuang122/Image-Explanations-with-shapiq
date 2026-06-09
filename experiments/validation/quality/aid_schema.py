"""Shared schema and configuration for AID quality benchmarks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


QUALITY_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = QUALITY_DIR.parent
PROJECT_ROOT = QUALITY_DIR.parents[2]
EQUIVALENCE_DIR = VALIDATION_DIR / "equivalence"
RESULTS_DIR = QUALITY_DIR / "results"
MANIFEST_FILENAME = "manifest.csv"
DEFAULT_RANDOM_STATE = 0

for path in (PROJECT_ROOT, EQUIVALENCE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from benchmark_schema import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PRESET,
    IMAGE_SUFFIXES,
    MASKER_CHOICES,
    MODEL_PRESETS,
    SEGMENTER_CHOICES,
    slug,
)


DEFAULT_STRATEGIES = (
    {"name": "patch_crossmodal_mean", "segmenter": "patch", "masker": "crossmodal_mean"},
    {
        "name": "slic_crossmodal_mean",
        "segmenter": "slic",
        "masker": "crossmodal_mean",
        "slic": {"n_segments": 49, "compactness": 10.0, "sigma": 0.0},
    },
)
DEFAULT_METHODS = (
    {
        "name": "kernelshap_shapley_order1",
        "mode": "shapley",
        "order": 1,
        "explainer_name": "fixlip",
        "approximation_type": "original",
        "budget": 4096,
    },
    {
        "name": "proxyshap_banzhaf_p03_order2",
        "mode": "banzhaf/0.3",
        "order": 2,
        "explainer_name": "fixlip",
        "approximation_type": "proxyshap",
        "budget": 4096,
    },
    {
        "name": "proxyshap_banzhaf_p05_order2",
        "mode": "banzhaf/0.5",
        "order": 2,
        "explainer_name": "fixlip",
        "approximation_type": "proxyshap",
        "budget": 4096,
    },
    {
        "name": "proxyshap_banzhaf_p07_order2",
        "mode": "banzhaf/0.7",
        "order": 2,
        "explainer_name": "fixlip",
        "approximation_type": "proxyshap",
        "budget": 4096,
    },
)

RUN_KEY_FIELDS = (
    "sample_id",
    "model_preset",
    "model_name",
    "strategy_name",
    "method_name",
    "mode",
    "order",
    "approximation_type",
    "explanation_budget",
)
SUMMARY_FIELDS = (
    "run_id",
    "status",
    "passed",
    "error_message",
    "quality_scope",
    "quality_metric",
    "sample_id",
    "sample_index",
    "source_dataset",
    "source_key",
    "input_path",
    "text",
    "text_full",
    "text_source",
    "model_preset",
    "model_name",
    "strategy_name",
    "segmenter_strategy",
    "segmenter_params",
    "masker_strategy",
    "masker_params",
    "explainer_name",
    "method_name",
    "mode",
    "sampler_name",
    "sampler_p",
    "order",
    "approximation_type",
    "explanation_budget",
    "interaction_value_path",
    "interaction_cache_hit",
    "device",
    "use_amp",
    "batch_size",
    "curve_points",
    "random_state",
    "n_players",
    "n_players_image",
    "n_players_text",
    "estimation_budget",
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
    "model_load_runtime_s",
    "game_build_runtime_s",
    "explanation_runtime_s",
    "explanation_game_runtime_s",
    "interaction_value_load_runtime_s",
    "curve_evaluation_runtime_s",
    "total_runtime_s",
)
CURVE_FIELDS = (
    "run_id",
    "quality_scope",
    "sample_id",
    "sample_index",
    "input_path",
    "text",
    "model_preset",
    "model_name",
    "strategy_name",
    "method_name",
    "mode",
    "order",
    "curve_source",
    "curve_name",
    "curve_step",
    "removed_fraction",
    "normalized_output",
    "raw_output",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sample_index: int
    path: Path
    text: str
    text_full: str
    text_source: str
    source_dataset: str = ""
    source_key: str = ""
