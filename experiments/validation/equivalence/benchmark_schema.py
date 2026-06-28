"""Shared schema and configuration for validation benchmarks."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "validation" / "equivalence" / "results"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input" / "wds_mscoco_captions_test_100"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

MODEL_PRESETS = {
    "clip-vit-b-32": "openai/clip-vit-base-patch32",
    "clip-vit-b-16": "openai/clip-vit-base-patch16",
    "clip-vit-l-14": "openai/clip-vit-large-patch14",
    "siglip-base-p16-224": "google/siglip-base-patch16-224",
    "siglip2-base-p32-256": "google/siglip2-base-patch32-256",
    "siglip2-so400m-p14-384": "google/siglip2-so400m-patch14-384",
}
DEFAULT_MODEL_PRESET = "clip-vit-b-32"
MODEL_PRESET_BY_NAME = {model_name: preset for preset, model_name in MODEL_PRESETS.items()}

SEGMENTER_CHOICES = ("patch", "slic", "gradient_guided")
MASKER_CHOICES = (
    "crossmodal_mean",
    "crossmodal_blur",
    "vision_mean",
    "vision_blur",
    "text_attn",
    "attention",
)

ORIGINAL_PIPELINE = "src.game_huggingface.VisionLanguageGame"
MIGRATED_PIPELINE = "ImageImputerFactory + Game.VisionLanguageGame"

DEFAULT_STRATEGY_SPECS = (
    {
        "strategy_name": "patch_crossmodal_mean",
        "segmenter_strategy": "patch",
        "masker_strategy": "crossmodal_mean",
    },
    {
        "strategy_name": "patch_vision_mean",
        "segmenter_strategy": "patch",
        "masker_strategy": "vision_mean",
    },
    {
        "strategy_name": "patch_text_attn",
        "segmenter_strategy": "patch",
        "masker_strategy": "text_attn",
    },
    {
        "strategy_name": "slic_crossmodal_mean",
        "segmenter_strategy": "slic",
        "masker_strategy": "crossmodal_mean",
        "slic_n_segments": 49,
        "slic_compactness": 10.0,
        "slic_sigma": 0.0,
    },
)

CASES = {
    **{
        case_name: {"comparison_type": "1d", "model_preset": DEFAULT_MODEL_PRESET}
        for case_name in (
            "faithfulness",
            "insertion_deletion",
            "pointing_game_banzhaf",
            "pointing_game_shapley",
            "explain_mscoco",
        )
    },
    "insertion_deletion_siglip": {"comparison_type": "1d", "model_preset": "siglip-base-p16-224"},
    "pointing_game_crossmodal": {
        "comparison_type": "crossmodal",
        "model_preset": DEFAULT_MODEL_PRESET,
    },
    "explain_mscoco_siglip": {
        "comparison_type": "1d",
        "model_preset": "siglip2-base-p32-256",
    },
}

RUN_CONTEXT_FIELDS = (
    "case", "input_path", "strategy_name",
    "original_pipeline", "migrated_pipeline", "model_preset", "model_name",
    "model_type", "text", "text_full", "text_source", "comparison_type", "comparison_mode",
    "device", "use_amp",
)
COMPARISON_SCOPE_FIELDS = (
    "comparison_scope", "reference_name", "candidate_name",
    "equivalence_expected", "metric_family",
)
STRATEGY_FIELDS = (
    "segmenter_strategy", "segmenter_params", "masker_strategy", "masker_params",
)
LAYOUT_FIELDS = (
    "image_size", "patch_size", "grid_size", "text_total_length",
    "n_players", "n_players_image", "n_players_text",
    "original_n_players", "original_n_players_image", "original_n_players_text",
    "coalition_comparison_available",
)
PARAMETER_FIELDS = (
    "num_coalitions", "batch_size", "random_state", "tolerance",
)
RUNTIME_FIELDS = (
    "model_load_runtime_s", "original_game_build_runtime_s", "migrated_game_build_runtime_s",
    "original_anchor_runtime_s", "migrated_anchor_runtime_s", "build_runtime_s",
    "original_pipeline_runtime_s", "migrated_pipeline_runtime_s", "total_runtime_s",
)
ANCHOR_FIELDS = (
    "original_pipeline_empty_coalition_output",
    "migrated_pipeline_empty_coalition_output",
    "abs_empty_coalition_output_diff",
    "original_pipeline_full_coalition_output",
    "migrated_pipeline_full_coalition_output",
    "abs_full_coalition_output_diff",
    "empty_full_anchor_max_abs_output_diff",
    "empty_full_anchor_mean_abs_output_diff",
    "empty_full_anchor_passed",
)
DIFF_FIELDS = (
    "coalition_max_abs_output_diff", "coalition_mean_abs_output_diff",
    "benchmark_metric_max_output_diff", "benchmark_metric_mean_output_diff",
    "equivalence_passed", "passed", "strict_equivalence",
)
SUMMARY_FIELDS = (
    *RUN_CONTEXT_FIELDS,
    *COMPARISON_SCOPE_FIELDS,
    *STRATEGY_FIELDS,
    *LAYOUT_FIELDS,
    *PARAMETER_FIELDS,
    *RUNTIME_FIELDS,
    *ANCHOR_FIELDS,
    *DIFF_FIELDS,
)
BENCHMARK_SUMMARY_FIELDS = (*SUMMARY_FIELDS, "result_csv")
ROW_FIELDS = (
    "row_type", "coalition_index", "image_coalition_index", "text_coalition_index", "coalition_size",
    "image_coalition_size", "text_coalition_size",
    "original_pipeline_coalition_output",
    "migrated_pipeline_coalition_output",
    "abs_coalition_output_diff",
)


def slug(value: str, max_length: int = 48) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")[:max_length] or "empty"
