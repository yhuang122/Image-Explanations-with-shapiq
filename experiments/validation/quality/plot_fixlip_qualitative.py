"""Plot one cached FIxLIP explanation with the repository's paper-style visualizer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from shapiq import InteractionValues

from aid_outputs import CSV_DIRNAME, output_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a FIxLIP qualitative figure from aid_summary.csv.")
    parser.add_argument("--summary", required=True, help="Path to aid_summary.csv.")
    parser.add_argument("--run-id", help="Completed run_id to plot. Defaults to the highest-order completed run.")
    parser.add_argument("--output", help="Output PNG path. Defaults to <suite>/plots/qualitative/<run_id>_fixlip_paper_style.png.")
    parser.add_argument("--top-k", type=int, default=14, help="Number of cross-modal interaction edges to draw.")
    parser.add_argument("--fontsize", type=int, default=22, help="Token font size.")
    return parser.parse_args()


def main() -> None:
    configure_warnings()
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    row = select_summary_row(summary_path, args.run_id)
    interaction_path = resolve_existing_path(row["interaction_value_path"], summary_path.parent)
    image_path = resolve_existing_path(row["input_path"], summary_path.parent)
    output_path = Path(args.output).resolve() if args.output else default_output_path(summary_path, row["run_id"])

    interaction_values = InteractionValues.load(str(interaction_path))
    remove_baseline_from_color_scale(interaction_values)

    image = Image.open(image_path).convert("RGB")
    image_array, tokens = prepare_visual_inputs(
        image=image,
        text=row.get("text", ""),
        model_name=row.get("model_name", ""),
        n_text=parse_int(row, "n_players_text"),
    )
    write_paper_style_plot(
        image_array=image_array,
        tokens=tokens,
        interaction_values=interaction_values,
        row=row,
        output_path=output_path,
        top_k=args.top_k,
        fontsize=args.fontsize,
    )
    print(json.dumps({"run_id": row["run_id"], "output_path": str(output_path)}, indent=2))


def select_summary_row(summary_path: Path, run_id: str | None) -> dict[str, str]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV does not exist: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as file:
        completed = [
            row
            for row in csv.DictReader(file)
            if row.get("status") == "completed" and row.get("interaction_value_path")
        ]
    if not completed:
        raise ValueError(f"No completed rows with interaction_value_path in {summary_path}")
    if run_id:
        for row in completed:
            if row.get("run_id") == run_id:
                return row
        raise ValueError(f"run_id not found in completed rows: {run_id}")
    return max(completed, key=lambda row: (parse_int(row, "order"), parse_float(row, "aid_area_between_curves")))


def default_output_path(summary_path: Path, run_id: str) -> Path:
    suite_dir = summary_path.parent.parent if summary_path.parent.name == CSV_DIRNAME else summary_path.parent
    return output_paths(suite_dir)["plots_dir"] / "qualitative" / f"{run_id}_fixlip_paper_style.png"


def resolve_existing_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = (base_dir / value).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Path does not exist: {value}")


def prepare_visual_inputs(image: Image.Image, text: str, model_name: str, n_text: int) -> tuple[np.ndarray, list[str]]:
    try:
        from transformers import AutoProcessor

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using a slow image processor.*")
            processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
            encoded = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        image_array = denormalized_pixel_values(
            encoded["pixel_values"],
            processor.image_processor.image_mean,
            processor.image_processor.image_std,
        )
        tokens = tokenizer_tokens(processor, encoded, n_text)
    except Exception:
        image_array = np.asarray(image.resize((224, 224)), dtype=float) / 255.0
        tokens = fallback_tokens(text, n_text)
    return image_array, tokens


def configure_warnings() -> None:
    warnings.filterwarnings("ignore", message="Using a slow image processor.*")
    warnings.filterwarnings("ignore", message="Index FWBII is not a valid index.*")


def denormalized_pixel_values(pixel_values, mean: list[float], std: list[float]) -> np.ndarray:
    array = pixel_values.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    array = array * np.asarray(std).reshape(1, 1, 3) + np.asarray(mean).reshape(1, 1, 3)
    return np.clip(array, 0.0, 1.0)


def tokenizer_tokens(processor, encoded, n_text: int) -> list[str]:
    tokenizer = processor.tokenizer
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded.get("attention_mask")
    mask = attention_mask[0].tolist() if attention_mask is not None else [1] * len(input_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    tokens = [
        clean_token(token)
        for token_id, active, token in zip(input_ids, mask, tokenizer.convert_ids_to_tokens(input_ids))
        if active and token_id not in special_ids
    ]
    tokens = [token for token in tokens if token]
    return fit_token_count(tokens, n_text)


def fallback_tokens(text: str, n_text: int) -> list[str]:
    tokens = [token for token in text.split() if token]
    if n_text == 1 and text.strip():
        tokens = [text.strip()]
    return fit_token_count(tokens, n_text)


def fit_token_count(tokens: list[str], n_text: int) -> list[str]:
    if n_text <= 0:
        return tokens
    while len(tokens) < n_text:
        tokens.append("")
    return tokens[:n_text]


def clean_token(token: str) -> str:
    return token.replace("</w>", "").replace("▁", "").strip()


def remove_baseline_from_color_scale(interaction_values) -> None:
    index = interaction_values.interaction_lookup.get(())
    if index is not None:
        interaction_values.values[index] = 1e-10


def write_paper_style_plot(
    image_array: np.ndarray,
    tokens: list[str],
    interaction_values,
    row: dict[str, str],
    output_path: Path,
    top_k: int,
    fontsize: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plot import plot_slicimage_and_text_together

    n_image = parse_int(row, "n_players_image")
    fig = plot_slicimage_and_text_together(
        img=image_array,
        text=tokens,
        image_players=list(range(n_image)),
        iv=interaction_values,
        plot_interactions=True,
        plot_heatmap=True,
        top_k=top_k,
        figsize=(8, 8),
        fontsize=fontsize,
        show=False,
    )
    fig.suptitle(plot_title(row), fontsize=8, y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_title(row: dict[str, str]) -> str:
    fields = [
        row.get("model_preset", ""),
        row.get("strategy_name", ""),
        row.get("method_name", ""),
        f"AID={format_float(row.get('aid_area_between_curves', ''))}",
    ]
    return " | ".join(field for field in fields if field)


def parse_int(row: dict[str, str], field: str, default: int = 0) -> int:
    try:
        return int(float(row.get(field, "")))
    except (TypeError, ValueError):
        return default


def parse_float(row: dict[str, str], field: str, default: float = float("-inf")) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return default


def format_float(value: str) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    main()
