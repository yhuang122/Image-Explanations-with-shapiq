"""A2 value-function equivalence harness for migrated VLM experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "validation" / "results"
BASELINES_DIR = PROJECT_ROOT / "experiments" / "validation" / "baselines"
sys.path.insert(0, str(PROJECT_ROOT))

import src  # noqa: E402
from Game import VisionLanguageGame  # noqa: E402
from ImputerFactory import ImageImputerFactory  # noqa: E402

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

CASES = {
    **{
        case_name: {"comparison_type": "1d", "model_name": DEFAULT_CLIP_MODEL}
        for case_name in (
            "faithfulness",
            "insertion_deletion",
            "pointing_game_banzhaf",
            "pointing_game_shapley",
            "explain_mscoco",
        )
    },
    "insertion_deletion_siglip": {"comparison_type": "1d", "model_name": "google/siglip-base-patch16-224"},
    "pointing_game_crossmodal": {
        "comparison_type": "crossmodal",
        "model_name": DEFAULT_CLIP_MODEL,
    },
    "explain_mscoco_siglip": {
        "comparison_type": "1d",
        "model_name": "google/siglip2-base-patch32-256",
    },
}

SUMMARY_FIELDS = (
    "model_name", "n_players", "n_players_image", "n_players_text", "num_coalitions",
    "batch_size", "tolerance", "max_abs_diff", "mean_abs_diff", "passed",
)
ROW_FIELDS = (
    "coalition_index", "image_coalition_index", "text_coalition_index", "coalition_size",
    "image_coalition_size", "text_coalition_size", "reference_value", "candidate_value", "abs_diff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old and migrated VLM Game outputs.")
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--input", required=True, help="Image path. Relative paths use project root.")
    parser.add_argument("--text", required=True, help="Text input.")
    for name in ("random-state", "num-coalitions", "batch-size"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--cuda", action="store_true", help="Run comparison on CUDA. Defaults to CPU.")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    return parser.parse_args()


def resolve_input_path(input_path: str) -> Path:
    path = Path(input_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    return path


def resolve_device(use_cuda: bool) -> torch.device:
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


def build_games(case: dict, device: torch.device):
    image = Image.open(case["input_path"]).convert("RGB")
    text = case["text"]

    model = AutoModel.from_pretrained(case["model_name"])
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(case["model_name"])

    old_game = src.game_huggingface.VisionLanguageGame(
        model, processor, input_image=image, input_text=text, batch_size=case["batch_size"]
    )
    imputer = ImageImputerFactory().build(model, processor, image, text, segmenter=None, masker=None)
    new_game = VisionLanguageGame(imputer, batch_size=case["batch_size"])

    return old_game, new_game


def assert_same_layout(old_game, new_game) -> None:
    fields = ("n_players", "n_players_image", "n_players_text")
    mismatches = {
        field: (getattr(old_game, field), getattr(new_game, field))
        for field in fields
        if getattr(old_game, field) != getattr(new_game, field)
    }
    if mismatches:
        raise AssertionError(f"Player layout mismatch: {mismatches}")


def build_comparison_inputs(old_game, case: dict) -> dict:
    if case["comparison_type"] == "crossmodal":
        return {
            "comparison_type": "crossmodal",
            "image_coalitions": generate_coalitions(
                old_game.n_players_image, case["num_coalitions"], case["random_state"]
            ),
            "text_coalitions": generate_coalitions(
                old_game.n_players_text, case["num_coalitions"], case["random_state"] + 1
            ),
        }
    return {
        "comparison_type": "1d",
        "coalitions": generate_coalitions(old_game.n_players, case["num_coalitions"], case["random_state"]),
    }


def evaluate_game(game, inputs: dict) -> np.ndarray:
    if inputs["comparison_type"] == "crossmodal":
        values = game.value_function_crossmodal(
            inputs["image_coalitions"],
            inputs["text_coalitions"],
        )
        return values.reshape(-1)
    return game.value_function(inputs["coalitions"])


def value_row(index: int, reference: float, candidate: float, diff: float, **coalition_fields) -> dict:
    row = {"coalition_index": index, **coalition_fields}
    row.update(reference_value=float(reference), candidate_value=float(candidate), abs_diff=float(diff))
    return row


def iter_result_rows(inputs: dict, reference_values: np.ndarray, candidate_values: np.ndarray):
    diffs = np.abs(reference_values - candidate_values)
    if inputs["comparison_type"] == "crossmodal":
        image_coalitions = inputs["image_coalitions"]
        text_coalitions = inputs["text_coalitions"]
        for index, (image_index, text_index) in enumerate(np.ndindex(len(image_coalitions), len(text_coalitions))):
            image_coalition = image_coalitions[image_index]
            text_coalition = text_coalitions[text_index]
            yield value_row(
                index,
                reference_values[index],
                candidate_values[index],
                diffs[index],
                image_coalition_index=image_index,
                text_coalition_index=text_index,
                coalition_size=int(np.sum(image_coalition) + np.sum(text_coalition)),
                image_coalition_size=int(np.sum(image_coalition)),
                text_coalition_size=int(np.sum(text_coalition)),
            )
        return

    for index, (coalition, reference, candidate, diff) in enumerate(
        zip(inputs["coalitions"], reference_values, candidate_values, diffs)
    ):
        yield value_row(
            index,
            reference,
            candidate,
            diff,
            image_coalition_index="",
            text_coalition_index="",
            coalition_size=int(np.sum(coalition)),
            image_coalition_size="",
            text_coalition_size="",
        )


def run_name(case: dict) -> str:
    text_hash = hashlib.sha1(case["text"].encode("utf-8")).hexdigest()[:8]
    return f"{case['case']}_{case['input_path'].stem}_text{text_hash}_rs{case['random_state']}_n{case['num_coalitions']}"


def write_baseline(path: Path, case_name: str, case: dict, inputs: dict, old_values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "old_values": old_values,
        "metadata": json.dumps(
            {
                "case": case_name,
                "comparison_type": case["comparison_type"],
                "input_path": str(case["input_path"]),
                "model_name": case["model_name"],
                "text": case["text"],
                "random_state": case["random_state"],
            }
        ),
    }
    coalition_keys = ("image_coalitions", "text_coalitions") if inputs["comparison_type"] == "crossmodal" else ("coalitions",)
    payload.update({key: inputs[key] for key in coalition_keys})
    np.savez_compressed(path, **payload)


def load_baseline(path: Path) -> tuple[dict, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")
    baseline = np.load(path, allow_pickle=False)
    has_crossmodal = "image_coalitions" in baseline.files and "text_coalitions" in baseline.files
    inputs = (
        {
            "comparison_type": "crossmodal",
            "image_coalitions": baseline["image_coalitions"],
            "text_coalitions": baseline["text_coalitions"],
        }
        if has_crossmodal
        else {
            "comparison_type": "1d",
            "coalitions": baseline["coalitions"],
        }
    )
    return inputs, baseline["old_values"]


def write_results(case: dict, report: dict, inputs: dict, reference_values: np.ndarray, candidate_values: np.ndarray) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{run_name(case)}_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS + ROW_FIELDS)
        writer.writeheader()
        for result_row in iter_result_rows(inputs, reference_values, candidate_values):
            row = {field: report[field] for field in SUMMARY_FIELDS}
            row.update(result_row)
            writer.writerow(row)

    return {"csv": str(csv_path)}


def main() -> int:
    args = parse_args()
    case = dict(CASES[args.case])
    case.update(
        case=args.case,
        input_path=resolve_input_path(args.input),
        text=args.text,
        random_state=args.random_state,
        num_coalitions=args.num_coalitions,
        batch_size=args.batch_size,
    )

    device = resolve_device(args.cuda)
    src.utils.set_seed(case["random_state"])
    old_game, new_game = build_games(case, device)
    assert_same_layout(old_game, new_game)

    path = BASELINES_DIR / f"{run_name(case)}_game_outputs.npz"
    if args.compare_baseline:
        inputs, reference_values = load_baseline(path)
        comparison_mode = "baseline_vs_new"
    else:
        inputs = build_comparison_inputs(old_game, case)
        reference_values = evaluate_game(old_game, inputs)
        comparison_mode = "old_vs_new"

    candidate_values = evaluate_game(new_game, inputs)
    diff = np.abs(reference_values - candidate_values)
    result = {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "passed": bool(np.max(diff) <= args.tolerance),
    }

    if args.write_baseline:
        write_baseline(path, args.case, case, inputs, reference_values)

    report = {
        "case": args.case,
        "comparison_type": case["comparison_type"],
        "comparison_mode": comparison_mode,
        "baseline_path": str(path) if args.write_baseline or args.compare_baseline else None,
        "input_path": str(case["input_path"]),
        "model_name": case["model_name"],
        "device": str(device),
        "n_players": int(old_game.n_players),
        "n_players_image": int(old_game.n_players_image),
        "n_players_text": int(old_game.n_players_text),
        "num_coalitions": int(reference_values.shape[0]),
        "batch_size": int(case["batch_size"]),
        "tolerance": args.tolerance,
        **result,
    }
    report["result_paths"] = write_results(case, report, inputs, reference_values, candidate_values)
    print(json.dumps(report, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
