from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "validation" / "results"

SUMMARY_FIELDS = (
    "result_name",
    "model_name",
    "text",
    "n_players",
    "n_players_image",
    "n_players_text",
    "num_coalitions",
    "batch_size",
    "tolerance",
    "max_abs_diff",
    "mean_abs_diff",
    "passed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validation comparison CSV files.")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "summary.csv"),
        help="Summary CSV path. Defaults to experiments/validation/results/summary.csv.",
    )
    parser.add_argument(
        "--plot",
        default=str(RESULTS_DIR / "max_abs_diff_summary.png"),
        help="Summary plot path. Defaults to experiments/validation/results/max_abs_diff_summary.png.",
    )
    return parser.parse_args()


def result_name(path: Path) -> str:
    return path.stem.removesuffix("_comparison")


def text_from_result_name(name: str) -> str:
    if "_text_" not in name:
        return ""
    text_part = name.split("_text_", 1)[1]
    return text_part.rsplit("_rs", 1)[0].replace("_", " ")


def read_summary_row(path: Path) -> dict:
    name = result_name(path)
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    values = {field: row.get(field, "") for field in SUMMARY_FIELDS if field != "result_name"}
    values["text"] = values["text"] or text_from_result_name(name)
    return {
        "result_name": name,
        **values,
    }


def resolve_output_path(path: str) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_summary(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], output_path: Path) -> None:
    names = [plot_label(row) for row in rows]
    max_diffs = [float(row["max_abs_diff"]) for row in rows]
    tolerance = max(float(row["tolerance"]) for row in rows)
    upper_limit = max([tolerance, *max_diffs]) * 1.2
    lower_limit = -upper_limit * 0.05

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.5))
    ax.bar(names, max_diffs)
    ax.axhline(tolerance, color="red", linestyle="--", linewidth=1, label="tolerance")
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_ylabel("max_abs_diff")
    ax.set_xlabel("result")
    ax.set_title("Validation max_abs_diff summary")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_label(row: dict) -> str:
    name = row["result_name"]
    if row.get("text"):
        name = name.split("_text_", 1)[0]
        return f"{name}\ntext={row['text']}"
    return name


def main() -> int:
    args = parse_args()
    paths = sorted(RESULTS_DIR.glob("*_comparison.csv"))
    if not paths:
        raise FileNotFoundError(f"No comparison CSV files found in {RESULTS_DIR}")

    rows = [read_summary_row(path) for path in paths]
    output_path = resolve_output_path(args.output)
    plot_path = resolve_output_path(args.plot)
    write_summary(rows, output_path)
    write_plot(rows, plot_path)

    print(output_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
