"""Plot helpers for AID quality benchmark outputs."""

from __future__ import annotations

from pathlib import Path
import numpy as np

def write_aid_plots(summary_path: Path, curves_path: Path, plots_dir: Path) -> dict[str, str]:
    if not summary_path.exists() or not curves_path.exists():
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    curves = pd.read_csv(curves_path)
    completed = summary[summary["status"] == "completed"].copy()
    if completed.empty:
        return {}

    for column in ("aid_area_between_curves", "total_runtime_s", "explanation_runtime_s"):
        completed[column] = pd.to_numeric(completed[column], errors="coerce")

    score = aggregate_scores(completed)
    score_plot = plots_dir / "quality_aid_score_mean_std.png"
    _write_score_plot(
        score_plot,
        score,
    )

    curve_plots = _write_mean_curve_plots(curves, plots_dir)
    sample_curve_plots = _write_sample_curve_grid_plots(completed, curves, plots_dir)

    scatter_plot = plots_dir / "quality_aid_quality_runtime_tradeoff.png"
    _write_runtime_scatter(scatter_plot, score)

    coverage_plot = plots_dir / "quality_aid_coverage_table.png"
    _write_coverage_table(coverage_plot, summary)

    plot_paths = {
        "score_plot": str(score_plot),
        "runtime_tradeoff_plot": str(scatter_plot),
        "coverage_table": str(coverage_plot),
    }
    if curve_plots:
        plot_paths["curve_plots"] = [str(path) for path in curve_plots]
    if sample_curve_plots:
        plot_paths["sample_curve_grid_plots"] = [str(path) for path in sample_curve_plots]
    return plot_paths


def aggregate_scores(completed):
    score = (
        completed.groupby(["model_preset", "strategy_name", "method_name"], dropna=False)
        .agg(
            mean_aid_area=("aid_area_between_curves", "mean"),
            std_aid_area=("aid_area_between_curves", "std"),
            mean_total_runtime_s=("total_runtime_s", "mean"),
            mean_explanation_runtime_s=("explanation_runtime_s", "mean"),
            runs=("run_id", "count"),
        )
        .reset_index()
        .sort_values("mean_aid_area", ascending=False)
    )
    score["std_aid_area"] = score["std_aid_area"].fillna(0.0)
    score["label"] = score["model_preset"] + "\n" + score["strategy_name"] + "\n" + score["method_name"]
    return score


def _write_score_plot(path: Path, score) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(9, len(score) * 0.6), 5.5))
    ax.bar(score["label"], score["mean_aid_area"], yerr=score["std_aid_area"], color="#4477aa", capsize=3)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("AID Quality: Mean Area Between LIF and MIF Curves")
    ax.set_ylabel("Mean AID area +/- std; higher is better")
    ax.set_xlabel("Model / strategy / method")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_mean_curve_plots(curves, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    interaction_curves = curves[curves["curve_source"] == "interaction"].copy()
    if interaction_curves.empty:
        return []

    interaction_curves["removed_fraction"] = pd.to_numeric(
        interaction_curves["removed_fraction"],
        errors="coerce",
    )
    interaction_curves["normalized_output"] = pd.to_numeric(
        interaction_curves["normalized_output"],
        errors="coerce",
    )
    mean_curves = (
        interaction_curves.groupby(
            ["model_preset", "strategy_name", "method_name", "curve_name", "removed_fraction"],
            dropna=False,
        )["normalized_output"]
        .mean()
        .reset_index()
    )

    paths = []
    for (model_preset, strategy_name), subset in mean_curves.groupby(["model_preset", "strategy_name"], dropna=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, group in subset.groupby(["method_name", "curve_name"]):
            method_name, curve_name = key
            line_style = "--" if curve_name.startswith("least") else "-"
            label = f"{method_name} | {curve_name}"
            ax.plot(group["removed_fraction"], group["normalized_output"], line_style, linewidth=1.5, label=label)
        ax.set_title(f"AID Quality: Mean Deletion Curves\n{model_preset} / {strategy_name}")
        ax.set_xlabel("Removed player fraction")
        ax.set_ylabel("Normalized model output")
        y_min = min(-0.05, float(mean_curves["normalized_output"].quantile(0.01)) - 0.05)
        y_max = max(1.05, float(mean_curves["normalized_output"].quantile(0.99)) + 0.05)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=7, ncol=1, loc="best")
        fig.tight_layout()
        path = plots_dir / f"quality_aid_mean_deletion_curves_{_slug(model_preset)}_{_slug(strategy_name)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_sample_curve_grid_plots(summary, curves, plots_dir: Path, max_samples: int = 10) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    interaction_curves = curves[curves["curve_source"] == "interaction"].copy()
    if interaction_curves.empty:
        return []

    for column in ("removed_fraction", "normalized_output"):
        interaction_curves[column] = pd.to_numeric(interaction_curves[column], errors="coerce")
    summary = summary.copy()
    for column in ("sample_index", "aid_area_between_curves", "n_players"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    paths = []
    group_fields = ["model_preset", "strategy_name", "method_name"]
    for group_key, group_summary in summary.groupby(group_fields, dropna=False):
        group_summary = group_summary.sort_values(["sample_index", "run_id"]).head(max_samples)
        if group_summary.empty:
            continue

        n_samples = len(group_summary)
        n_cols = 5 if n_samples > 5 else max(1, n_samples)
        n_rows = 2 if n_samples > 5 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        for position, (axis, (_, row)) in enumerate(zip(axes, group_summary.iterrows())):
            run_curves = interaction_curves[interaction_curves["run_id"] == row["run_id"]]
            draw_sample_curve(axis, run_curves, row, position)

        for axis in axes[len(group_summary) :]:
            axis.set_visible(False)

        model_preset, strategy_name, method_name = group_key
        fig.suptitle(
            f"AID Curves - {n_samples} samples\n{model_preset} / {strategy_name}\n{method_name}",
            fontsize=10 if n_cols < 3 else 13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        path = plots_dir / (
            "quality_aid_sample_curves_"
            f"{_slug(model_preset)}_{_slug(strategy_name)}_{_slug(method_name)}.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def draw_sample_curve(axis, curves, summary_row, position: int) -> None:
    mif = curves[curves["curve_name"] == "most_important_first_deletion"].sort_values("removed_fraction")
    lif = curves[curves["curve_name"] == "least_important_first_deletion"].sort_values("removed_fraction")
    if mif.empty or lif.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "Missing curve", ha="center", va="center")
        return

    x = mif["removed_fraction"].to_numpy(dtype=float)
    mif_y = mif["normalized_output"].to_numpy(dtype=float)
    lif_y = lif["normalized_output"].to_numpy(dtype=float)

    axis.plot(x, mif_y, "r-", linewidth=1.5, label="MIF")
    axis.plot(x, lif_y, "b-", linewidth=1.5, label="LIF")
    axis.fill_between(x, lif_y, mif_y, where=(lif_y >= mif_y), alpha=0.12, color="green")
    axis.fill_between(x, lif_y, mif_y, where=(lif_y < mif_y), alpha=0.12, color="red")

    axis.set_xlim(-0.02, 1.02)

    finite_values = np.concatenate(
        [
            mif_y[np.isfinite(mif_y)],
            lif_y[np.isfinite(lif_y)],
        ]
    )

    if finite_values.size:
        y_min = min(0.0, float(finite_values.min())) - 0.05
        y_max = max(1.0, float(finite_values.max())) + 0.05
        if y_max - y_min < 0.1:
            y_min -= 0.05
            y_max += 0.05
        axis.set_ylim(y_min, y_max)
    else:
        axis.set_ylim(-0.05, 1.05)

    axis.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    axis.axhline(1.0, color="gray", linewidth=0.6, alpha=0.5)

    axis.set_title(
        f"#{int(summary_row['sample_index'])}  "
        f"AID={float(summary_row['aid_area_between_curves']):.3f}  "
        f"({int(summary_row['n_players'])}p)",
        fontsize=9,
    )
    if position >= 5:
        axis.set_xlabel("Fraction removed")
    if position % 5 == 0:
        axis.set_ylabel("Norm. score")

def _write_runtime_scatter(path: Path, score) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(score["mean_explanation_runtime_s"], score["mean_aid_area"], s=70, color="#228833")
    for _, row in score.iterrows():
        ax.annotate(
            f"{row['model_preset']} / {row['strategy_name']}",
            (row["mean_explanation_runtime_s"], row["mean_aid_area"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("AID Quality vs Explanation Runtime")
    ax.set_xlabel("Mean explanation runtime (s)")
    ax.set_ylabel("Mean AID area; higher is better")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_coverage_table(path: Path, summary) -> None:
    import matplotlib.pyplot as plt

    coverage = (
        summary.groupby(["model_preset", "strategy_name", "method_name"], dropna=False)
        .agg(
            completed=("status", lambda values: int((values == "completed").sum())),
            failed=("status", lambda values: int((values == "failed").sum())),
            total=("run_id", "count"),
        )
        .reset_index()
    )
    coverage["success_rate"] = coverage["completed"] / coverage["total"]
    display = coverage.copy()
    display["success_rate"] = display["success_rate"].map(lambda value: f"{value:.1%}")
    display = display[["model_preset", "strategy_name", "method_name", "completed", "failed", "success_rate"]]

    fig_height = max(3.5, 0.35 * len(display) + 1.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=["Model", "Strategy", "Method", "Completed", "Failed", "Success Rate"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    ax.set_title("AID Quality Coverage Summary", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "value"
