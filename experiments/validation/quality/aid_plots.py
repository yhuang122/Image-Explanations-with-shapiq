"""Plot helpers for AID quality benchmark outputs."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def write_aid_plots(summary_path: Path, curves_path: Path, plots_dir: Path) -> dict[str, str]:
    if not summary_path.exists() or not curves_path.exists():
        return {}

    import matplotlib

    matplotlib.use("Agg")

    plots_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_sample_plots(plots_dir)
    summary = _read_csv_rows(summary_path)
    curves = _read_csv_rows(curves_path)
    _attach_curve_metadata(curves, summary)
    completed = [row for row in summary if row.get("status") == "completed"]
    if not completed:
        return {}

    score = aggregate_scores(completed)
    if not score:
        return {}

    score_plot = plots_dir / "quality_aid_score_mean_std.png"
    _write_score_plot(score_plot, score)

    curve_plots = _write_mean_curve_plots(curves, plots_dir)
    sample_curve_plots = _write_sample_curve_grid_plots(completed, curves, plots_dir)

    scatter_plot = plots_dir / "quality_aid_quality_runtime_tradeoff.png"
    _write_runtime_scatter(scatter_plot, score)

    coverage_plot = plots_dir / "quality_aid_coverage_table.png"
    _write_coverage_table(coverage_plot, summary)

    plot_paths: dict[str, Any] = {
        "score_plot": str(score_plot),
        "runtime_tradeoff_plot": str(scatter_plot),
        "coverage_table": str(coverage_plot),
    }
    if curve_plots:
        plot_paths["curve_plots"] = [str(path) for path in curve_plots]
    if sample_curve_plots:
        plot_paths["sample_curve_grid_plots"] = [str(path) for path in sample_curve_plots]
    return plot_paths


def aggregate_scores(completed: list[dict[str, str]]) -> list[dict[str, Any]]:
    scores = []
    for key, rows in _group_rows(
        completed,
        ("model_preset", "segmenter_strategy", "masker_strategy", "method_name", "explanation_budget"),
    ).items():
        model_preset, segmenter_strategy, masker_strategy, method_name, explanation_budget = key
        representative = rows[0]
        strategy_label = f"{segmenter_strategy} / {masker_strategy}"
        method_label = _method_label(representative)
        aid_values = _numeric_values(rows, "aid_area_between_curves")
        if not aid_values:
            continue
        scores.append(
            {
                "model_preset": model_preset,
                "segmenter_strategy": segmenter_strategy,
                "masker_strategy": masker_strategy,
                "strategy_label": strategy_label,
                "method_name": method_name,
                "method_label": method_label,
                "explanation_budget": explanation_budget,
                "mean_aid_area": _mean(aid_values),
                "std_aid_area": _std(aid_values),
                "mean_total_runtime_s": _mean(_numeric_values(rows, "total_runtime_s")),
                "mean_explanation_runtime_s": _mean(_numeric_values(rows, "explanation_runtime_s")),
                "runs": len(rows),
                "label": f"{model_preset}\n{strategy_label}\n{method_label}, budget={explanation_budget}",
            }
        )
    return sorted(scores, key=lambda row: row["mean_aid_area"], reverse=True)


def _write_score_plot(path: Path, score: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in score]
    means = [row["mean_aid_area"] for row in score]
    stds = [row["std_aid_area"] for row in score]
    y_positions = np.arange(len(score))
    fig_height = max(7.0, min(72.0, len(score) * 0.55))
    label_size = 8 if len(score) <= 40 else 6

    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(y_positions, means, xerr=stds, color="#4477aa", capsize=3)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=label_size)
    ax.invert_yaxis()
    ax.set_title("AID Quality: Mean Area Between LIF and MIF Curves")
    ax.set_xlabel("Mean AID area +/- std; higher is better")
    ax.set_ylabel("Model / segmenter / masker / method and explanation budget")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_mean_curve_plots(curves: list[dict[str, str]], plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    interaction_curves = [row for row in curves if row.get("curve_source") == "interaction"]
    if not interaction_curves:
        return []

    mean_rows = _mean_curve_rows(interaction_curves)
    normalized_values = [row["normalized_output"] for row in mean_rows if np.isfinite(row["normalized_output"])]
    if normalized_values:
        y_min = min(-0.05, float(np.quantile(normalized_values, 0.01)) - 0.05)
        y_max = max(1.05, float(np.quantile(normalized_values, 0.99)) + 0.05)
    else:
        y_min, y_max = -0.05, 1.05

    paths = []
    for (model_preset, strategy_name), subset in _group_rows(
        mean_rows,
        ("model_preset", "strategy_name"),
    ).items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for (method_label, explanation_budget, curve_name), group in _group_rows(
            subset,
            ("method_label", "explanation_budget", "curve_name"),
        ).items():
            sorted_group = sorted(group, key=lambda row: row["removed_fraction"])
            x_values = [row["removed_fraction"] for row in sorted_group]
            y_values = [row["normalized_output"] for row in sorted_group]
            line_style = "--" if curve_name.startswith("least") else "-"
            label = f"{method_label} | budget={explanation_budget} | {curve_name}"
            ax.plot(x_values, y_values, line_style, linewidth=1.5, label=label)
        ax.set_title(f"AID Quality: Mean Deletion Curves\n{model_preset} / {strategy_name}")
        ax.set_xlabel("Removed player fraction")
        ax.set_ylabel("Normalized model output")
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=7, ncol=1, loc="best")
        fig.tight_layout()
        path = plots_dir / _plot_filename(
            "quality_aid_mean_deletion_curves",
            model_preset,
            strategy_name,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_sample_curve_grid_plots(
    summary: list[dict[str, str]],
    curves: list[dict[str, str]],
    plots_dir: Path,
    max_samples: int = 10,
) -> list[Path]:
    import matplotlib.pyplot as plt

    interaction_curves = [row for row in curves if row.get("curve_source") == "interaction"]
    if not interaction_curves:
        return []

    curves_by_run = _group_rows(interaction_curves, ("run_id",))
    paths = []
    for group_key, group_summary in _group_rows(
        summary,
        ("model_preset", "strategy_name", "method_name", "method_label", "explanation_budget"),
    ).items():
        group_summary = sorted(
            group_summary,
            key=lambda row: (_to_int(row.get("sample_index")), row.get("run_id", "")),
        )[:max_samples]
        if not group_summary:
            continue

        n_samples = len(group_summary)
        n_cols = 5 if n_samples > 5 else max(1, n_samples)
        n_rows = 2 if n_samples > 5 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        for position, (axis, row) in enumerate(zip(axes, group_summary)):
            draw_sample_curve(axis, curves_by_run.get((row["run_id"],), []), row, position)

        for axis in axes[len(group_summary) :]:
            axis.set_visible(False)

        model_preset, strategy_name, method_name, method_label, explanation_budget = group_key
        fig.suptitle(
            f"AID Curves - {n_samples} samples\n"
            f"{model_preset} / {strategy_name}\n"
            f"{method_label} | explanation budget={explanation_budget}",
            fontsize=10 if n_cols < 3 else 13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        path = plots_dir / _plot_filename(
            "quality_aid_sample_curves",
            model_preset,
            strategy_name,
            method_name,
            f"budget{explanation_budget}",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def draw_sample_curve(axis, curves: list[dict[str, str]], summary_row: dict[str, str], position: int) -> None:
    mif = _curve_points(curves, "most_important_first_deletion")
    lif = _curve_points(curves, "least_important_first_deletion")
    if mif.size == 0 or lif.size == 0:
        axis.axis("off")
        axis.text(0.5, 0.5, "Missing curve", ha="center", va="center")
        return

    x = mif[:, 0]
    mif_y = mif[:, 1]
    lif_y = lif[:, 1]

    axis.plot(x, mif_y, "r-", linewidth=1.5, label="MIF")
    axis.plot(x, lif_y, "b-", linewidth=1.5, label="LIF")
    axis.fill_between(x, lif_y, mif_y, where=(lif_y >= mif_y), alpha=0.12, color="green")
    axis.fill_between(x, lif_y, mif_y, where=(lif_y < mif_y), alpha=0.12, color="red")

    axis.set_xlim(-0.02, 1.02)

    finite_values = np.concatenate([mif_y[np.isfinite(mif_y)], lif_y[np.isfinite(lif_y)]])
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
        f"#{_to_int(summary_row.get('sample_index'))}  "
        f"AID={_to_float(summary_row.get('aid_area_between_curves')):.3f}  "
        f"({_to_int(summary_row.get('n_players'))}p)",
        fontsize=9,
    )
    if position >= 5:
        axis.set_xlabel("Fraction removed")
    if position % 5 == 0:
        axis.set_ylabel("Norm. score")


def _write_runtime_scatter(path: Path, score: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in score]
    x_values = [row["mean_explanation_runtime_s"] for row in score]
    aid_values = [row["mean_aid_area"] for row in score]
    y_positions = np.arange(len(score))
    fig_height = max(7.0, min(72.0, len(score) * 0.55))
    label_size = 8 if len(score) <= 40 else 6

    fig, ax = plt.subplots(figsize=(12, fig_height))
    scatter = ax.scatter(x_values, y_positions, c=aid_values, cmap="viridis", s=70)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=label_size)
    ax.invert_yaxis()
    ax.set_title("AID Quality and Explanation Runtime Tradeoff")
    ax.set_xlabel("Mean explanation runtime (s)")
    ax.set_ylabel("Model / segmenter / masker / method and explanation budget")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Mean AID area; higher is better")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_coverage_table(path: Path, summary: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for key, group in _group_rows(
        summary,
        ("model_preset", "segmenter_strategy", "masker_strategy", "method_name", "explanation_budget"),
    ).items():
        model_preset, segmenter_strategy, masker_strategy, method_name, explanation_budget = key
        representative = group[0]
        completed = sum(1 for row in group if row.get("status") == "completed")
        failed = sum(1 for row in group if row.get("status") == "failed")
        total = len(group)
        rows.append(
            {
                "model_preset": model_preset,
                "segmenter_strategy": segmenter_strategy,
                "masker_strategy": masker_strategy,
                "method_name": method_name,
                "method_label": _method_label(representative),
                "explanation_budget": explanation_budget,
                "completed": completed,
                "failed": failed,
                "success_rate": completed / total if total else 0.0,
            }
        )
    rows.sort(
        key=lambda row: (
            row["model_preset"],
            row["segmenter_strategy"],
            row["masker_strategy"],
            row["method_name"],
            _to_float(row["explanation_budget"]),
        )
    )
    display = [
        [
            row["model_preset"],
            row["segmenter_strategy"],
            row["masker_strategy"],
            row["method_label"],
            row["explanation_budget"],
            row["completed"],
            row["failed"],
            f"{row['success_rate']:.1%}",
        ]
        for row in rows
    ]

    fig_height = max(3.5, 0.35 * len(display) + 1.3)
    fig, ax = plt.subplots(figsize=(17, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display,
        colLabels=[
            "Model",
            "Segmenter",
            "Masker",
            "Method",
            "Explanation Budget",
            "Completed",
            "Failed",
            "Success Rate",
        ],
        loc="center",
        cellLoc="center",
        colWidths=[0.14, 0.13, 0.15, 0.21, 0.12, 0.08, 0.07, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.25)
    ax.set_title("AID Quality Coverage Summary", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _attach_curve_metadata(curves: list[dict[str, str]], summary: list[dict[str, str]]) -> None:
    metadata_by_run = {
        row.get("run_id", ""): {
            "explanation_budget": row.get("explanation_budget", "n/a"),
            "method_label": _method_label(row),
        }
        for row in summary
    }
    for row in curves:
        metadata = metadata_by_run.get(row.get("run_id", ""), {})
        row["explanation_budget"] = metadata.get("explanation_budget", "n/a")
        row["method_label"] = metadata.get("method_label", row.get("method_name", "unknown"))


def _remove_stale_sample_plots(plots_dir: Path) -> None:
    for path in plots_dir.glob("quality_aid_sample_curves_*.png"):
        path.unlink()


def _mean_curve_rows(curves: list[dict[str, str]]) -> list[dict[str, Any]]:
    mean_rows = []
    fields = (
        "model_preset",
        "strategy_name",
        "method_name",
        "method_label",
        "explanation_budget",
        "curve_name",
        "removed_fraction",
    )
    for key, group in _group_rows(curves, fields).items():
        values = _numeric_values(group, "normalized_output")
        if not values:
            continue
        model_preset, strategy_name, method_name, method_label, explanation_budget, curve_name, removed_fraction = key
        mean_rows.append(
            {
                "model_preset": model_preset,
                "strategy_name": strategy_name,
                "method_name": method_name,
                "method_label": method_label,
                "explanation_budget": explanation_budget,
                "curve_name": curve_name,
                "removed_fraction": _to_float(removed_fraction),
                "normalized_output": _mean(values),
            }
        )
    return mean_rows


def _curve_points(curves: list[dict[str, str]], curve_name: str) -> np.ndarray:
    points = [
        (_to_float(row.get("removed_fraction")), _to_float(row.get("normalized_output")))
        for row in curves
        if row.get("curve_name") == curve_name
    ]
    points = [point for point in points if np.isfinite(point[0]) and np.isfinite(point[1])]
    points.sort(key=lambda point: point[0])
    return np.asarray(points, dtype=float)


def _group_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in fields)].append(row)
    return dict(grouped)


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = [_to_float(row.get(field)) for row in rows]
    return [value for value in values if np.isfinite(value)]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _method_label(row: dict[str, Any]) -> str:
    order = _to_int(row.get("order"))
    mode = str(row.get("mode") or "")
    approximation_type = str(row.get("approximation_type") or "")

    if approximation_type == "proxyshap":
        if mode.startswith("banzhaf/"):
            return f"ProxySHAP Banzhaf p={mode.split('/', 1)[1]} O{order}"
        return f"ProxySHAP {mode or 'unknown'} O{order}"

    if mode == "shapley" or "shapley" in str(row.get("method_name") or ""):
        return f"KernelSHAP Shapley O{order}"

    return str(row.get("method_name") or "unknown")


def _plot_filename(prefix: str, *parts: Any) -> str:
    full_text = "|".join(str(part) for part in parts)
    short_hash = hashlib.sha1(full_text.encode("utf-8")).hexdigest()[:10]
    readable = "_".join(_slug(str(part), max_length=24) for part in parts[:2])
    return f"{prefix}_{readable}_{short_hash}.png"


def _slug(value: str, max_length: int = 48) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    while "__" in text:
        text = text.replace("__", "_")
    return (text.strip("_") or "value")[:max_length]
