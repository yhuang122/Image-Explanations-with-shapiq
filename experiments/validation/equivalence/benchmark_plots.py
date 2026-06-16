"""Plot helpers for equivalence and coverage benchmarks."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark_schema import CSV_DIRNAME, PLOT_MODES, PLOTS_DIRNAME


def parse_report_float(report: dict, field: str):
    value = report.get(field)
    if value in (None, ""):
        return None
    return float(value)


def write_top_bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    tolerance=None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    upper_candidates = [abs(value) for value in values]
    if tolerance is not None:
        upper_candidates.append(float(tolerance))
    upper_limit = max(upper_candidates or [1.0]) * 1.2
    if upper_limit == 0:
        upper_limit = 1.0
    lower_limit = -max(upper_limit * 0.08, 1e-6)
    zero_marker = max(upper_limit * 0.025, 1e-8)
    plot_bottoms = [-zero_marker if value == 0 else 0 for value in values]
    plot_heights = [zero_marker if value == 0 else value for value in values]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5.0))
    bars = ax.bar(labels, plot_heights, bottom=plot_bottoms)
    for bar, value in zip(bars, values):
        if value == 0:
            bar.set_alpha(0.45)
    if tolerance is not None:
        ax.axhline(float(tolerance), color="red", linestyle="--", linewidth=1, label="tolerance")
        ax.legend()
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    for index, value in enumerate(values):
        label = "0.0" if value == 0 else f"{value:.3g}"
        ax.text(index, max(value, 0) + upper_limit * 0.025, label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_histogram(path: Path, values: list[float], xlabel: str, title: str, tolerance=None) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if tolerance is not None and values and min(values) >= 0 and max(values) <= float(tolerance):
        bins = np.linspace(0, float(tolerance), 21)
        ax.hist(values, bins=bins)
        ax.set_xlim(0, float(tolerance) * 1.2)
    else:
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values) or 1)))))
    if tolerance is not None:
        ax.axvline(float(tolerance), color="red", linestyle="--", linewidth=1, label="tolerance")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("run count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def group_mean(reports: list[dict], row_field: str, col_field: str, value_field: str) -> tuple[list[str], list[str], np.ndarray]:
    grouped = defaultdict(list)
    rows = []
    cols = []
    for report in reports:
        value = parse_report_float(report, value_field)
        if value is None:
            continue
        row = str(report.get(row_field) or "unknown")
        col = str(report.get(col_field) or "unknown")
        grouped[(row, col)].append(value)
        if row not in rows:
            rows.append(row)
        if col not in cols:
            cols.append(col)

    matrix = np.full((len(rows), len(cols)), np.nan)
    for row_index, row in enumerate(rows):
        for col_index, col in enumerate(cols):
            values = grouped.get((row, col))
            if values:
                matrix[row_index, col_index] = float(np.mean(values))
    return rows, cols, matrix


def write_heatmap(path: Path, rows: list[str], cols: list[str], matrix: np.ndarray, title: str, colorbar_label: str) -> None:
    if not rows or not cols:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.3), max(4, len(rows) * 0.7)))
    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked_matrix, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    for row_index in range(len(rows)):
        for col_index in range(len(cols)):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f"{value:.3g}", ha="center", va="center", color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def benchmark_max_diff(report: dict) -> float:
    for field in (
        "benchmark_metric_max_output_diff",
        "coalition_max_abs_output_diff",
        "empty_full_anchor_max_abs_output_diff",
    ):
        value = parse_report_float(report, field)
        if value is not None:
            return value
    return 0.0


def benchmark_mean_diff(report: dict) -> float:
    for field in (
        "benchmark_metric_mean_output_diff",
        "coalition_mean_abs_output_diff",
        "empty_full_anchor_mean_abs_output_diff",
    ):
        value = parse_report_float(report, field)
        if value is not None:
            return value
    return 0.0


def passed(report: dict) -> bool:
    return str(report.get("passed", "")).strip().lower() == "true"


def true_field(report: dict, field: str) -> bool:
    return str(report.get(field, "")).strip().lower() == "true"


def display_label(value: str, max_length: int = 34) -> str:
    value = str(value or "unknown")
    return value if len(value) <= max_length else f"{value[: max_length - 3]}..."


def ordered_values(reports: list[dict], field: str) -> list[str]:
    values = []
    for report in reports:
        value = str(report.get(field) or "unknown")
        if value not in values:
            values.append(value)
    return values


def has_multiple_values(reports: list[dict], field: str) -> bool:
    return len(ordered_values(reports, field)) > 1


def experiment_context(reports: list[dict]) -> str:
    if not reports:
        return ""
    report = reports[0]
    input_paths = [Path(str(row.get("input_path", ""))) for row in reports if row.get("input_path")]
    dataset = input_paths[0].parent.name if input_paths else "dataset"
    dataset = "MS COCO 100" if "mscoco" in dataset.lower() and "100" in dataset else dataset
    tolerance = parse_report_float(report, "tolerance")
    tolerance_text = f"{tolerance:.0e}" if tolerance is not None else "n/a"
    return (
        f"{dataset} | {len(reports)} runs | "
        f"coalitions={report.get('num_coalitions', 'n/a')} | "
        f"batch={report.get('batch_size', 'n/a')} | tolerance={tolerance_text}"
    )


def group_summary(reports: list[dict], group_field: str) -> list[dict]:
    groups = []
    for group in ordered_values(reports, group_field):
        group_reports = [report for report in reports if str(report.get(group_field) or "unknown") == group]
        original_runtime = [parse_report_float(report, "original_pipeline_runtime_s") for report in group_reports]
        migrated_runtime = [parse_report_float(report, "migrated_pipeline_runtime_s") for report in group_reports]
        original_runtime = [value for value in original_runtime if value is not None]
        migrated_runtime = [value for value in migrated_runtime if value is not None]
        deltas = [
            migrated - original
            for original, migrated in zip(original_runtime, migrated_runtime)
        ]
        ratios = [
            migrated / original
            for original, migrated in zip(original_runtime, migrated_runtime)
            if original
        ]
        max_diffs = [benchmark_max_diff(report) for report in group_reports]
        mean_diffs = [benchmark_mean_diff(report) for report in group_reports]
        groups.append(
            {
                "group": group,
                "runs": len(group_reports),
                "passed": sum(passed(report) for report in group_reports),
                "pass_rate": sum(passed(report) for report in group_reports) / len(group_reports),
                "max_diff": max(max_diffs or [0.0]),
                "mean_diff": float(np.mean(mean_diffs or [0.0])),
                "original_runtime": float(np.mean(original_runtime)) if original_runtime else 0.0,
                "migrated_runtime": float(np.mean(migrated_runtime)) if migrated_runtime else 0.0,
                "runtime_delta": float(np.mean(deltas)) if deltas else 0.0,
                "runtime_ratio": float(np.mean(ratios)) if ratios else 0.0,
            }
        )
    return groups


def write_table_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def write_table_image(
    path: Path,
    title: str,
    headers: list[str],
    rows: list[list],
    col_widths: list[float] | None = None,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(3.0, 0.45 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(14, figure_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row_index == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f5597")
        elif row_index % 2 == 0:
            cell.set_facecolor("#eef3f8")
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_pass_rate_plot(path: Path, groups: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [display_label(group["group"]) for group in groups]
    values = [group["pass_rate"] * 100.0 for group in groups]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4.6))
    ax.bar(labels, values, color="#2ca02c")
    ax.set_ylim(0, 112)
    ax.set_ylabel("pass rate (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for index, value in enumerate(values):
        ax.text(index, min(value + 2, 106), f"{value:.0f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_group_runtime_plot(path: Path, groups: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [display_label(group["group"]) for group in groups]
    original = [group["original_runtime"] for group in groups]
    migrated = [group["migrated_runtime"] for group in groups]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5.0))
    ax.bar(x - width / 2, original, width, label="Original HF baseline")
    ax.bar(x + width / 2, migrated, width, label="Migrated pipeline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean runtime per run (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_group_metric_plot(path: Path, groups: list[dict], metric: str, ylabel: str, title: str, tolerance=None) -> None:
    values = [float(group[metric]) for group in groups]
    write_top_bar_plot(
        path,
        [display_label(group["group"]) for group in groups],
        values,
        ylabel,
        title,
        tolerance=tolerance,
    )


def write_single_runtime_plot(path: Path, groups: list[dict], title: str) -> None:
    values = [float(group["migrated_runtime"]) for group in groups]
    write_top_bar_plot(
        path,
        [display_label(group["group"]) for group in groups],
        values,
        "Migrated pipeline mean runtime per run (s)",
        title,
    )


def write_runtime_scatter(path: Path, reports: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    points = [
        (
            parse_report_float(report, "original_pipeline_runtime_s"),
            parse_report_float(report, "migrated_pipeline_runtime_s"),
        )
        for report in reports
    ]
    points = [(original, migrated) for original, migrated in points if original is not None and migrated is not None]
    if not points:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    original = [point[0] for point in points]
    migrated = [point[1] for point in points]
    lower = min(original + migrated) * 0.95
    upper = max(original + migrated) * 1.05

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(original, migrated, alpha=0.75)
    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", linewidth=1, label="equal runtime")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Original HF baseline runtime per run (s)")
    ax.set_ylabel("Migrated pipeline runtime per run (s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_paired_sample_runtime_line(path: Path, reports: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for report in sorted(reports, key=lambda item: str(item.get("input_path", ""))):
        original = parse_report_float(report, "original_pipeline_runtime_s")
        migrated = parse_report_float(report, "migrated_pipeline_runtime_s")
        if original is not None and migrated is not None:
            rows.append((original, migrated))
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    original = [row[0] for row in rows]
    migrated = [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, original, marker="o", linewidth=1.6, label="Original HF baseline")
    ax.plot(x, migrated, marker="o", linewidth=1.6, label="Migrated pipeline")
    ax.set_xlabel("Image-text sample index")
    ax.set_ylabel("Runtime per run (s)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tolerance_value(reports: list[dict]):
    values = [parse_report_float(report, "tolerance") for report in reports]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def coverage_table(groups: list[dict], group_header: str) -> tuple[list[str], list[list]]:
    headers = [
        group_header,
        "Runs",
        "Passed",
        "Pass rate",
        "Max |output diff|",
        "Original mean (s)",
        "Migrated mean (s)",
        "Migrated / Original",
    ]
    rows = [
        [
            display_label(group["group"], 48),
            group["runs"],
            group["passed"],
            f"{group['pass_rate'] * 100:.1f}%",
            f"{group['max_diff']:.3g}",
            f"{group['original_runtime']:.2f}s",
            f"{group['migrated_runtime']:.2f}s",
            f"{group['runtime_ratio']:.2f}x",
        ]
        for group in groups
    ]
    return headers, rows


def write_coverage_outputs(
    csv_dir: Path,
    plots_dir: Path,
    prefix: str,
    title: str,
    groups: list[dict],
    group_header: str,
) -> dict:
    headers, rows = coverage_table(groups, group_header)
    csv_path = csv_dir / f"{prefix}_coverage_table.csv"
    image_path = plots_dir / f"{prefix}_coverage_table.png"
    write_table_csv(csv_path, headers, rows)
    write_table_image(
        image_path,
        title,
        headers,
        rows,
        col_widths=[0.22, 0.08, 0.08, 0.10, 0.14, 0.13, 0.13, 0.12],
    )
    return {
        f"{prefix}_coverage_csv": str(csv_path),
        f"{prefix}_coverage_table": str(image_path),
    }


def strategy_summary(reports: list[dict]) -> list[dict]:
    split_by_model = has_multiple_values(reports, "model_preset")
    keys = []
    for report in reports:
        model = str(report.get("model_preset") or "unknown")
        strategy = str(report.get("strategy_name") or "unknown")
        key = (model if split_by_model else "", strategy)
        if key not in keys:
            keys.append(key)

    groups = []
    for model, strategy in keys:
        strategy_reports = [
            report
            for report in reports
            if str(report.get("strategy_name") or "unknown") == strategy
            and (not split_by_model or str(report.get("model_preset") or "unknown") == model)
        ]
        comparable_reports = [
            report for report in strategy_reports if true_field(report, "coalition_comparison_available")
        ]
        migrated_runtime = [
            parse_report_float(report, "migrated_pipeline_runtime_s") for report in strategy_reports
        ]
        migrated_runtime = [value for value in migrated_runtime if value is not None]
        groups.append(
            {
                "group": f"{model} / {strategy}" if split_by_model else strategy,
                "model": model if split_by_model else "",
                "strategy": strategy,
                "runs": len(strategy_reports),
                "completed": len(strategy_reports),
                "strict_equivalent": sum(true_field(report, "strict_equivalence") for report in strategy_reports),
                "baseline_comparable": len(comparable_reports),
                "baseline_deviation": (
                    max(benchmark_max_diff(report) for report in comparable_reports)
                    if comparable_reports
                    else None
                ),
                "migrated_runtime": float(np.mean(migrated_runtime)) if migrated_runtime else 0.0,
            }
        )
    return groups


def write_strategy_table(csv_dir: Path, plots_dir: Path, groups: list[dict], title: str) -> dict:
    split_by_model = any(group.get("model") for group in groups)
    metric_headers = [
        "Runs",
        "Completed",
        "Strict-equivalent runs",
        "Baseline-comparable runs",
        "Max baseline deviation",
        "Migrated runtime (s)",
    ]
    headers = ["Model", "Strategy", *metric_headers] if split_by_model else ["Strategy", *metric_headers]
    rows = []
    for group in groups:
        metrics = [
            group["runs"],
            group["completed"],
            group["strict_equivalent"],
            group["baseline_comparable"],
            "N/A" if group["baseline_deviation"] is None else f"{group['baseline_deviation']:.3g}",
            f"{group['migrated_runtime']:.2f}s",
        ]
        if split_by_model:
            rows.append([display_label(group["model"], 24), display_label(group["strategy"], 44), *metrics])
        else:
            rows.append([display_label(group["strategy"], 48), *metrics])
    csv_path = csv_dir / "equivalence_strategies_coverage_table.csv"
    image_path = plots_dir / "equivalence_strategies_coverage_table.png"
    write_table_csv(csv_path, headers, rows)
    col_widths = (
        [0.13, 0.21, 0.07, 0.09, 0.15, 0.16, 0.10, 0.09]
        if split_by_model
        else [0.22, 0.08, 0.10, 0.15, 0.17, 0.14, 0.14]
    )
    write_table_image(
        image_path,
        title,
        headers,
        rows,
        col_widths=col_widths,
    )
    return {
        "equivalence_strategies_coverage_csv": str(csv_path),
        "equivalence_strategies_coverage_table": str(image_path),
    }


def write_strategy_baseline_deviation_plot(path: Path, groups: list[dict], title: str) -> None:
    comparable_groups = [group for group in groups if group["baseline_deviation"] is not None]
    if not comparable_groups:
        return
    write_top_bar_plot(
        path,
        [display_label(group["group"]) for group in comparable_groups],
        [float(group["baseline_deviation"]) for group in comparable_groups],
        "Max |baseline output - strategy output|",
        title,
    )


def strict_plots(csv_dir: Path, plots_dir: Path, reports: list[dict]) -> dict:
    outputs = {}
    groups = group_summary(reports, "case")
    context = experiment_context(reports)
    outputs.update(
        write_coverage_outputs(
            csv_dir,
            plots_dir,
            "equivalence_strict",
            f"Strict equivalence summary by case\n{context}",
            groups,
            "Validation case",
        )
    )
    write_histogram(
        plots_dir / "equivalence_strict_max_output_diff_distribution.png",
        [benchmark_max_diff(report) for report in reports],
        "Max |original output - migrated output|",
        f"Strict equivalence: output-difference distribution\n{context}",
        tolerance=tolerance_value(reports),
    )
    write_group_runtime_plot(
        plots_dir / "equivalence_strict_runtime_by_case.png",
        groups,
        f"Strict equivalence: runtime comparison by case\n{context}",
    )
    outputs.update(
        {
            "equivalence_strict_max_output_diff_distribution": str(
                plots_dir / "equivalence_strict_max_output_diff_distribution.png"
            ),
            "equivalence_strict_runtime_by_case": str(plots_dir / "equivalence_strict_runtime_by_case.png"),
        }
    )
    return outputs


def models_plots(csv_dir: Path, plots_dir: Path, reports: list[dict]) -> dict:
    outputs = {}
    groups = group_summary(reports, "model_preset")
    context = experiment_context(reports)
    outputs.update(
        write_coverage_outputs(
            csv_dir,
            plots_dir,
            "equivalence_models",
            f"Model coverage summary\n{context}",
            groups,
            "Vision-language model",
        )
    )
    write_pass_rate_plot(
        plots_dir / "equivalence_models_pass_rate.png",
        groups,
        f"Pipeline equivalence pass rate by model\n{context}",
    )
    write_group_runtime_plot(
        plots_dir / "equivalence_models_runtime_by_model.png",
        groups,
        f"Runtime comparison by model\n{context}",
    )
    write_group_metric_plot(
        plots_dir / "equivalence_models_max_output_diff_by_model.png",
        groups,
        "max_diff",
        "Max |original output - migrated output|",
        f"Worst output difference by model\n{context}",
        tolerance=tolerance_value(reports),
    )
    outputs.update(
        {
            "equivalence_models_pass_rate": str(plots_dir / "equivalence_models_pass_rate.png"),
            "equivalence_models_runtime_by_model": str(plots_dir / "equivalence_models_runtime_by_model.png"),
            "equivalence_models_max_output_diff_by_model": str(
                plots_dir / "equivalence_models_max_output_diff_by_model.png"
            ),
        }
    )
    return outputs


def strategies_plots(csv_dir: Path, plots_dir: Path, reports: list[dict]) -> dict:
    outputs = {}
    groups = strategy_summary(reports)
    context = experiment_context(reports)
    split_by_model = has_multiple_values(reports, "model_preset")
    outputs.update(write_strategy_table(csv_dir, plots_dir, groups, f"Strategy equivalence coverage summary\n{context}"))
    write_single_runtime_plot(
        plots_dir / "equivalence_strategies_migrated_runtime_by_strategy.png",
        groups,
        f"Migrated pipeline runtime by strategy\n{context}",
    )
    write_strategy_baseline_deviation_plot(
        plots_dir / "equivalence_strategies_baseline_deviation_by_strategy.png",
        groups,
        f"Baseline output deviation for comparable strategies\n{context}",
    )
    heatmap_row_field = "model_preset" if split_by_model else "case"
    heatmap_title_axis = "model and strategy" if split_by_model else "case and strategy"
    heatmap_path = (
        plots_dir / "equivalence_strategies_runtime_model_strategy_heatmap.png"
        if split_by_model
        else plots_dir / "equivalence_strategies_runtime_case_heatmap.png"
    )
    rows, cols, matrix = group_mean(reports, heatmap_row_field, "strategy_name", "migrated_pipeline_runtime_s")
    write_heatmap(
        heatmap_path,
        rows,
        cols,
        matrix,
        f"Mean runtime by {heatmap_title_axis}\n{context}",
        "mean runtime (s)",
    )
    outputs.update(
        {
            "equivalence_strategies_migrated_runtime_by_strategy": str(
                plots_dir / "equivalence_strategies_migrated_runtime_by_strategy.png"
            ),
            "equivalence_strategies_baseline_deviation_by_strategy": str(
                plots_dir / "equivalence_strategies_baseline_deviation_by_strategy.png"
            ),
            "equivalence_strategies_runtime_heatmap": str(heatmap_path),
        }
    )
    return outputs


def crossmodal_plots(csv_dir: Path, plots_dir: Path, reports: list[dict]) -> dict:
    outputs = {}
    groups = group_summary(reports, "case")
    context = experiment_context(reports)
    outputs.update(
        write_coverage_outputs(
            csv_dir,
            plots_dir,
            "equivalence_crossmodal",
            f"Crossmodal equivalence summary\n{context}",
            groups,
            "Crossmodal validation case",
        )
    )
    write_histogram(
        plots_dir / "equivalence_crossmodal_max_output_diff_distribution.png",
        [benchmark_max_diff(report) for report in reports],
        "Max |original output - migrated output|",
        f"Crossmodal equivalence: output-difference distribution\n{context}",
        tolerance=tolerance_value(reports),
    )
    write_runtime_scatter(
        plots_dir / "equivalence_crossmodal_original_vs_migrated_runtime.png",
        reports,
        f"Crossmodal runtime scatter: original baseline vs migrated pipeline\n{context}",
    )
    write_paired_sample_runtime_line(
        plots_dir / "equivalence_crossmodal_runtime_by_sample.png",
        reports,
        f"Crossmodal runtime by image-text sample\n{context}",
    )
    outputs.update(
        {
            "equivalence_crossmodal_max_output_diff_distribution": str(
                plots_dir / "equivalence_crossmodal_max_output_diff_distribution.png"
            ),
            "equivalence_crossmodal_original_vs_migrated_runtime": str(
                plots_dir / "equivalence_crossmodal_original_vs_migrated_runtime.png"
            ),
            "equivalence_crossmodal_runtime_by_sample": str(
                plots_dir / "equivalence_crossmodal_runtime_by_sample.png"
            ),
        }
    )
    return outputs


def write_benchmark_plots(
    output_dir: Path,
    reports: list[dict],
    mode: str,
    plots_dir: Path | None = None,
    csv_dir: Path | None = None,
) -> dict:
    if mode not in PLOT_MODES:
        raise ValueError(f"mode must be one of {', '.join(PLOT_MODES)}")
    plots_dir = plots_dir or output_dir / PLOTS_DIRNAME
    csv_dir = csv_dir or output_dir / CSV_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    if mode == "strict":
        outputs = strict_plots(csv_dir, plots_dir, reports)
    elif mode == "models":
        outputs = models_plots(csv_dir, plots_dir, reports)
    elif mode == "strategies":
        outputs = strategies_plots(csv_dir, plots_dir, reports)
    else:
        outputs = crossmodal_plots(csv_dir, plots_dir, reports)
    outputs["plot_mode"] = mode
    return outputs


