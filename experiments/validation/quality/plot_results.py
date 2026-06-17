"""Regenerate AID quality benchmark plots from saved CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aid_outputs import CSV_DIRNAME, CURVES_FILENAME, SUMMARY_FILENAME, output_paths
from aid_schema import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AID quality benchmark CSV files.")
    parser.add_argument(
        "--input",
        required=True,
        help="AID result suite folder or its csv folder.",
    )
    parser.add_argument(
        "--plot",
        help="Plot directory. Defaults to <suite-folder>/plots.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def suite_root(input_path: Path) -> Path:
    return input_path.parent if input_path.name == CSV_DIRNAME else input_path


def csv_paths(input_path: Path) -> tuple[Path, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"AID result input not found: {input_path}")
    root = suite_root(input_path)
    if input_path.name == CSV_DIRNAME:
        summary_path = input_path / SUMMARY_FILENAME
        curves_path = input_path / CURVES_FILENAME
    else:
        paths = output_paths(root)
        summary_path = paths["summary_csv"]
        curves_path = paths["curves_csv"]
    if not summary_path.exists():
        raise FileNotFoundError(f"AID summary CSV not found: {summary_path}")
    if not curves_path.exists():
        raise FileNotFoundError(f"AID curves CSV not found: {curves_path}")
    return summary_path, curves_path


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    root = suite_root(input_path)
    summary_path, curves_path = csv_paths(input_path)
    plots_dir = resolve_path(args.plot) if args.plot else output_paths(root)["plots_dir"]

    from aid_plots import write_aid_plots

    plot_paths = write_aid_plots(summary_path, curves_path, plots_dir)
    print(
        json.dumps(
            {
                "summary_csv": str(summary_path),
                "curves_csv": str(curves_path),
                "plots_dir": str(plots_dir),
                "plot_paths": plot_paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
