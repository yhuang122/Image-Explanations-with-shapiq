"""
Plot insertion/deletion curves (Figure 3 style) from saved .npy results.

Usage:
    python experiments/migrated/plot_insertion_deletion.py \
        --path /path/to/mscoco_aid_fixlip_0_5.npy \
        --out insertion_deletion.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Path to mscoco_aid_fixlip_*.npy")
parser.add_argument("--out", type=str, default="insertion_deletion.png")
parser.add_argument("--n_points", type=int, default=100, help="Resolution of x-axis interpolation")
args = parser.parse_args()

data = np.load(args.path, allow_pickle=True).item()
print("Available keys:", list(data.keys()))

# ── colour / style config (matches paper Figure 3) ───────────────────────────
METHOD_STYLE = {
    "banzhaf/0.7/order2": dict(label="FIxLIP p=0.7", color="#e41a1c"),
    "banzhaf/0.5/order2": dict(label="FIxLIP p=0.5", color="#ff7f00"),
    "banzhaf/0.3/order2": dict(label="FIxLIP p=0.3", color="#f0c040"),
    "shapley/order1":     dict(label="Shapley values", color="#377eb8"),
}

x_common = np.linspace(1, 0, args.n_points)  # 100% → 0%

fig, ax = plt.subplots(figsize=(7, 5))

for key, style in METHOD_STYLE.items():
    if key not in data:
        print(f"  Skipping {key} (not in file)")
        continue

    mif_curves, lif_curves = [], []

    for i, entry in data[key].items():
        raw_mif = entry["predictions_deletion_mif"]
        raw_lif = entry["predictions_deletion_lif"]

        empty = float(raw_mif[-1])
        full  = float(raw_mif[0])
        if abs(full - empty) < 1e-8:
            continue  # degenerate image

        mif_norm = (raw_mif - empty) / (full - empty)
        lif_norm = (raw_lif - empty) / (full - empty)

        n = len(mif_norm)
        x_orig = np.linspace(1, 0, n)  # 100% → 0%

        mif_curves.append(np.interp(x_common, x_orig[::-1], mif_norm[::-1]))
        lif_curves.append(np.interp(x_common, x_orig[::-1], lif_norm[::-1]))

    if not mif_curves:
        continue

    mif_mean = np.mean(mif_curves, axis=0)
    lif_mean = np.mean(lif_curves, axis=0)
    auc = np.mean(lif_mean - mif_mean)

    label = f"{style['label']} ({auc:.2f})"
    c = style["color"]

    # delete important first (dotted, goes down)
    ax.plot(x_common * 100, mif_mean, color=c, linestyle="dotted", linewidth=1.5)
    # insert important first = lif curve (keep most important = remove least important)
    ax.plot(x_common * 100, lif_mean, color=c, linestyle="dashed", linewidth=1.5, label=label)

ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
ax.set_xlabel(r"Percentage of input $k\,/\,(n_\mathrm{txt}+n_\mathrm{img})$")
ax.set_ylabel("Prediction change (normalized)")
ax.set_xlim(100, 0)
ax.legend(title="Method (Area between curves)", fontsize=8, title_fontsize=8)
ax.annotate("insert", xy=(15, 0.65), fontsize=9,
            arrowprops=dict(arrowstyle="->"), xytext=(30, 0.75))
ax.annotate("delete", xy=(70, -0.25), fontsize=9,
            arrowprops=dict(arrowstyle="->"), xytext=(55, -0.35))

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved to {args.out}")
