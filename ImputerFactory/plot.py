"""
Visualization utilities for the ImputerFactory pipeline.

Follows the same single-axes layout as ``src.plot.plot_image_and_text_together``
while adding support for irregular superpixel layouts (SLIC, GradientGuided, …)
via the segmenter's ``_label_map``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── Reuse proven src.plot primitives ──────────────────────────────────────
from src.plot import (
    interactions_to_color,
    interactions_to_heatmap,
    draw_fancy_hyper_edges,
    _get_word_dimensions,
    _get_line_lengths,
    value_to_color,
)
from src.utils import get_subset, sort_interactions, convert_iv_to_first_order


def plot_image_and_text_together(
    img: np.ndarray,
    text: list[str],
    image_players: list[int],
    iv,  # shapiq.InteractionValues
    *,
    segmenter=None,
    plot_heatmap: bool = True,
    plot_interactions: bool = True,
    top_k: int = 14,
    figsize: tuple[float, float] = (7, 8),
    fontsize: int = 22,
    image_span: float = 0.9,
    margin: float = 0.0,
    line_padding: float = 0.5,
    margin_text: tuple[float, float] = (0.0, 0.0),
    color_text: bool = True,
    max_value: float | None = None,
    sort_by_abs: bool = True,
    debug: bool = False,
    show: bool = True,
):
    """Image + text explanation visualisation.

    Follows the same layout algorithm as
    ``src.plot.plot_image_and_text_together``: a single
    :class:`~matplotlib.axes.Axes` with the image anchored at the top
    and coloured token boxes below it.

    When a *segmenter* with a ``_label_map`` attribute is provided,
    the heatmap and interaction-curve anchors are derived from the
    actual superpixel regions instead of a synthetic square grid.

    Args:
        img: Denormalised image as ``(H, W, 3)`` numpy array.
        text: Token strings (may contain empty padding entries).
        image_players: Player indices that belong to the image
            (typically ``list(range(n_players_image))``).
        iv: :class:`shapiq.InteractionValues` object.
        segmenter: Optional segmenter.  If it exposes a ``_label_map``
            ``(H, W)`` integer tensor, centroids are computed from it;
            otherwise a regular sqrt-grid is assumed.
        plot_heatmap: Overlay the first-order attribution heatmap.
        plot_interactions: Draw second-order interaction curves between
            text tokens and image regions.
        top_k: How many top interaction pairs to draw.
        figsize: Matplotlib figure size ``(width, height)``.
        fontsize: Font size for text tokens.
        image_span: Fraction of the figure height taken by the image
            (default 0.9).
        margin: Gap between image bottom and first text line (in line
            heights).
        line_padding: Vertical spacing between text lines (in line
            heights).
        margin_text: ``(left, right)`` horizontal margins as fractions
            of figure width.
        color_text: Whether to colour text-token backgrounds by their
            first-order attribution.
        max_value: Max absolute value for colour scaling.  ``None``
            auto-scales per call.
        sort_by_abs: Sort interactions by absolute value.
        debug: Print layout diagnostics.
        show: Call ``plt.show()`` before returning.
    """

    n_players_image = len(image_players)
    H, W = img.shape[:2]

    # ── Detect irregular layout via segmenter._label_map ──────────────
    label_map = None
    if segmenter is not None:
        lm = getattr(segmenter, "_label_map", None)
        if lm is not None:
            label_map = lm.cpu().numpy()  # (H, W) int

    # ── First-order values ─────────────────────────────────────────────
    first_order = convert_iv_to_first_order(iv)
    values1 = first_order.get_n_order(1).values
    img_vals = values1[:n_players_image]
    txt_vals = values1[n_players_image:]

    max_abs_img = float(np.quantile(np.abs(img_vals), 0.99) or 1.0)

    # ── Build heatmap ──────────────────────────────────────────────────
    if label_map is None:
        iv_image = get_subset(iv, players=image_players)
        heatmap_img = interactions_to_heatmap(iv=iv_image, img=img, colors=None)
    else:
        heatmap_arr = np.zeros((H, W, 4), dtype=np.float32)
        for k in range(n_players_image):
            v = float(img_vals[k])
            c_rgb = value_to_color(v)
            alpha = min(abs(v) / max_abs_img, 1.0)
            heatmap_arr[label_map == k] = (*c_rgb, alpha)
        heatmap_img = Image.fromarray((heatmap_arr * 255).astype(np.uint8))

    # ── Figure & axes (single axes like src.plot) ──────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Image placement ────────────────────────────────────────────────
    image_aspect_ratio = 1  # square image
    fig_aspect_ratio = figsize[0] / figsize[1]
    image_height = image_span
    image_width = image_height * image_aspect_ratio / fig_aspect_ratio
    x_center_image = 0.5
    left = x_center_image - image_width / 2
    right = x_center_image + image_width / 2
    top = 1.0
    bottom = 1.0 - image_span
    extent = (left, right, bottom, top)

    image_patch = ax.imshow(img, extent=extent, aspect="auto", zorder=0)

    if plot_heatmap:
        ax.imshow(heatmap_img, extent=extent, aspect="auto")

    # ── Text layout (pixel-space helpers from src.plot) ─────────────────
    word_dimensions = _get_word_dimensions(words=text, figsize=figsize, font_size=fontsize)
    space_width = _get_word_dimensions(words=[" "], figsize=figsize, font_size=fontsize)[0][0]
    line_height = _get_word_dimensions(words=["A"], figsize=figsize, font_size=fontsize)[0][1]

    conversion_factor = figsize[0] * 100
    x0 = 0
    x1 = conversion_factor
    word_spacing = space_width * 1.5
    line_start = x0 + space_width * 2 + margin_text[0] * (x1 - x0)
    line_end = x1 - space_width * 2 - margin_text[1] * (x1 - x0)

    line_lengths = _get_line_lengths(
        word_dimensions=word_dimensions,
        word_spacing=word_spacing,
        line_start=line_start,
        line_end=line_end,
    )
    left_margins = [
        float((line_end - line_start - ll) / 2) for ll in line_lengths
    ]

    # Y position for first line (just below the image)
    y0_data = bottom  # top of text area = bottom of image extents
    y_pos = y0_data - line_height * (1.0 + margin) / conversion_factor

    # ── Draw text tokens ───────────────────────────────────────────────
    positions: dict[int, np.ndarray] = {}

    # -- image player positions (for interaction curves) --
    if label_map is not None:
        # Fix: Matplotlib Y axis points upward. The image top is at `top`,
        # bottom at `bottom`. Centroid mapping:
        #   yd = bottom + (1.0 - cy / H) * image_height
        for k in range(n_players_image):
            ys, xs = np.where(label_map == k)
            if len(ys):
                cx, cy = xs.mean(), ys.mean()
            else:
                cx, cy = 0.0, 0.0
            xd = left + (cx / W) * image_width
            yd = bottom + (1.0 - cy / H) * image_height
            positions[k] = np.array([xd, yd])
    else:
        grid_size = int(np.sqrt(n_players_image))
        for k in range(n_players_image):
            col = k % grid_size
            row = k // grid_size
            xd = left + image_width / grid_size * (col + 0.5)
            yd = top - image_height / grid_size * (row + 0.5)
            positions[k] = np.array([xd, yd])

    if debug:
        print(f"image_span={image_span} grid_size={grid_size if label_map is None else 'irregular'}")
        for k in range(min(4, n_players_image)):
            print(f"  image player {k}: pos={positions[k]}")

    # -- draw text tokens line-by-line --
    line_counter = 0
    x_pos = line_start + left_margins[line_counter]
    for i, (word, (wd, _ht)) in enumerate(zip(text, word_dimensions)):
        player = n_players_image + i
        v = txt_vals[i] if i < len(txt_vals) else 0.0
        alpha = min(abs(v) / (max_abs_img or 1.0), 1.0)

        # Fix: skip empty padding tokens — don't render or register
        # their positions, avoiding ghost interaction curves.
        if not word.strip():
            x_pos += wd + word_spacing
            continue

        if x_pos + wd > line_end:
            line_counter += 1
            try:
                x_pos = line_start + left_margins[line_counter]
            except IndexError:
                x_pos = line_start + left_margins[-1]
            # Move down one line
            y_pos -= (line_height + line_height * line_padding) / conversion_factor

        # colour from first-order value
        if color_text:
            c_rgb = value_to_color(v)
            facecolor = c_rgb
        else:
            facecolor = (1.0, 1.0, 1.0)
            alpha = 1.0

        text_color = "black" if alpha < 0.8 else "white"

        bbox = {
            "facecolor": facecolor,
            "alpha": alpha,
            "edgecolor": facecolor,
            "boxstyle": "round,pad=0.1",
            "linestyle": "solid",
            "linewidth": 1.5,
        }
        if not color_text:
            bbox["edgecolor"] = "lightgrey"
            bbox["alpha"] = 1.0

        ax.text(
            x_pos / conversion_factor,
            y_pos,
            word,
            fontsize=fontsize,
            bbox=bbox,
            zorder=100,
            color=text_color,
        )

        # Fix: in data coords, ax.text anchors at the bbox bottom-left.
        # Route interaction curves to the top-centre of the text box:
        x_center = (x_pos + wd / 2) / conversion_factor
        y_top_edge = y_pos + (line_height * 1.1) / conversion_factor  # slight upward bias to the top edge
        positions[player] = np.array([x_center, y_top_edge])

        x_pos += wd + word_spacing

    if debug:
        for i in range(min(4, len(text))):
            p = n_players_image + i
            if p in positions:
                print(f"  text player {p}: pos={positions[p]}")

    # ── Interaction curves ──────────────────────────────────────────────
    if iv.max_order >= 2 and plot_interactions:
        iv_second_order = iv.get_n_order(order=2, min_order=2, max_order=2)
        top_interactions = sort_interactions(
            iv_second_order, reverse=True, sort_by_abs=sort_by_abs
        )[:top_k]

        if debug:
            print(f"Top {top_k} interactions: {top_interactions}")

        colors = interactions_to_color(iv, max_value=max_value)

        hyper_edges_to_draw = []
        for interaction in top_interactions:
            interaction = interaction[0]
            
            # Fix: second safety check — drop interaction pairs whose
            # players were filtered out (not in `positions`).
            if not all(p in positions for p in interaction):
                continue
                
            hyper_edges_to_draw.append(interaction)

        draw_fancy_hyper_edges(
            axis=ax,
            pos=positions,
            colors=colors,
            hyper_edges=hyper_edges_to_draw,
            debug=debug,
        )

    plt.tight_layout(pad=0.05)
    if show:
        plt.show()
