"""
Chunked (flash-attention-style) regression aggregation for FIxLIP.

Avoids OOM by never materialising the full N×P regression matrix.
Instead, processes coalitions in chunks, accumulates the normal
equations X^T W X and X^T W y in P×P space, then solves once.

Uses L2 (ridge) regularisation with a tiny lambda=1e-6 to ensure the
normal-equation system is numerically well-posed.  Without this
the design matrix interactions are highly collinear (cond ≈ 1e34)
and the unregularised solve returns garbage (coefficients ~ 1e5).

Usage:
    from src.regression import crossmodal_approximation
    iv = crossmodal_approximation(fixlip, game, budget=2**19, chunk_size=10000)

.. note::
    Temporary location under ``src/``. May be deleted in the future.
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from scipy.linalg import cho_factor, cho_solve, blas as blas
import shapiq
from shapiq.utils.sets import generate_interaction_lookup


# ─── Chunked aggregation (the core routine) ──────────────────────────────


def chunked_aggregate(
    coalition_matrix: np.ndarray,
    regression_weights: np.ndarray,
    coalition_values: np.ndarray,
    n_players: int,
    max_order: int = 2,
    mode: str = "banzhaf",
    chunk_size: int = 10000,
    interaction_lookup: Optional[Dict] = None,
    baseline_value: float = 0.0,
    ridge_lambda: float = 1e-8,
) -> shapiq.InteractionValues:
    """Aggregate coalition values using chunked + ridge-regularised normal equations.

    Flash-attention-style tiling: processes coalitions in chunks,
    accumulates X^T W X (P×P) and X^T W y (P×1) in-register, then
    solves the linear system once.  Peak memory = O(chunk_size × P)
    instead of O(budget × P).

    Args:
        coalition_matrix: (N, n_players) bool array.
        regression_weights: (N,) weight per coalition.
        coalition_values: (N,) observed values (already centred).
        n_players: total number of players.
        max_order: maximum interaction order (default 2).
        mode: "banzhaf" (FWBII) or "shapley" (FSII).
        chunk_size: rows per chunk (default 10 000).
        interaction_lookup: optional pre-built lookup.
        baseline_value: fallback for the empty-coalition value.
        ridge_lambda: L2 penalty.  The design matrix is severely
            rank-deficient (cond ≈ 1e34); a tiny ridge keeps the
            normal equations well-posed.  Default 1e-8 matches
            full-matrix ``lstsq`` to <0.002.

    Returns:
        shapiq.InteractionValues with estimated Moebius coefficients.
    """
    n_coalitions = coalition_matrix.shape[0]

    if interaction_lookup is None:
        interaction_lookup = generate_interaction_lookup(
            set(range(n_players)), min_order=0, max_order=max_order,
        )
    n_interactions = len(interaction_lookup)

    # ── Accumulators for normal equations (P×P and P×1) ────────────
    # XtWX stores only the upper triangle (filled by BLAS dsyrk);
    # it is treated as a full matrix for the Ridge diagonal add and
    # cho_factor / cho_solve below.
    XtWX = np.zeros((n_interactions, n_interactions), dtype=np.float64)
    XtWy = np.zeros(n_interactions, dtype=np.float64)

    # ── Process in chunks ──────────────────────────────────────────
    for start in range(0, n_coalitions, chunk_size):
        end = min(start + chunk_size, n_coalitions)
        actual = end - start

        # Build small regression matrix for this chunk (float64).
        X_chunk = np.zeros((actual, n_interactions), dtype=np.float64)
        for i, interaction in enumerate(interaction_lookup.keys()):
            if interaction == ():
                X_chunk[:, i] = 1.0
            else:
                X_chunk[:, i] = (
                    coalition_matrix[start:end, interaction]
                    .prod(axis=1)
                    .astype(np.float64)
                )

        w_chunk = regression_weights[start:end]
        y_chunk = coalition_values[start:end]

        # Pre-whiten: Xs = sqrt(w) * X,  ys = sqrt(w) * y
        sw = np.sqrt(w_chunk)
        Xs = sw[:, np.newaxis] * X_chunk          # (actual, P)
        ys = sw * y_chunk                         # (actual,)

        # Symmetric rank-k update: XtWX += Xs.T @ Xs
        # BLAS dsyrk writes into the upper triangle only, reusing
        # the existing XtWX buffer — no temporary P×P matrix.
        XtWX = blas.dsyrk(
            alpha=1.0, a=Xs, beta=1.0, c=XtWX,
            trans=1, lower=0, overwrite_c=1,
        )

        # Vector update: XtWy += Xs.T @ ys
        XtWy += Xs.T @ ys

        del X_chunk, Xs, w_chunk, y_chunk, sw, ys

    # ── Ridge regularisation ───────────────────────────────────────
    # The design matrix X is highly rank-deficient because interaction
    # columns are near-collinear.  The normal equations cond(X^T W X)
    # is ~1e34 — a direct solve returns garbage.  A tiny ridge fixes
    # the condition number without distorting the well-determined part.
    if ridge_lambda > 0.0:
        idx = np.arange(n_interactions)
        XtWX[idx, idx] += ridge_lambda

    # ── Solve (Cholesky on the symmetric positive-definite system) ─
    # XtWX is symmetric (only upper triangle is valid from dsyrk).
    # cho_factor(lower=False) reads the upper triangle and factors
    # in-place.  cho_solve then uses the factor — no extra copies.
    try:
        c, low = cho_factor(XtWX, lower=False)
        beta = cho_solve((c, low), XtWy)
    except np.linalg.LinAlgError:
        # Fallback: lstsq on the full matrix
        XtWX_sym = XtWX + XtWX.T - np.diag(XtWX.diagonal())
        beta = np.linalg.lstsq(XtWX_sym, XtWy, rcond=None)[0]

    # ── Build InteractionValues ────────────────────────────────────
    final_index = "FWBII" if mode.lower() == "banzhaf" else "FSII"
    baseline = baseline_value if baseline_value != 0.0 else beta[interaction_lookup[()]]

    iv = shapiq.InteractionValues(
        values=beta,
        interaction_lookup=interaction_lookup,
        baseline_value=baseline,
        n_players=n_players,
        index="Moebius",
        max_order=max_order,
        min_order=0,
        estimated=2 ** n_players > n_coalitions,
        estimation_budget=n_coalitions,
    )
    iv.index = final_index
    return iv


# ─── Weight helpers (inlined from src.fixlip) ────────────────────────────


def _regression_weights(sampler, kernel_weights: np.ndarray) -> np.ndarray:
    """Compute regression weights from sampler state.

    Coalitions enumerated exhaustively get weight = 1.
    Sampled coalitions get weight = empirical_count / n_total_samples.
    """
    rw = sampler.empirical_occurrences.copy()
    is_sampled = sampler.is_coalition_sampled
    rw_not_sampled = kernel_weights[np.sum(sampler.coalitions_matrix, axis=1)]
    rw[~is_sampled] = rw[~is_sampled] * rw_not_sampled[~is_sampled]
    return rw


# ─── Public entry point ──────────────────────────────────────────────────


def crossmodal_approximation(
    fixlip,
    game,
    budget: Optional[int] = None,
    budget_image: Optional[int] = None,
    budget_text: Optional[int] = None,
    interaction_lookup: Optional[Dict] = None,
    time_game: bool = False,
    chunk_size: int = 10000,
    ridge_lambda: float = 1e-8,
) -> shapiq.InteractionValues:
    """Crossmodal Banzhaf/Shapley approximation with chunked aggregation.

    Parallels ``FIxLIP.approximate_crossmodal()`` but uses chunked
    ridge-regression for the aggregation step, avoiding an
    O(N_interactions × budget) dense matrix.

    The caller's ``fixlip.sampler_image`` and ``fixlip.sampler_text`` are
    mutated (re-sampled) — same as the original ``approximate_crossmodal()``.

    Args:
        fixlip: A ``src.fixlip.FIxLIP`` instance configured with
            ``n_players_image`` and ``n_players_text``.
        game: A ``VisionLanguageGame`` (or any ``shapiq.Game`` with a
            ``value_function_crossmodal`` method and ``normalization_value``).
        budget: Total coalition evaluations.  Split between modalities
            via ``fixlip.split_budget()``.
        budget_image, budget_text: Override automatic budget split.
        interaction_lookup: Optional pre-built interaction lookup dict.
        time_game: If True, time the game evaluation and print.
        chunk_size: Rows per chunk in the regression matrix
            (default 10 000;  ~150 MiB peak for 1654 interactions).
        ridge_lambda: L2 penalty.  Pass 0 for the original unregularised
            behaviour (unstable for rank-deficient data at small budgets).

    Returns:
        ``shapiq.InteractionValues`` (same structure as the original method).
    """
    if not fixlip.is_crossmodal:
        raise ValueError(
            "Crossmodal approximation requires n_players_image "
            "and n_players_text."
        )

    # ── 1. Budget ──────────────────────────────────────────────────
    if budget is not None:
        if budget < 4:
            raise ValueError("`budget` should be at least 4.")
        budget_image, budget_text = fixlip.split_budget(budget)
    elif budget_image is None or budget_text is None:
        raise ValueError("Pass either `budget` or `budget_image`+`budget_text`.")

    # ── 2. Sample coalitions (mutates fixlip samplers) ─────────────
    fixlip.sampler_image.sample(budget_image)
    fixlip.sampler_text.sample(budget_text)

    # ── 3. Evaluate the game (the expensive forward pass) ──────────
    if time_game:
        import time
        _t0 = time.time()

    coalition_values_xm = game.value_function_crossmodal(
        coalitions_image=fixlip.sampler_image.coalitions_matrix,
        coalitions_text=fixlip.sampler_text.coalitions_matrix,
    )

    if time_game:
        print(f"Game evaluation: {time.time() - _t0:.1f}s")

    coalition_values_xm = coalition_values_xm - game.normalization_value

    # ── 4. Flatten → (N,) ─────────────────────────────────────────
    coalition_values = coalition_values_xm.reshape(-1)

    # ── 5. Build full coalition matrix (bool,  ~30 MiB for 2^19) ──
    img_mat = fixlip.sampler_image.coalitions_matrix
    txt_mat = fixlip.sampler_text.coalitions_matrix
    n_img, n_txt = img_mat.shape[0], txt_mat.shape[0]

    coalitions_matrix = np.concatenate(
        [
            np.repeat(img_mat, n_txt, axis=0),
            np.tile(txt_mat, (n_img, 1)),
        ],
        axis=1,
    )

    # ── 6. Regression weights ──────────────────────────────────────
    p = fixlip.p
    kw_img = np.array(
        [p ** k * (1 - p) ** (fixlip.n_players_image - k)
         for k in range(fixlip.n_players_image + 1)]
    )
    kw_txt = np.array(
        [p ** k * (1 - p) ** (fixlip.n_players_text - k)
         for k in range(fixlip.n_players_text + 1)]
    )
    rw_img = _regression_weights(fixlip.sampler_image, kw_img)
    rw_txt = _regression_weights(fixlip.sampler_text, kw_txt)
    regression_weights = np.outer(rw_img, rw_txt).reshape(-1)

    # ── 7. Chunked aggregation (the OOM-safe part) ─────────────────
    return chunked_aggregate(
        coalition_matrix=coalitions_matrix,
        regression_weights=regression_weights,
        coalition_values=coalition_values,
        n_players=fixlip.n_players_image + fixlip.n_players_text,
        max_order=fixlip.max_order,
        mode=fixlip.mode,
        chunk_size=chunk_size,
        interaction_lookup=interaction_lookup,
        ridge_lambda=ridge_lambda,
    )
