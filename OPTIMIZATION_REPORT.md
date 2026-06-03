# FIxLIP OOM Optimization Report

## Problem

Calling `FIxLIP.approximate_crossmodal(game, budget=2^16)` on **SigLIP base patch16-224** with 24 GiB RAM runs out of memory.

---

## Root Cause Analysis

### Player Counts (SigLIP)

| Parameter | Value |
|---|---|
| Model | google/siglip-base-patch16-224 |
| Image size / patch size | 224 / 16 |
| Grid | $14 \times 14$ |
| $n_{\text{players}}^{\text{image}}$ | **196** |
| $n_{\text{players}}^{\text{text}}$ | **11** |
| $n_{\text{players}}$ | **207** |
| Interactions $(\sum_{k=0}^{2} \binom{207}{k})$ | **21,529** |

### Memory Breakdown of the Original Code

The original `FIxLIP.aggregate()` + `solve_regression()` pipeline materialises three large float64 arrays:

| # | Object | Shape | Size (SigLIP, $2^{16}$) | Alive during step |
|---|---|---|---|---|
| 1 | regression_matrix ($X$) | $(N, P)$ | **10.51 GiB** | aggregate → solve |
| 2 | WX = diag($w$) @ $X$ | $(N, P)$ | **10.51 GiB** | solve only |
| 3 | $X^\top W X$ result | $(P, P)$ | **3.45 GiB** | factorised in-place by solve |
| | **Peak** | | **24.48 GiB** | all three simultaneously |

Adding GPU memory (~4 GiB for SigLIP): **~28.5 GiB total** — a 24 GiB machine inevitably OOMs.

#### Why the peak reaches 24.48 GiB

The call chain is:

```python
# In FIxLIP.aggregate():
regression_matrix = np.zeros((N, P))          # (1) X:    10.51 GiB

# In solve_regression():
WX = kernel_weights[:, None] * regression_matrix   # (2) diag(w)@X: 10.51 GiB
phi = np.linalg.solve(X.T @ WX, WX.T @ y)          # (3) X.T@WX temp + LU in-place
```

At the moment `np.linalg.solve` executes, all three arrays coexist:
- `X` (10.51 GiB) — still live as `regression_matrix` in the caller's scope
- `WX` (10.51 GiB) — live in `solve_regression`'s local scope
- `X.T @ WX` result (3.45 GiB) — created by matmul, then LU-factorised in-place

**The temporary `X.T @ WX` matrix built during `np.linalg.solve` adds the final 3.45 GiB, bringing the peak to 24.48 GiB.**

### Hidden Bug: Normal Equation Condition Number Explosion

Even with sufficient memory, $\kappa(X^\top W X) \approx 10^{34}$ due to interaction
collinearity. `np.linalg.solve` returns garbage coefficients (up to **±244k**
vs expected **±3–4**), or raises `LinAlgError: Singular matrix`.

---

## Solutions

### Chunked Ridge Regression (`ImputerFactory/regression.py`)

A flash-attention-style rewrite of the aggregation step. Instead of building
the full $N \times P$ regression matrix, coalitions are processed in chunks and
the normal equations are accumulated in $P \times P$ / $P \times 1$ space.

#### Algorithm

$$
\begin{aligned}
X^\top W X &= \sum_{i=1}^{N} w_i \cdot \mathbf{x}_i \mathbf{x}_i^\top
           = \sum_{\text{chunks}} X_c^\top W_c X_c \\
X^\top W y &= \sum_{i=1}^{N} w_i \cdot \mathbf{x}_i y_i
           = \sum_{\text{chunks}} X_c^\top W_c y_c
\end{aligned}
$$

Only the $P \times P$ accumulator is persistent; each chunk builds a small
$M \times P$ matrix ($M = \text{chunk\_size}$) that is freed immediately
after accumulation.

#### Optimisation A: Symmetric accumulator (BLAS dsyrk)

$X^\top W X$ is symmetric. Instead of computing the full $P \times P$ product
via generic `gemm` (which allocates a temporary of the same size as the
accumulator, doubling peak memory during each chunk's matmul), we use
**BLAS `dsyrk`** which computes only the upper triangle and accumulates
directly into the existing buffer:

```python
XtWX = blas.dsyrk(
    alpha=1.0, a=Xs, beta=1.0, c=XtWX,
    trans=1, lower=0, overwrite_c=1,
)
```

This eliminates the temporary $P \times P$ allocation — memory drops from
two simultaneous $P \times P$ matrices to one.

#### Optimisation B: Cholesky solve

A symmetric positive-definite system (ensured by Ridge regularisation) can
be solved via Cholesky decomposition instead of LU:

```python
c, low = cho_factor(XtWX, lower=False)
beta = cho_solve((c, low), XtWy)
```

- **Speed:** Cholesky is ~2× faster than LU for the same matrix
- **Stability:** Cholesky on a positive-definite matrix does not need
  pivoting, eliminating the `LinAlgError: Singular matrix` failure mode
  (the Ridge $\lambda$ guarantees definiteness)
- **Memory:** Factorisation is done in-place — no extra copy

#### Full chunk loop (optimised)

```python
for start in range(0, N, chunk_size):
    X_chunk = build_X(coalitions[start:end])     # (M, P)
    sw = np.sqrt(weights[start:end])
    Xs = sw[:, None] * X_chunk                   # pre-whiten
    ys = sw * values[start:end]

    # In-place symmetric rank-k update — no temp P×P matrix
    XtWX = blas.dsyrk(1.0, Xs, 1.0, c=XtWX, trans=1, lower=0, overwrite_c=1)
    XtWy += Xs.T @ ys                            # (P,) vector — negligible

    del X_chunk, Xs, sw, ys                      # free immediately
```

#### Memory

| Object | SigLIP $2^{16}$, max_order=2 | SigLIP $2^{16}$, max_order=1 |
|---|---|---|
| $X_{\text{chunk}}$ $(M \times P)$ | 1.64 GiB (peak) | 15.9 MiB |
| $X^\top W X$ accumulator $(P \times P)$ | **3.45 GiB** (persistent, upper-tri only) | **0.34 MiB** |
| **Peak** | **~5.1 GiB** | **~16 MiB** |

Comparison with the original code:

| Scenario | Original peak | Chunked + optimised | Reduction |
|---|---|---|---|
| CLIP $2^{19}$ (57 players, 1654 ints) | ~13 GiB | ~170 MiB | **78×** |
| SigLIP $2^{16}$ (207 players, 21529 ints) | **24.48 GiB ❌** | **~5.1 GiB ✅** | **4.8×** |

#### Usage

```python
from ImputerFactory.regression import crossmodal_approximation

iv = crossmodal_approximation(fixlip, game, budget=2**16, chunk_size=10000)
```

#### Accuracy

Verified against full-matrix `solve_regression` (the original code path):

```
max|β_chunked - β_reference| = 4 × 10^{-15}   (synthetic, P=12, N=500)
max|β_chunked - β_reference| = 2 × 10^{-11}   (CLIP-scale, P=57, N=2000)
```

The Ridge penalty $\lambda = 10^{-8}$ introduces negligible bias:
```
max|β_λ=1e-8 - β_λ=0| = 2 × 10^{-10}
```

---

### Forward-Crossmodal Edge Case Optimisation (`ImputerFactory/core/imputer.py`)

**The problem**: when `img_bs ≠ txt_bs`, the original code called
`_preprocess_batch()` which re-runs the full HuggingFace image processor
(resize, normalise, tensor conversion) on every image batch — extremely
slow for SigLIP (32 calls × ~500ms ≈ 16s).

**The fix**: reuse the already-preprocessed image tensor via `expand()`,
only re-tokenise text (cheap, ~5ms).

**Impact**: game evaluation time for SigLIP $2^{12}$ dropped from ~22s to ~3s.

---

### Visualisation Fix (`example_siglip.ipynb`)

**The problem**: with 207 players and `max_order=2`, the top-14 pair
interactions were all text-text pairs (e.g. `(202, 206)`). The connection
lines collapsed in the text region, making it look like a coordinate
mapping bug.

**The fix**: filter the interaction values to zero out image-image and
text-text pair scores before passing to the plot function. Cell 16 shows
the unfiltered (broken) version for comparison; cell 17 shows the fixed
crossmodal-only visualisation.

---

## File Changes

| File | Change | Purpose |
|---|---|---|
| `ImputerFactory/regression.py` | **New** | Chunked ridge regression with dsyrk + Cholesky |
| `ImputerFactory/__init__.py` | +1 line | Export `crossmodal_approximation`, `chunked_aggregate` |
| `ImputerFactory/core/imputer.py` | +31 lines | Edge-case optimisation (avoid redundant image processing) |
| `example_siglip.ipynb` | Modified | Use `crossmodal_approximation`, max_order=2, crossmodal visualisation |
| `OPTIMIZATION_REPORT.md` | **New** | This report |

**`src/` is untouched**.
