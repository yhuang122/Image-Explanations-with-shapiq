"""
VisionBlurMasker — Image occlusion via Gaussian blur in masked regions.

Operates exclusively on ``pixel_values``. Must never touch ``input_ids`` or
``attention_mask``.

Instead of zeroing out pixels (as VisionMeanMasker does), this masker applies
a Gaussian blur to the masked regions, blending the blurred image with the
original according to the binary mask. This produces a smoother transition
at occlusion boundaries and better preserves the input distribution for
CNN-based models.

Design (Phase 1 — CPU, current):
    For each batch element, convert ``pixel_values`` from GPU to CPU numpy,
    call ``skimage.filters.gaussian(image, sigma)`` per channel, blend, and
    convert back to torch on the original device.

Design (Phase 2 — GPU, future):
    Replace CPU convolution with a pre-computed ``torch.nn.functional.conv2d``
    kernel to avoid CPU↔GPU transfer. Semantics are identical.

Blending formula:
    ``output = original * mask + blurred * (1 - mask)``
    (mask=1 → keep original pixel; mask=0 → use blurred pixel).

References:
    Analogous to ``shapiq.imputer.gaussian_imputer.GaussianImputer`` for
    tabular data, which draws Monte Carlo samples from a conditional
    multivariate Gaussian instead of using a point estimate (mean).

Registered as ``"vision_blur"`` in the masker registry.
"""

from typing import Optional

import numpy as np
import torch

from .base import BaseMasker
from . import register_masker
from ImputerFactory.data import PhysicalMask, ProcessorOutput, MaskerConfig

try:
    from skimage.filters import gaussian as _gaussian_blur
except ImportError:
    _gaussian_blur = None


@register_masker("vision_blur")
class VisionBlurMasker(BaseMasker):
    """Image occlusion via Gaussian blur in masked regions.

    Uses ``skimage.filters.gaussian`` on CPU for the blur (Phase 1).
    Future: replace with GPU conv2d to eliminate CPU↔GPU transfer.

    Contracts:
        - Only ``pixel_values`` is modified; ``input_ids`` and
          ``attention_mask`` pass through unchanged.
        - Originals are never mutated — the blend creates new tensors.

    Args:
        config: Optional MaskerConfig. Sigma is read from
            ``config.vision_blur.sigma`` when provided, falling back
            to the constructor default.
        sigma: Default sigma (standard deviation) for the Gaussian
            kernel. Only used when ``config`` is ``None``.
    """

    def __init__(
        self,
        config: Optional[MaskerConfig] = None,
        sigma: float = 3.0,
    ):
        super().__init__(config)
        if _gaussian_blur is None:
            raise ImportError(
                "VisionBlurMasker requires scikit-image. "
                "Install with: pip install scikit-image"
            )

        # Resolve sigma: typed config overrides constructor default
        if config is not None:
            self._sigma = config.vision_blur.sigma
        else:
            self._sigma = sigma

    # ─── Public API ───────────────────────────────────────────────────────

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """Apply Gaussian blur to masked regions of ``pixel_values``.

        For each (batch, channel) slice:
            1. Blur the entire image with ``skimage.filters.gaussian``
               using the configured sigma.
            2. Blend: keep original pixels where mask=1, use blurred
               pixels where mask=0.

        The full-image blur per channel is simpler and faster than
        per-region blurring, and produces correct results because the
        mask selects which parts of the blurred image to use.

        Args:
            processor_output: Original model inputs. Only ``pixel_values``
                is consumed; ``input_ids`` and ``attention_mask`` pass
                through unchanged.
            physical_mask: Must contain ``image_binary_mask`` of shape
                ``(N, C, H, W)`` with dtype float (1=keep, 0=blur).

        Returns:
            ProcessorOutput with blurred ``pixel_values``.
        """
        if physical_mask.image_binary_mask is None:
            return processor_output

        pixel_values = processor_output.pixel_values          # (N, C, H, W)
        mask = physical_mask.image_binary_mask                # (N, C, H, W)
        device = pixel_values.device

        # ── Phase 1: CPU-based per-channel Gaussian blur ──────────────
        # skimage operates on numpy arrays, so we transfer to CPU once.
        im_np = pixel_values.detach().cpu().numpy()            # (N, C, H, W)
        mask_np = mask.detach().cpu().numpy()                  # (N, C, H, W)

        sigma = self._sigma
        n_batch, n_chan = im_np.shape[:2]

        blurred = np.empty_like(im_np)
        for b in range(n_batch):
            for c in range(n_chan):
                # skimage.filters.gaussian applies a 2D Gaussian to
                # (H, W) single-channel arrays
                blurred[b, c] = _gaussian_blur(
                    im_np[b, c], sigma=sigma
                )

        # Blend: original where mask=1, blurred where mask=0
        blended_np = im_np * mask_np + blurred * (1.0 - mask_np)

        # Convert back to original device and dtype
        pixel_values = torch.from_numpy(blended_np).to(
            device=device, dtype=processor_output.pixel_values.dtype
        )

        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=processor_output.input_ids,          # pass-through
            attention_mask=processor_output.attention_mask, # pass-through
            model_type=processor_output.model_type,
        )
