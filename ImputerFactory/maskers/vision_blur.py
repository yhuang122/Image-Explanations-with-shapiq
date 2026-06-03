"""
VisionBlurMasker — Image occlusion via Gaussian blur in masked regions.

Operates exclusively on ``pixel_values``. Must never touch ``input_ids`` or
``attention_mask``.

Instead of zeroing out pixels (as VisionMeanMasker does), this masker applies
a Gaussian blur to the masked regions, blending the blurred image with the
original according to the binary mask. This produces a smoother transition
at occlusion boundaries and better preserves the input distribution for
CNN-based models.

.. note::
    Skeleton only. Implementation is tracked as Team B task B3.2.

    The approach is analogous to the tabular ``GaussianImputer``
    (``shapiq.imputer.gaussian_imputer.GaussianImputer``), which draws Monte Carlo
    samples from a conditional multivariate Gaussian rather than using a point
    estimate (mean). The vision analogue replaces masked pixels with a local
    Gaussian-weighted average instead of the global dataset mean.

Registered as ``"vision_blur"`` in the masker registry.

Design (CPU Planning, GPU optional):
    Phase 1 (CPU only):
        For each batch, convert ``pixel_values`` from GPU to CPU numpy,
        call ``skimage.filters.gaussian(image, sigma)``, then convert the
        result back to a GPU tensor and blend with the mask.

    Phase 2 (GPU optimisation, future):
        Replace CPU conv with a pre-computed ``torch.nn.functional.conv2d``
        kernel to avoid CPU↔GPU transfer. This is a pure optimisation and
        does not change the occlusion semantics.

    Blending formula:
        ``blurred_pixels * (1 - mask) + original_pixels * mask``
        (where mask=1 means "keep", mask=0 means "occlude").
"""

from typing import Optional

import torch

from .base import BaseMasker
from . import register_masker
from ImputerFactory.data import PhysicalMask, ProcessorOutput, MaskerConfig


@register_masker("vision_blur")
class VisionBlurMasker(BaseMasker):
    """
    Image occlusion via Gaussian blur in masked regions.

    Uses ``skimage.filters.gaussian`` on CPU for the blur (Phase 1).
    Future optimisation: replace with GPU conv2d to eliminate CPU↔GPU transfer.

    This produces a smoother, more natural occlusion boundary compared to the
    hard zero-out of VisionMeanMasker, and helps preserve the input distribution
    for CNN-based models (e.g. CLIP-ResNet with SLIC segmenter).

    Registered as ``"vision_blur"``.

    Args:
        config: Optional MaskerConfig (per-strategy params via ``config.blur``).
        sigma: Standard deviation of the Gaussian kernel. Default 3.0.
               Larger values produce softer blur, smaller values produce
               a sharper transition.
    """

    def __init__(
        self,
        config: Optional[MaskerConfig] = None,
        sigma: float = 3.0,
    ):
        super().__init__(config)
        self._sigma = sigma

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """
        Apply Gaussian blur occlusion to ``pixel_values``.

        Args:
            processor_output: Original model inputs (``pixel_values``, ``input_ids``,
                ``attention_mask``). Only ``pixel_values`` is modified.
            physical_mask: Contains ``image_binary_mask`` of shape ``(N, C, H, W)``
                with dtype float, where 1 = keep, 0 = occlude (blur).

        Returns:
            ProcessorOutput with blurred ``pixel_values``; ``input_ids`` and
            ``attention_mask`` pass through unchanged.
        """
        pixel_values = processor_output.pixel_values  # (N, C, H, W)

        if physical_mask.image_binary_mask is None:
            return processor_output

        mask = physical_mask.image_binary_mask.to(pixel_values.device)

        # TODO (B3.2): implement blur + blend logic
        #
        #   import numpy as np
        #   from skimage.filters import gaussian as gaussian_blur
        #
        #   # Convert to CPU numpy (skimage operates on numpy arrays)
        #   im_np = pixel_values.cpu().numpy()              # (N, C, H, W)
        #   mask_np = mask.cpu().numpy()                     # (N, C, H, W)
        #
        #   blurred = np.empty_like(im_np)
        #   for batch_idx in range(im_np.shape[0]):
        #       # skimage.filters.gaussian expects (H, W, C) or (H, W) per channel.
        #       # Apply per-channel: iterate over channels.
        #       for c in range(im_np.shape[1]):
        #           blurred[batch_idx, c] = gaussian_blur(
        #               im_np[batch_idx, c], sigma=self._sigma
        #           )
        #
        #   # Blend: keep original where mask=1, blurred where mask=0
        #   blended_np = im_np * mask_np + blurred * (1.0 - mask_np)
        #
        #   # Convert back to torch on original device
        #   pixel_values = torch.from_numpy(blended_np).to(
        #       device=pixel_values.device, dtype=pixel_values.dtype
        #   )

        # Skeleton: pass through unchanged until B3.2 is implemented
        _ = mask  # placeholder
        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=processor_output.input_ids,
            attention_mask=processor_output.attention_mask,
            model_type=processor_output.model_type,
        )
