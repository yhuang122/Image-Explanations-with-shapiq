"""
CrossModalGaussianMasker — Composite masker with Gaussian-blur image occlusion.

Follows the Composite Pattern: internally instantiates two atomic maskers
(VisionBlurMasker + TextAttentionMasker) and delegates image/text
occlusion to each respectively.

.. deprecated::
    This masker will be replaced by ``CrossModalBlurMasker`` once
    ``VisionBlurMasker`` is fully implemented. The name ``"crossmodal_gaussian"``
    is retained for backward compatibility.

Registered as ``"crossmodal_gaussian"`` in the masker registry.
"""

from typing import Optional

from .base import BaseMasker
from .vision_mean import VisionMeanMasker  # fallback until VisionBlurMasker is ready
from .text_attention import TextAttentionMasker
from ImputerFactory.data import PhysicalMask, ProcessorOutput, MaskerConfig
from . import register_masker


@register_masker("crossmodal_gaussian")
class CrossModalGaussianMasker(BaseMasker):
    """
    Cross-modal occlusion orchestrator with Gaussian-blur image occlusion.

    Delegates:
        - Image occlusion → VisionMeanMasker (placeholder; will be replaced
          by VisionBlurMasker once implemented)
        - Text occlusion  → TextAttentionMasker (registered as ``"text_attn"``)

    The composite itself performs no element-wise operations.

    .. note::
        Skeleton only. The image occlusion side currently falls back to
        VisionMeanMasker (zero-out) behaviour. See Team B task B3.2.
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        super().__init__(config)
        # TODO (B3.2): replace VisionMeanMasker with VisionBlurMasker once
        #              Gaussian blur occlusion is implemented.
        self._vision_masker = VisionMeanMasker(config=config)
        self._text_masker = TextAttentionMasker(config=config)

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # 1. Delegate image occlusion — currently falls back to zero-out
        masked = self._vision_masker.apply(processor_output, physical_mask)

        # 2. Delegate text occlusion to TextAttentionMasker (pixel_values pass-through)
        masked = self._text_masker.apply(masked, physical_mask)

        return masked
