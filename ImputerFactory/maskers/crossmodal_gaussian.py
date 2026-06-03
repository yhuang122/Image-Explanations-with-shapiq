"""
CrossModalGaussianMasker — Composite masker with Gaussian-mean image occlusion.

Follows the Composite Pattern: internally instantiates two atomic maskers
(GaussianMeanMasker stub + TextAttentionMasker) and delegates image/text
occlusion to each respectively.

Registered as ``"crossmodal_gaussian"`` in the masker registry.

.. note::
    Skeleton only.  The image occlusion side (GaussianMeanMasker) is a
    placeholder — it currently falls back to VisionMeanMasker behaviour.
    The text side uses TextAttentionMasker (``"text_attn"``).
"""

from typing import Optional

from .base import BaseMasker
from .vision_mean import VisionMeanMasker  # placeholder for GaussianMeanMasker
from .text_attention import TextAttentionMasker
from ImputerFactory.data import PhysicalMask, ProcessorOutput, MaskerConfig
from . import register_masker


@register_masker("crossmodal_gaussian")
class CrossModalGaussianMasker(BaseMasker):
    """
    Cross-modal occlusion orchestrator with Gaussian-mean image occlusion.

    Delegates:
        - Image occlusion → GaussianMeanMasker (stub — falls back to
          VisionMeanMasker for now)
        - Text occlusion  → TextAttentionMasker (registered as ``"text_attn"``)

    The composite itself performs no element-wise operations.
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        super().__init__(config)
        # TODO: replace VisionMeanMasker with GaussianMeanMasker once
        #       the Gaussian-mean occlusion strategy is implemented.
        self._vision_masker = VisionMeanMasker(config=config)
        self._text_masker = TextAttentionMasker(config=config)

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # 1. Delegate image occlusion to GaussianMeanMasker (stub)
        masked = self._vision_masker.apply(processor_output, physical_mask)

        # 2. Delegate text occlusion to TextAttentionMasker (pixel_values pass-through)
        masked = self._text_masker.apply(masked, physical_mask)

        return masked
