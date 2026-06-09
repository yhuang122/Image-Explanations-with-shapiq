"""
CrossModalMeanMasker — Composite masker for Vision-Language Models.

Follows the Composite Pattern: internally instantiates two atomic maskers
(VisionMeanMasker + TextAttentionMasker) and delegates image/text occlusion
to each respectively. Owns no low-level tensor math itself.

Registered as ``"crossmodal_mean"`` in the masker registry.
"""

from typing import Optional

from .base import BaseMasker
from .vision_mean import VisionMeanMasker
from .text_attention import TextAttentionMasker
from ..data import PhysicalMask, ProcessorOutput, MaskerConfig
from . import register_masker


@register_masker("crossmodal_mean")
class CrossModalMeanMasker(BaseMasker):
    """
    Cross-modal occlusion orchestrator for VLMs.

    Delegates:
        - Image occlusion → VisionMeanMasker (registered as ``"vision_mean"``)
        - Text occlusion  → TextAttentionMasker (registered as ``"text_attn"``)

    The composite itself performs no element-wise operations.
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        super().__init__(config)
        self._vision_masker = VisionMeanMasker(config=config)
        self._text_masker = TextAttentionMasker(config=config)

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # 1. Delegate image occlusion to VisionMeanMasker
        masked = self._vision_masker.apply(processor_output, physical_mask)

        # 2. Delegate text occlusion to TextAttentionMasker (pixel_values pass-through)
        masked = self._text_masker.apply(masked, physical_mask)

        return masked
