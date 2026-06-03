"""
VisionMeanMasker — Pure image occlusion masker.

Operates exclusively on pixel_values. Must never touch input_ids or
attention_mask. Designed to serve PatchSegmenter (ViT), SLICSegmenter
(CNN), or any future image-only segmenter.
"""

import torch

from .base import BaseMasker
from . import register_masker
from ImputerFactory.data import PhysicalMask, ProcessorOutput


@register_masker("vision_mean")
class VisionMeanMasker(BaseMasker):
    """
    Pure image occlusion via multiplicative binary mask.

    Since CLIP/SigLIP inputs are normalized (mean ≈ 0), zeroing out
    pixels is equivalent to filling with the dataset mean.

    Registered as ``"vision_mean"`` in the masker registry.

    Contract:
        - Receives: ProcessorOutput (only pixel_values consumed) + PhysicalMask (.image_binary_mask)
        - Returns: ProcessorOutput with only pixel_values modified
        - Must NOT access input_ids or attention_mask
    """

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # `*` is out-of-place → no mutation of originals
        pixel_values = processor_output.pixel_values

        if physical_mask.image_binary_mask is not None:
            pixel_values = pixel_values * \
                physical_mask.image_binary_mask.to(pixel_values.device)

        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=processor_output.input_ids,       # pass-through
            attention_mask=processor_output.attention_mask,  # pass-through
            model_type=processor_output.model_type,
        )
