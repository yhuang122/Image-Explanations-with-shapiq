"""
DEPRECATED — CrossModalMeanMasker.

Kept for backward compatibility. The preferred cross-modal masker is
CrossModalCompositeMasker (composite of VisionMeanMasker + TextAttentionMasker).
"""

import torch

from .base import BaseMasker
from ImputerFactory.data import PhysicalMask, ProcessorOutput


class CrossModalMeanMasker(BaseMasker):
    """
    Cross-modal occlusion for Vision-Language Models.  (DEPRECATED)

    Image: multiplies pixel_values with a binary mask. Since CLIP/SigLIP
    inputs are normalized (mean ≈ 0), zeroing out pixels is equivalent to
    filling with the dataset mean.

    Text: replaces attention_mask with the coalition-derived text mask
    (1 = attend, 0 = ignore).

    Users should prefer CrossModalCompositeMasker, which decomposes into
    VisionMeanMasker + TextAttentionMasker via the Composite pattern.
    """

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # No defensive clones: `*` is out-of-place (creates a new tensor)
        # and attention_mask is replaced by rebinding the reference.
        # Originals are never mutated, so the "clone before mutate" contract
        # is satisfied without paying the (N,C,H,W) allocation up-front.
        pixel_values = processor_output.pixel_values
        input_ids = processor_output.input_ids
        attention_mask = processor_output.attention_mask

        if physical_mask.image_binary_mask is not None:
            pixel_values = pixel_values * \
                physical_mask.image_binary_mask.to(pixel_values.device)

        if physical_mask.text_attention_mask is not None:
            attention_mask = physical_mask.text_attention_mask.to(attention_mask.device)

        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_type=processor_output.model_type,
        )
