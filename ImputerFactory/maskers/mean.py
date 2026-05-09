import torch

from .base import BaseMasker
from ImputerFactory.data import PhysicalMask, ProcessorOutput


class MeanMasker(BaseMasker):
    """
    Mean value occlusion: masks pixels by multiplying pixel_values with a
    binary mask. Since CLIP/SigLIP inputs are normalized (mean ≈ 0),
    zeroing out pixels is equivalent to filling with the dataset mean.

    For text, this masker replaces the attention_mask with the coalition-derived
    text attention mask.
    """

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # Shallow copy the container, then replace tensors
        masked = ProcessorOutput(
            pixel_values=processor_output.pixel_values.clone(),
            input_ids=processor_output.input_ids.clone(),
            attention_mask=processor_output.attention_mask.clone(),
            model_type=processor_output.model_type,
        )

        # Image: multiply pixel values by binary mask
        if physical_mask.image_binary_mask is not None:
            masked.pixel_values = masked.pixel_values * \
                physical_mask.image_binary_mask.to(masked.device)

        # Text: replace attention mask
        if physical_mask.text_attention_mask is not None:
            masked.attention_mask = physical_mask.text_attention_mask.to(masked.device)

        return masked
