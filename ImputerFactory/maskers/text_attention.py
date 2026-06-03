"""
TextAttentionMasker — Pure text occlusion masker.

Operates exclusively on attention_mask. Must never touch pixel_values.
Designed to serve text-only models or be composed into cross-modal pipelines.
"""

import torch

from .base import BaseMasker
from . import register_masker
from ImputerFactory.data import PhysicalMask, ProcessorOutput


@register_masker("text_attn")
class TextAttentionMasker(BaseMasker):
    """
    Pure text occlusion via attention_mask replacement.

    Registered as ``"text_attn"`` in the masker registry.

    Contract:
        - Receives: ProcessorOutput (only attention_mask consumed) + PhysicalMask (.text_attention_mask)
        - Returns: ProcessorOutput with only attention_mask modified
        - Must NOT access pixel_values
    """

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        attention_mask = processor_output.attention_mask

        if physical_mask.text_attention_mask is not None:
            attention_mask = physical_mask.text_attention_mask.to(attention_mask.device)

        return ProcessorOutput(
            pixel_values=processor_output.pixel_values,    # pass-through
            input_ids=processor_output.input_ids,          # pass-through
            attention_mask=attention_mask,
            model_type=processor_output.model_type,
        )
