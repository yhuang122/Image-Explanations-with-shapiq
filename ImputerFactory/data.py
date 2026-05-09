"""
Standardized data transfer formats for the ImageImputer pipeline.

All modules communicate through these well-defined objects,
ensuring a clean separation of concerns and framework-agnostic interfaces.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


# ─── Spatial Layout ───────────────────────────────────────────────────────────
# Produced by: Segmenter (once per image)
# Consumed by: ImageImputer (to translate coalitions → physical masks)

@dataclass
class SpatialLayout:
    """
    Describes the spatial division of the input into players.

    Attributes:
        n_players_image: Number of image players (patches/superpixels).
        n_players_text: Number of text players (tokens after stripping BOS/EOS).
        image_size: Height/width of the input image in pixels.
        patch_size: Edge length of a single patch.
        grid_size: Number of patches per side (image_size // patch_size).
        n_channels: Number of image channels (typically 3).
        model_type: 'clip', 'siglip', or 'siglip2'.
        text_total_length: Total token length expected by the model (e.g., 64).
        is_stateful: Whether the layout can change across iterations (e.g., AdaptiveSegmenter).
    """
    n_players_image: int
    n_players_text: int
    image_size: int
    patch_size: int
    grid_size: int
    n_channels: int
    model_type: str
    text_total_length: int
    is_stateful: bool = False


# ─── Physical Mask ────────────────────────────────────────────────────────────
# Produced by: ImageImputer (translating coalitions using the layout)
# Consumed by: Masker

@dataclass
class PhysicalMask:
    """
    Concrete, pixel/token-level masks ready to be applied to model inputs.

    Attributes:
        image_binary_mask: torch.Tensor (N_img, C, H, W) boolean/binary.
            Indicates which image pixels are active (1 = keep, 0 = occlude).
        text_attention_mask: torch.Tensor (N_txt, L) int.
            Attention mask for text tokens (1 = attend, 0 = ignore).
            Already padded with BOS/EOS ones based on model_type.
        batch_size_img: Number of image coalitions in this mask (may differ from text).
        batch_size_txt: Number of text coalitions in this mask.
    """
    image_binary_mask: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None
    
    @property
    def batch_size_img(self) -> int:
        return self.image_binary_mask.shape[0] if self.image_binary_mask is not None else 0
    
    @property
    def batch_size_txt(self) -> int:
        return self.text_attention_mask.shape[0] if self.text_attention_mask is not None else 0


# ─── Model Inputs ─────────────────────────────────────────────────────────────
# Standardized HuggingFace-style inputs dict wrapper

@dataclass
class ProcessorOutput:
    """
    Standardized wrapper around HuggingFace processor outputs.

    Attributes:
        pixel_values: torch.Tensor (B, C, H, W).
        input_ids: torch.Tensor (B, L).
        attention_mask: torch.Tensor (B, L).
        model_type: str identifying the model family.
    """
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    model_type: str
    
    @classmethod
    def from_hf_processor(cls, inputs: dict, model_type: str) -> "ProcessorOutput":
        """Create from a HuggingFace processor output dict."""
        return cls(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            model_type=model_type,
        )
    
    def to_dict(self) -> dict:
        """Convert back to a dict for model.forward(**kwargs)."""
        return {
            "pixel_values": self.pixel_values,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
    
    @property
    def device(self):
        return self.pixel_values.device
    
    def to(self, device):
        """Move all tensors to the specified device."""
        self.pixel_values = self.pixel_values.to(device)
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self
