from typing import Optional
import numpy as np
import torch

from .base import BaseSegmenter
from . import register_segmenter
from ..data import SegmenterConfig, SpatialLayout, PhysicalMask


@register_segmenter("patch")
class PatchSegmenter(BaseSegmenter):
    """
    Rigid-grid segmenter aligned with Vision Transformer patch embeddings.

    Each patch is a single player. This is the default baseline for VLMs
    (CLIP, SigLIP) since their vision encoders natively operate on patches.
    """

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.n_channels = config.n_channels
        self.n_players_text = config.n_players_text
        self.model_type = config.model_type
        self.text_total_length = config.text_total_length
        self.grid_size = config.grid_size
        self.n_players_image = config.n_players_image

        # Pre-compute the layout
        self._layout = SpatialLayout(
            n_players_image=self.n_players_image,
            n_players_text=self.n_players_text,
            image_size=self.image_size,
            patch_size=self.patch_size,
            grid_size=self.grid_size,
            n_channels=self.n_channels,
            model_type=self.model_type,
            text_total_length=self.text_total_length,
            is_stateful=False,
        )

    def get_layout(self) -> SpatialLayout:
        return self._layout

    def generate_masks(
        self,
        coalitions_image: Optional[np.ndarray] = None,
        coalitions_text: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> PhysicalMask:
        mask = PhysicalMask()

        if coalitions_image is not None:
            mask.image_binary_mask = self._generate_image_mask(
                coalitions_image,
                device=device,
            )

        if coalitions_text is not None:
            mask.text_attention_mask = self._build_text_attention_mask(
                coalitions_text,
                device=device,
            )

        return mask

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _generate_image_mask(
        self,
        coalitions: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert patch-level coalition array to pixel-level binary mask.

        Args:
            coalitions: np.ndarray[bool], shape (N, n_players_image).

        Returns:
            torch.Tensor (N, C, H, W) with 1=keep, 0=occlude.
        """
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        n_coalitions = coalition_t.shape[0]

        # Expand each coalition value into a patch_size × patch_size block
        binary_masks = coalition_t \
            .repeat_interleave(self.patch_size ** 2, dim=1) \
            .reshape(n_coalitions, self.grid_size, self.grid_size,
                     self.patch_size, self.patch_size)

        # Rearrange to form the full image
        binary_masks = binary_masks \
            .permute(0, 1, 3, 2, 4) \
            .reshape(n_coalitions, self.image_size, self.image_size)

        # Add channel dimension: (N, H, W) → (N, C, H, W)
        binary_masks = binary_masks \
            .unsqueeze(1) \
            .repeat(1, self.n_channels, 1, 1)

        return binary_masks.float()
