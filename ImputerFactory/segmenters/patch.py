from typing import Optional
import numpy as np
import torch

from .base import BaseSegmenter
from ImputerFactory.data import SpatialLayout, PhysicalMask


class PatchSegmenter(BaseSegmenter):
    """
    Rigid-grid segmenter aligned with Vision Transformer patch embeddings.

    Each patch is a single player. This is the default baseline for VLMs
    (CLIP, SigLIP) since their vision encoders natively operate on patches.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_channels: int,
        n_players_text: int,
        model_type: str,
        text_total_length: int = 64,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_players_text = n_players_text
        self.model_type = model_type
        self.text_total_length = text_total_length
        self.grid_size = image_size // patch_size
        self.n_players_image = self.grid_size ** 2

        # Pre-compute the layout
        self._layout = SpatialLayout(
            n_players_image=self.n_players_image,
            n_players_text=n_players_text,
            image_size=image_size,
            patch_size=patch_size,
            grid_size=self.grid_size,
            n_channels=n_channels,
            model_type=model_type,
            text_total_length=text_total_length,
            is_stateful=False,
        )

    def get_layout(self) -> SpatialLayout:
        return self._layout

    def generate_masks(
        self,
        coalitions_image: Optional[np.ndarray] = None,
        coalitions_text: Optional[np.ndarray] = None,
    ) -> PhysicalMask:
        mask = PhysicalMask()

        if coalitions_image is not None:
            mask.image_binary_mask = self._generate_image_mask(coalitions_image)

        if coalitions_text is not None:
            mask.text_attention_mask = self._generate_text_mask(coalitions_text)

        return mask

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _generate_image_mask(self, coalitions: np.ndarray) -> torch.Tensor:
        """
        Convert patch-level coalition array to pixel-level binary mask.

        Args:
            coalitions: np.ndarray[bool], shape (N, n_players_image).

        Returns:
            torch.Tensor (N, C, H, W) with 1=keep, 0=occlude.
        """
        coalition_t = torch.from_numpy(coalitions)
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

    def _generate_text_mask(self, coalitions: np.ndarray) -> torch.Tensor:
        """
        Convert token-level coalition array to attention mask.

        Handles BOS/EOS padding differences between CLIP and SigLIP.

        Args:
            coalitions: np.ndarray[bool], shape (N, n_players_text).

        Returns:
            torch.IntTensor (N, text_total_length) with 1=attend, 0=ignore.
        """
        coalition_t = torch.from_numpy(coalitions)
        n_coalitions = coalition_t.shape[0]

        if self.model_type == "siglip2":
            # SigLIP2: pad with ones after the valid text tokens
            pad_len = self.text_total_length - self.n_players_text
            text_masks = torch.cat(
                (coalition_t, torch.ones(n_coalitions, pad_len)),
                dim=1,
            ).int()
        elif self.model_type == "siglip":
            # SigLIP: same padding logic as siglip2
            pad_len = self.text_total_length - self.n_players_text
            text_masks = torch.cat(
                (coalition_t, torch.ones(n_coalitions, pad_len)),
                dim=1,
            ).int()
        elif self.model_type == "clip":
            # CLIP: BOS=1 at position 0, EOS=1 at the end
            text_masks = torch.cat(
                (torch.ones(n_coalitions, 1),
                 coalition_t,
                 torch.ones(n_coalitions, 1)),
                dim=1,
            ).int()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return text_masks
