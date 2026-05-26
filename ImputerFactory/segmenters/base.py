from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np

from ImputerFactory.data import ImputerConfig, SpatialLayout, PhysicalMask


class BaseSegmenter(ABC):
    """
    Abstract base class for spatial division methods.

    A Segmenter produces a SpatialLayout describing the mapping
    from players to physical pixels/tokens, and can optionally
    convert coalition arrays into PhysicalMasks.

    Lifecycle:
        1. __init__(config) → receive shared ImputerConfig
        2. get_layout()    → produce SpatialLayout (called once per image)
        3. generate_masks(coalitions_image, coalitions_text) → PhysicalMask
           (called many times during Shapley sampling)
    """

    def __init__(self, config: ImputerConfig):
        self.config = config

    @abstractmethod
    def get_layout(self) -> SpatialLayout:
        """
        Produce the spatial layout describing player ↔ pixel/token mapping.

        Called once per image. Stateful segmenters (Adaptive/Hybrid) may
        produce different layouts across iterations.
        """
        pass

    @abstractmethod
    def generate_masks(
        self,
        coalitions_image: Optional[np.ndarray] = None,
        coalitions_text: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> PhysicalMask:
        """
        Translate boolean coalition arrays into concrete physical masks.

        Args:
            coalitions_image: np.ndarray[bool], shape (N_img, n_players_image).
            coalitions_text: np.ndarray[bool], shape (N_txt, n_players_text).

        Returns:
            PhysicalMask with image_binary_mask and/or text_attention_mask set.
        """
        pass

    # ─── Shared helper for VLM text masking ───────────────────────────────
    # Image segmentation differs per Segmenter (patch / SLIC / future),
    # but the text-side padding is purely a function of model_type and is
    # identical across all VLM segmenters. Centralised here so subclasses
    # only need to implement the image side.

    def _build_text_attention_mask(
        self,
        coalitions: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert token-level coalitions to an attention mask matching the
        model's expected total length.

        Args:
            coalitions: np.ndarray[bool], shape (N, n_players_text).

        Returns:
            torch.IntTensor (N, text_total_length) — 1=attend, 0=ignore.
        """
        cfg = self.config
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        n_coalitions = coalition_t.shape[0]

        if cfg.model_type in ("siglip", "siglip2"):
            # SigLIP / SigLIP2: right-pad with 1s after the valid tokens
            pad_len = cfg.text_total_length - cfg.n_players_text
            return torch.cat(
                (
                    coalition_t,
                    torch.ones(n_coalitions, pad_len, device=coalition_t.device),
                ),
                dim=1,
            ).int()
        if cfg.model_type == "clip":
            # CLIP: wrap with BOS=1, EOS=1
            return torch.cat(
                (torch.ones(n_coalitions, 1, device=coalition_t.device),
                 coalition_t,
                 torch.ones(n_coalitions, 1, device=coalition_t.device)),
                dim=1,
            ).int()
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")
