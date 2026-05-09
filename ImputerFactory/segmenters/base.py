from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np

from ImputerFactory.data import SpatialLayout, PhysicalMask


class BaseSegmenter(ABC):
    """
    Abstract base class for spatial division methods.

    A Segmenter produces a SpatialLayout describing the mapping
    from players to physical pixels/tokens, and can optionally
    convert coalition arrays into PhysicalMasks.

    Lifecycle:
        1. __init__(...)   → configure parameters
        2. get_layout()    → produce SpatialLayout (called once per image)
        3. generate_masks(coalitions_image, coalitions_text) → PhysicalMask
           (called many times during Shapley sampling)
    """

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
