from abc import ABC, abstractmethod
from typing import Optional, Dict
import torch

from ..data import PhysicalMask, ProcessorOutput, MaskerConfig


class BaseMasker(ABC):
    """
    Abstract base class for feature occlusion strategies.

    A Masker takes a ProcessorOutput (original inputs) and a PhysicalMask
    (from the Segmenter) and produces modified inputs ready for model.forward().

    Lifecycle:
        1. __init__(config)   → receive MaskerConfig (strategy + per-strategy params)
        2. apply(processor_output, physical_mask) → modified ProcessorOutput
           (called for every batch during Shapley sampling)
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        self.config = config or MaskerConfig()

    @abstractmethod
    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """
        Apply the physical mask to the original inputs.

        Args:
            processor_output: Standardized original model inputs.
            physical_mask: Concrete pixel/token-level masks.

        Returns:
            ProcessorOutput with pixel_values and/or attention_mask modified.
        """
        pass
