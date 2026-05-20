"""
ImageImputerFactory — Modular, framework-agnostic image imputation pipeline.

Exports:
    ImageImputerFactory       — Central assembly line.
    ImageImputer              — Core orchestration engine.
    ImputerConfig             — Shared configuration (model metadata + segmenter + masker).
    ProcessorOutput           — Standardized model input format.
    PhysicalMask              — Concrete pixel/token-level mask.
    SpatialLayout             — Player-to-pixel spatial mapping.
    PatchSegmenter            — Rigid-grid segmenter (VLM baseline).
    VisionMeanMasker          — Pure image occlusion (multiplicative mask).
    TextAttentionMasker       — Pure text occlusion (attention_mask swap).
    CrossModalCompositeMasker — Composite: VisionMeanMasker + TextAttentionMasker.
"""

from .factory import ImageImputerFactory
from .core.imputer import ImageImputer
from .data import ImputerConfig, ProcessorOutput, PhysicalMask, SpatialLayout
from .segmenters.patch import PatchSegmenter
from .maskers import (
    VisionMeanMasker,
    TextAttentionMasker,
    CrossModalCompositeMasker,
)
