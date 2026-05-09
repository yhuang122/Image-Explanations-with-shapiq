"""
ImageImputerFactory — Modular, framework-agnostic image imputation pipeline.

Exports:
    ImageImputerFactory  — Central assembly line.
    ImageImputer         — Core orchestration engine.
    ProcessorOutput      — Standardized model input format.
    PhysicalMask         — Concrete pixel/token-level mask.
    SpatialLayout        — Player-to-pixel spatial mapping.
    PatchSegmenter       — Rigid-grid segmenter (VLM baseline).
    MeanMasker           — Mean-value occlusion masker.
"""

from .factory import ImageImputerFactory
from .core.imputer import ImageImputer
from .data import ProcessorOutput, PhysicalMask, SpatialLayout
from .segmenters.patch import PatchSegmenter
from .maskers.mean import MeanMasker
