"""
ImageImputerFactory — Modular, framework-agnostic image imputation pipeline.

Exports:
    ImageImputerFactory       — Central assembly line.
    ImageImputer              — Core orchestration engine.
    SegmenterConfig           — Typed segmenter configuration (strategy + params + metadata).
    MaskerConfig              — Typed masker configuration (strategy + params).
    ProcessorOutput           — Standardized model input format.
    PhysicalMask              — Concrete pixel/token-level mask.
    SpatialLayout             — Player-to-pixel spatial mapping.
    PatchSegmenter            — Rigid-grid segmenter (VLM baseline).
    VisionMeanMasker          — Pure image occlusion (multiplicative mask).
    VisionBlurMasker          — Gaussian blur occlusion (pre-computed kernel, conv2d).
    TextAttentionMasker       — Pure text occlusion (attention_mask swap).
    CrossModalMeanMasker      — Composite: VisionMeanMasker + TextAttentionMasker.
    CrossModalBlurMasker      — Composite: VisionBlurMasker + TextAttentionMasker.
"""

from .factory import ImageImputerFactory
from .core.imputer import ImageImputer
from .data import (
    SegmenterConfig,
    MaskerConfig,
    PatchParams,
    SlicParams,
    GradientGuidedParams,
    CrossModalMeanParams,
    VisionBlurParams,
    VisionMeanParams,
    TextAttentionParams,
    ProcessorOutput,
    PhysicalMask,
    SpatialLayout,
)
from .segmenters.patch import PatchSegmenter
from .maskers import (
    VisionMeanMasker,
    VisionBlurMasker,
    TextAttentionMasker,
    CrossModalMeanMasker,
    CrossModalBlurMasker,
)
from .regression import crossmodal_approximation, chunked_aggregate
