"""
TensorOps — Framework-agnostic tensor manipulation interface.

This abstraction layer ensures Segmenters and Maskers remain
independent of PyTorch vs JAX specifics.

Current status: Placeholder. The prototype VLM pipeline operates
entirely in PyTorch; JAX support will be added in a future iteration.
"""

from abc import ABC, abstractmethod


class TensorOps(ABC):
    """Abstract interface for backend-specific tensor operations."""
    pass
