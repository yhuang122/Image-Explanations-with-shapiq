"""
AttentionMasker — Self-attention occlusion via negative-infinity mask injection.

Intercepts the model's self-attention mechanism and injects negative-infinity
masks to suppress attended features. Requires PyTorch ``register_forward_hook``
or HuggingFace ``output_attentions`` override.

.. note::
    Skeleton only. The :meth:`apply` method is not yet implemented.
    Implementation requires:
        1. Registering a forward hook on the attention layer(s).
        2. Constructing an attention mask of shape ``(batch, heads, seq_len, seq_len)``.
        3. Setting masked positions to ``-inf`` before softmax.

Registered as ``"attention"`` in the masker registry.
"""

from typing import Optional

from .base import BaseMasker
from . import register_masker
from ImputerFactory.data import MaskerConfig, PhysicalMask, ProcessorOutput


@register_masker("attention")
class AttentionMasker(BaseMasker):
    """
    Self-attention occlusion masker.

    This masker occludes image features by injecting negative-infinity masks
    into the self-attention mechanism, preventing the model from attending
    to masked regions. This is more faithful than pixel-level occlusion
    because it operates inside the model's latent space.

    Delegates:
        The actual hook registration is model-specific (CLIP vs SigLIP's
        vision encoder). Subclasses or wrapper strategies may be needed.

    Registered as ``"attention"``.
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        super().__init__(config)
        # TODO: implement forward hook registration and -inf mask construction.
        #       self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._hooks = []

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """
        Apply self-attention occlusion.

        Args:
            processor_output: Preprocessed model inputs (pixel_values, input_ids, attention_mask).
            physical_mask: Contains the image_binary_mask tensor to inject into attention.

        Returns:
            ProcessorOutput unchanged for now (skeleton — actual hook captures masks
            from an external reference and applies them during forward pass).
        """
        # TODO:
        #   1. Convert physical_mask.image_binary_mask to per-patch attention mask.
        #   2. Store it in a module-level reference for the forward hook.
        #   3. The hook (registered in __init__) reads the reference and constructs
        #      the -inf mask prior to softmax.
        #
        # For now, pass through unchanged.
        return processor_output
