"""AttentionMasker — Self-attention occlusion via negative-infinity mask injection.

Intercepts the model's vision self-attention and injects -inf masks before
softmax, preventing masked image patches from being attended to.

Registered as ``"attention"`` in the masker registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from shapiq.imputer.vision.base import PhysicalMask, ProcessorOutput
from shapiq.imputer.vision.maskers.base import Masker, MaskerConfig

from . import register_masker

if TYPE_CHECKING:
    from collections.abc import Callable


@register_masker("attention")
class AttentionMasker(Masker):
    """Self-attention occlusion masker.

    Instead of zeroing pixels (VisionMeanMasker), this masker injects
    negative-infinity into the vision encoder's self-attention scores
    for masked patches. This prevents the model from attending to
    occluded regions entirely within its latent space.

    Lifecycle:
        1. ``__init__`` — register forward pre-hooks on every CLIPEncoderLayer.
        2. ``apply()`` — convert pixel mask to patch attention bias, store for hooks.
        3. hooks fire during ``model.forward()`` — inject bias into causal mask.
    """

    def __init__(self, config: MaskerConfig | None = None, model=None) -> None:
        super().__init__(config)
        self._current_attn_bias: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._patch_size: int | None = None
        self._grid_size: int | None = None
        self._call_count = 0

        if model is not None:
            self._setup(model)

    def _setup(self, model) -> None:
        """Extract model metadata, register pre-forward hooks on vision encoder."""
        vision = model.vision_model
        embeddings = getattr(vision, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "patch_size"):
            self._patch_size = embeddings.patch_size
            self._grid_size = embeddings.image_size // embeddings.patch_size
        else:
            self._patch_size = 0
            self._grid_size = 0

        for layer in vision.encoder.layers:
            hook = layer.self_attn.register_forward_pre_hook(
                self._make_pre_hook(), with_kwargs=True
            )
            self._hooks.append(hook)

    def _make_pre_hook(self) -> Callable:
        """Create a pre-forward hook that injects attention bias."""

        def hook(module, args, kwargs):
            self._call_count += 1
            if self._current_attn_bias is None:
                return args, kwargs

            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return args, kwargs

            batch_size = hidden_states.shape[0]
            device = hidden_states.device
            dtype = hidden_states.dtype

            bias = self._current_attn_bias.to(device=device, dtype=dtype)
            if bias.shape[0] != batch_size:
                bias = bias.amax(dim=0, keepdim=True)
            if bias.shape[0] == 1 and batch_size > 1:
                bias = bias.expand(batch_size, -1, -1, -1)

            args_list = list(args)
            if "causal_attention_mask" in kwargs:
                m = kwargs["causal_attention_mask"]
                kwargs["causal_attention_mask"] = bias if m is None else m + bias
            elif len(args_list) > 2:
                m = args_list[2]
                args_list[2] = bias if m is None else m + bias
                args = tuple(args_list)

            return args, kwargs

        return hook

    def apply(
        self, processor_output: ProcessorOutput, physical_mask: PhysicalMask
    ) -> ProcessorOutput:
        """Apply attention bias based on the image mask.

        Args:
            processor_output: Original model inputs (pass-through,
                only used to determine device/dtype).
            physical_mask: Must contain ``image_binary_mask``.

        Returns:
            Unmodified ``processor_output`` (the bias is injected
            via hooks during ``model.forward()``).
        """
        image_mask = physical_mask.image_binary_mask
        if image_mask is not None and self._patch_size is not None and self._patch_size > 0:
            self._current_attn_bias = self._compute_attention_bias(image_mask)
        else:
            self._current_attn_bias = None
        return processor_output

    def _compute_attention_bias(self, image_mask: torch.Tensor) -> torch.Tensor:
        """Convert a pixel-level image mask to a per-patch attention bias.

        Args:
            image_mask: (N, C, H, W) float tensor (1 = keep, 0 = occlude).

        Returns:
            (N, 1, seq_len, seq_len) float tensor with -inf at masked positions.
        """
        N, C, H, W = image_mask.shape
        p = self._patch_size
        g = self._grid_size
        if p is None or g is None:
            msg = "patch_size and grid_size must be set before computing attention bias"
            raise RuntimeError(msg)
        seq_len = g * g + 1  # CLS token + patches

        # Average mask over channels, then reshape to patches
        patches = image_mask.reshape(N, C, g, p, g, p)
        patch_keep = patches.mean(dim=(1, 3, 5))  # (N, g, g)
        patch_masked_flat = (patch_keep < 0.5).reshape(N, -1)  # (N, g*g)

        bias = torch.zeros(N, 1, seq_len, seq_len, dtype=torch.float32, device=image_mask.device)
        patch_positions = torch.arange(1, seq_len, device=image_mask.device)

        for n in range(N):
            ids = patch_positions[patch_masked_flat[n]]
            if ids.numel() > 0:
                bias[n, 0, :, ids] = float("-inf")  # KEY dim only

        return bias

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
