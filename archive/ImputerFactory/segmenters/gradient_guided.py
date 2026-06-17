"""
GradientGuidedSegmenter — Non-uniform layout via gradient-based saliency.

Algorithm:
    1. __init__ (GPU once): forward+backward pass to extract pixel-level
       gradient of image-text similarity. Aggregate to saliency map, run
       SLIC superpixels ON the saliency (not the raw image), producing a
       non-uniform region map that puts more players in high-saliency areas.
    2. generate_masks (per batch): map region-level coalitions → pixel-level
       binary masks via a single fancy-index operation.

Runtime dependencies (passed by Factory via constructor):
    model, processor, image (PIL), text (str)
"""

from typing import Optional, Any
import numpy as np
import torch

from .base import BaseSegmenter
from . import register_segmenter
from ..data import SegmenterConfig, SpatialLayout, PhysicalMask

try:
    from skimage.segmentation import slic as _skimage_slic
except ImportError:
    _skimage_slic = None


@register_segmenter("gradient_guided")
class GradientGuidedSegmenter(BaseSegmenter):
    """Static non-uniform layout using gradient-based saliency.

    Computes pixel-level gradients of the image-text similarity,
    aggregates them into a saliency map, and uses SLIC superpixels
    to produce a non-uniform spatial division that puts more
    players in high-saliency regions.

    Args:
        config: ``SegmenterConfig`` with strategy ``"gradient_guided"``.
        model: HuggingFace VLM for the gradient-extraction forward pass.
        processor: Corresponding HF processor.
        image: PIL Image for preprocessing.
        text: Text string for preprocessing.

    Strategy parameters (via ``config.gradient_guided``):
        ``n_segments`` (int | None): target superpixel count.
        ``None`` means derive from ``grid_size`` or fall back to 49.
    """

    def __init__(
        self,
        config: SegmenterConfig,
        model: Any = None,
        processor: Any = None,
        image: Any = None,
        text: str = None,
    ):
        super().__init__(config)
        if _skimage_slic is None:
            raise ImportError(
                "GradientGuidedSegmenter requires scikit-image. "
                "Install with `pip install scikit-image`."
            )

        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.n_channels = config.n_channels
        self.n_players_text = config.n_players_text
        self.model_type = config.model_type
        self.text_total_length = config.text_total_length
        self.grid_size = config.grid_size

        self._saliency: Optional[np.ndarray] = None

        if model is None or processor is None or image is None or text is None:
            raise ValueError(
                "GradientGuidedSegmenter requires ``model``, ``processor``, "
                "``image``, and ``text``.  The Factory passes these "
                "automatically when ``strategy='gradient_guided'`` is selected."
            )

        # ── GPU: gradient extraction + SLIC on saliency ─────────────────
        device = next(model.parameters()).device

        # 1. Preprocess image (keep grad) and text
        img_inputs = processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device)
        pixel_values.requires_grad = True

        txt_inputs = processor(text=text, return_tensors="pt", padding=True)

        # 2. Forward: full CLIP model
        outputs = model(
            pixel_values=pixel_values,
            input_ids=txt_inputs["input_ids"].to(device),
            attention_mask=txt_inputs["attention_mask"].to(device),
        )

        # 3. Backward from the image-text similarity
        score = outputs.logits_per_image.sum()
        score.backward()

        grad = pixel_values.grad  # (1, 3, H, W)

        # 4. Saliency: |grad| averaged over colour channels → (H, W)
        saliency = grad.abs().mean(dim=1).squeeze(0)
        saliency = saliency.detach().cpu().numpy()

        # Normalise to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        self._saliency = saliency

        # 5. SLIC superpixels on the saliency map
        # Priority: typed param > grid-based > hardcoded default.
        n_segments_kw = config.gradient_guided.n_segments
        if n_segments_kw is not None:
            n_segments = int(n_segments_kw)
        elif self.grid_size > 0:
            n_segments = self.grid_size * self.grid_size
        else:
            n_segments = 49  # same default as SLICSegmenter
        saliency_rgb = np.stack([saliency] * 3, axis=-1)
        region_labels = _skimage_slic(
            saliency_rgb,
            n_segments=n_segments,
            compactness=5.0,
            start_label=0,
            channel_axis=-1,
        )

        # Safety: clamp in case SLIC produces more regions than requested
        if region_labels.max() >= n_segments:
            region_labels = np.clip(region_labels, 0, n_segments - 1)

        # Remap to contiguous [0, K) like SLICSegmenter
        unique_ids, packed = np.unique(region_labels, return_inverse=True)
        label_map = packed.reshape(self.image_size, self.image_size).astype(np.int64)
        self.n_players_image = int(unique_ids.size)
        self._label_map = torch.from_numpy(label_map)  # CPU (H, W) int64
        self._label_map_by_device = {torch.device("cpu"): self._label_map}

        # Recompute grid_size for the layout to reflect actual player count
        grid_side = int(np.sqrt(self.n_players_image))
        self.grid_size = grid_side + 1 if grid_side * grid_side < self.n_players_image else grid_side
        if self.grid_size < 1:
            self.grid_size = 1

        self._layout = SpatialLayout(
            n_players_image=self.n_players_image,
            n_players_text=self.n_players_text,
            image_size=self.image_size,
            patch_size=self.patch_size,
            grid_size=self.grid_size,
            n_channels=self.n_channels,
            model_type=self.model_type,
            text_total_length=self.text_total_length,
            is_stateful=False,
        )

        # Clean up
        pixel_values.grad = None
        model.zero_grad()
        torch.cuda.empty_cache()

    # ─── Public contract ──────────────────────────────────────────────────

    def get_layout(self) -> SpatialLayout:
        return self._layout

    def generate_masks(
        self,
        coalitions_image: Optional[np.ndarray] = None,
        coalitions_text: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> PhysicalMask:
        mask = PhysicalMask()

        if coalitions_image is not None:
            mask.image_binary_mask = self._scatter_image_mask(
                coalitions_image,
                device=device,
            )

        if coalitions_text is not None:
            mask.text_attention_mask = self._build_text_attention_mask(
                coalitions_text,
                device=device,
            )

        return mask

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _scatter_image_mask(
        self,
        coalitions: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Translate (N, K) bool coalitions → (N, C, H, W) float pixel masks."""
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        label_map = self._label_map_for(coalition_t.device)
        pixel_masks = coalition_t[:, label_map]  # (N, H, W) bool
        return pixel_masks.unsqueeze(1).expand(
            -1, self.n_channels, -1, -1,
        ).float()

    def _label_map_for(self, device: torch.device) -> torch.Tensor:
        """Return the 2D label map on the target device."""
        device = torch.device(device)
        cached = self._label_map_by_device.get(device)
        if cached is None:
            cached = self._label_map.to(device=device, non_blocking=True)
            self._label_map_by_device[device] = cached
        return cached
