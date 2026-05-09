"""
ImageImputer — Core orchestration engine.

Coordinates the Segmenter → Masker → Model pipeline.
Handles batching, device placement, and post-processing.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

from ImputerFactory.data import (
    SpatialLayout,
    PhysicalMask,
    ProcessorOutput,
)
from ImputerFactory.segmenters.base import BaseSegmenter
from ImputerFactory.maskers.base import BaseMasker


class ImageImputer:
    """
    Core Orchestration Container.

    Lifecycle:
        1. __init__(model, processor, segmenter, masker, inputs_original)
        2. forward_1d(coalitions, batch_size) → outputs
        3. forward_crossmodal(coalitions_img, coalitions_txt, batch_size) → outputs

    Responsibilities:
        - Translates coalitions → physical masks (via Segmenter).
        - Applies masks to inputs (via Masker).
        - Manages batch iteration and model forward passes.
        - Post-processes model outputs (diagonal extraction, etc.).
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        segmenter: BaseSegmenter,
        masker: BaseMasker,
        inputs_original: ProcessorOutput,
        inputs_raw: dict = None,  # Raw HF processor output (for backwards compat)
        input_image: Any = None,  # PIL Image (kept for crossmodal edge cases)
        input_text: str = "",     # Text string (kept for crossmodal edge cases)
    ):
        self.model = model
        self.processor = processor
        self.segmenter = segmenter
        self.masker = masker

        # Preprocessed 1-sample inputs (owned by imputer, shared across calls)
        self.inputs_original = inputs_original
        # Raw HuggingFace output dict (kept for .tokens() etc.)
        self.inputs_raw = inputs_raw or {}
        # Raw input data (kept for crossmodal edge cases)
        self.input_image = input_image
        self.input_text = input_text

        # Inspect model for metadata
        vision_emb = model.vision_model.embeddings
        self.image_size = vision_emb.image_size
        self.patch_size = vision_emb.patch_size
        self.n_channels = vision_emb.config.num_channels
        self.grid_size = self.image_size // self.patch_size
        self.n_players_image = self.grid_size ** 2

        # Determine model type from the model name/config
        self.model_type = self._infer_model_type(model)

        # Layout from segmenter
        self.layout: SpatialLayout = segmenter.get_layout()

    @property
    def n_players_text(self) -> int:
        return self.layout.n_players_text

    @property
    def n_players(self) -> int:
        return self.n_players_image + self.n_players_text

    # ─── Public API ───────────────────────────────────────────────────────

    def forward_1d(
        self,
        coalitions: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Evaluate the model on a set of coalitions (1D input space).

        Args:
            coalitions: np.ndarray[bool], shape (N, n_players_image + n_players_text).
            batch_size: Number of coalitions per forward pass. Defaults to all.

        Returns:
            np.ndarray, shape (N,). Diagonal logits (image↔text similarity scores).
        """
        if batch_size is None:
            batch_size = coalitions.shape[0]

        n_coalitions = coalitions.shape[0]

        # Split coalitions into image and text parts
        coalitions_image = coalitions[:, :self.n_players_image]
        coalitions_text = coalitions[:, self.n_players_image:]

        # Generate all masks upfront (vectorized)
        physical_mask = self.segmenter.generate_masks(
            coalitions_image=coalitions_image,
            coalitions_text=coalitions_text,
        )

        # Batch loop
        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        all_outputs = []

        for batch_idx in range(batch_iters + 1):
            start = batch_idx * batch_size
            end = start + batch_size

            if batch_idx < batch_iters:
                actual_batch = batch_size
                # Repeat the original 1-sample input to match batch size
                inputs_batched = self._repeat_inputs(self.inputs_original, actual_batch)
                # Slice the appropriate mask slice
                mask_slice = self._slice_mask(physical_mask, start, end)
                # Apply mask
                masked_inputs = self.masker.apply(inputs_batched, mask_slice)
                # Forward
                outputs = self._model_forward(masked_inputs)
                # Extract diagonal
                outputs_1d = self._extract_diagonal(outputs)
                all_outputs.append(outputs_1d)

            elif batch_left > 0:
                # Last batch (smaller size)
                inputs_batched = self._repeat_inputs(self.inputs_original, batch_left)
                mask_slice = self._slice_mask(physical_mask, start, end)
                masked_inputs = self.masker.apply(inputs_batched, mask_slice)
                outputs = self._model_forward(masked_inputs)
                outputs_1d = self._extract_diagonal(outputs)
                all_outputs.append(outputs_1d)

        return torch.cat(all_outputs).numpy()

    def forward_crossmodal(
        self,
        coalitions_image: np.ndarray,
        coalitions_text: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Evaluate cross-modal coalitions (2D: image × text).

        Args:
            coalitions_image: np.ndarray[bool], shape (N_img, n_players_image).
            coalitions_text: np.ndarray[bool], shape (N_txt, n_players_text).
            batch_size: Number of coalitions per forward pass.

        Returns:
            np.ndarray, shape (N_img, N_txt). Full logits matrix.
        """
        if batch_size is None:
            batch_size = max(coalitions_image.shape[0], coalitions_text.shape[0])

        n_img = coalitions_image.shape[0]
        n_txt = coalitions_text.shape[0]

        # Generate all masks upfront
        physical_mask = self.segmenter.generate_masks(
            coalitions_image=coalitions_image,
            coalitions_text=coalitions_text,
        )

        # Image batch loop (outer)
        img_iters = n_img // batch_size
        img_left = n_img % batch_size

        # Pre-compute the "leftover" text batch inputs
        txt_iters = n_txt // batch_size
        txt_left = n_txt % batch_size
        txt_base = n_txt  # for clarity

        all_rows = []

        for img_batch_idx in range(img_iters + 1):
            img_start = img_batch_idx * batch_size
            img_end = img_start + batch_size

            # Determine this image batch size and masks
            if img_batch_idx < img_iters:
                img_bs = batch_size
                img_slice_mask = self._slice_mask_img(physical_mask, img_start, img_end)
                inputs_img_batched = self._repeat_inputs(self.inputs_original, img_bs)
                # Apply image mask
                inputs_img_masked = self.masker.apply(inputs_img_batched, img_slice_mask)
            elif img_left > 0:
                img_bs = img_left
                img_slice_mask = self._slice_mask_img(physical_mask, img_start, img_end)
                inputs_img_batched = self._repeat_inputs(self.inputs_original, img_bs)
                inputs_img_masked = self.masker.apply(inputs_img_batched, img_slice_mask)
            else:
                break

            col_outputs = []

            # Text batch loop (inner)
            for txt_batch_idx in range(txt_iters + 1):
                txt_start = txt_batch_idx * batch_size
                txt_end = txt_start + batch_size
                txt_slice_mask = self._slice_mask_txt(physical_mask, txt_start, txt_end)

                if txt_batch_idx < txt_iters:
                    txt_bs = batch_size
                elif txt_left > 0:
                    txt_bs = txt_left
                else:
                    break

                if txt_bs == img_bs:
                    # Fast path: batch sizes match, just apply text mask
                    masked = self.masker.apply(inputs_img_masked, txt_slice_mask)
                else:
                    # Edge case: txt batch ≠ img batch → reprocess with processor
                    device = next(self.model.parameters()).device
                    raw = self._preprocess_batch(img_bs, txt_bs)
                    # Apply image mask
                    img_only = PhysicalMask(image_binary_mask=img_slice_mask.image_binary_mask)
                    masked = self.masker.apply(self._dict_to_po(raw, device), img_only)
                    # Apply text mask
                    masked = self.masker.apply(masked, txt_slice_mask)

                outputs = self._model_forward(masked)
                col_outputs.append(outputs.logits_per_image.cpu())

            all_rows.append(torch.cat(col_outputs, dim=1))

        return torch.cat(all_rows, dim=0).numpy()

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _repeat_inputs(self, inputs: ProcessorOutput, batch_size: int) -> ProcessorOutput:
        """Repeat a 1-sample ProcessorOutput to the desired batch size."""
        return ProcessorOutput(
            pixel_values=inputs.pixel_values.expand(batch_size, -1, -1, -1).clone(),
            input_ids=inputs.input_ids.expand(batch_size, -1).clone(),
            attention_mask=inputs.attention_mask.expand(batch_size, -1).clone(),
            model_type=inputs.model_type,
        )

    def _slice_mask(self, pm: PhysicalMask, start: int, end: int) -> PhysicalMask:
        """Slice a PhysicalMask to the [start, end) range."""
        return PhysicalMask(
            image_binary_mask=pm.image_binary_mask[start:end] if pm.image_binary_mask is not None else None,
            text_attention_mask=pm.text_attention_mask[start:end] if pm.text_attention_mask is not None else None,
        )

    def _slice_mask_img(self, pm: PhysicalMask, start: int, end: int) -> PhysicalMask:
        """Slice only the image portion."""
        return PhysicalMask(
            image_binary_mask=pm.image_binary_mask[start:end] if pm.image_binary_mask is not None else None,
            text_attention_mask=None,
        )

    def _slice_mask_txt(self, pm: PhysicalMask, start: int, end: int) -> PhysicalMask:
        """Slice only the text portion."""
        return PhysicalMask(
            image_binary_mask=None,
            text_attention_mask=pm.text_attention_mask[start:end] if pm.text_attention_mask is not None else None,
        )

    def _model_forward(self, inputs: ProcessorOutput):
        """Run model forward pass. Returns raw model outputs."""
        device = next(self.model.parameters()).device
        with torch.no_grad():
            outputs = self.model(**{k: v.to(device) for k, v in inputs.to_dict().items()})
        return outputs

    def _extract_diagonal(self, outputs) -> torch.Tensor:
        """Extract diagonal from logits_per_image matrix (1D mode)."""
        return torch.diagonal(outputs.logits_per_image).cpu()

    @staticmethod
    def _infer_model_type(model) -> str:
        """Infer model type string from model object."""
        name_or_path = getattr(model, "name_or_path", "")
        if "siglip2" in name_or_path:
            return "siglip2"
        elif "siglip" in name_or_path:
            return "siglip"
        # Default fallback
        config_type = getattr(model.config, "model_type", "")
        if "siglip" in config_type.lower():
            return "siglip"
        return "clip"

    def _preprocess_batch(self, img_bs: int, txt_bs: int) -> dict:
        """Reprocess with processor for crossmodal edge cases."""
        kwargs = dict(
            images=[self.input_image] * img_bs,
            text=[self.input_text] * txt_bs,
            return_tensors="pt",
        )
        if self.model_type in ("siglip", "siglip2"):
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = 64
        elif self.model_type == "clip":
            kwargs["padding"] = True
        return self.processor(**kwargs)

    def _dict_to_po(self, raw: dict, device) -> ProcessorOutput:
        """Convert raw processor dict to ProcessorOutput on device."""
        return ProcessorOutput(
            pixel_values=raw["pixel_values"].to(device),
            input_ids=raw["input_ids"].to(device),
            attention_mask=raw["attention_mask"].to(device),
            model_type=self.model_type,
        )
