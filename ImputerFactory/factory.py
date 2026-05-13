"""
ImageImputerFactory — Central assembly line.

Inspects the model, selects optimal defaults, injects accelerators,
and returns a fully wired ImageImputer ready for Shapley evaluation.
"""

from typing import Optional, Any

from .core.imputer import ImageImputer
from .data import ImputerConfig, ProcessorOutput
from .segmenters.patch import PatchSegmenter
from .segmenters.base import BaseSegmenter
from .maskers.mean import CrossModalMeanMasker
from .maskers.base import BaseMasker


class ImageImputerFactory:
    """
    Assembles the ImageImputer pipeline from components.

    Usage:
        factory = ImageImputerFactory()
        imputer = factory.build(model, processor, input_image, input_text)

    Accelerator options (future):
        - None        → PatchSegmenter + CrossModalMeanMasker (baseline for VLMs)
        - "gradient"  → GradientGuidedSegmenter
        - "adaptive"  → AdaptiveSegmenter
        - "hybrid"    → HybridSegmenter
    """

    def build(
        self,
        model: Any,
        processor: Any,
        input_image: Any,
        input_text: str,
        accelerator: Optional[str] = None,
    ) -> ImageImputer:
        """
        Build a fully assembled ImageImputer.

        Args:
            model: HuggingFace VLM (CLIPModel, SiglipModel, etc.).
            processor: Corresponding HuggingFace processor.
            input_image: PIL Image or path.
            input_text: Text string.
            accelerator: Optional accelerator strategy.

        Returns:
            Configured ImageImputer ready for forward_1d / forward_crossmodal.
        """
        # ── 1. Infer model type ─────────────────────────────────────────
        model_type = self._infer_model_type(model)

        # ── 2. Extract model dimensions ─────────────────────────────────
        image_size = model.vision_model.embeddings.image_size
        patch_size = model.vision_model.embeddings.patch_size
        n_channels = model.vision_model.embeddings.config.num_channels
        grid_size = image_size // patch_size
        n_players_image = grid_size ** 2

        # ── 3. Preprocess once to determine text players ─────────────────
        inputs_dict = self._preprocess(processor, input_image, input_text, model_type)
        n_players_text = self._count_text_players(inputs_dict, model_type)
        text_total_length = inputs_dict["input_ids"].shape[1]

        # ── 4. Build shared config ──────────────────────────────────────
        config = ImputerConfig(
            model_type=model_type,
            image_size=image_size,
            patch_size=patch_size,
            n_channels=n_channels,
            n_players_image=n_players_image,
            n_players_text=n_players_text,
            grid_size=grid_size,
            text_total_length=text_total_length,
            accelerator=accelerator,
            segmenter_kwargs={},  # populated by accelerators in future
        )

        # ── 5. Select Segmenter (receives config) ───────────────────────
        segmenter = self._create_segmenter(config)

        # ── 6. Select Masker ────────────────────────────────────────────
        masker = self._create_masker(accelerator)

        # ── 7. Build the standardized 1-sample inputs ───────────────────
        inputs_original = ProcessorOutput(
            pixel_values=inputs_dict["pixel_values"],
            input_ids=inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            model_type=model_type,
        )

        # ── 8. Assemble and return ──────────────────────────────────────
        return ImageImputer(
            model=model,
            processor=processor,
            segmenter=segmenter,
            masker=masker,
            config=config,
            inputs_original=inputs_original,
            inputs_raw=inputs_dict,
            input_image=input_image,
            input_text=input_text,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _infer_model_type(model) -> str:
        """Infer model type from model name/config."""
        name_or_path = getattr(model, "name_or_path", "")
        if "siglip2" in name_or_path:
            return "siglip2"
        elif "siglip" in name_or_path:
            return "siglip"
        config_type = getattr(model.config, "model_type", "").lower()
        if "siglip" in config_type:
            return "siglip"
        return "clip"

    @staticmethod
    def _preprocess(processor, image, text, model_type: str) -> dict:
        """Run processor once to get the input structure."""
        kwargs = dict(images=image, text=text, return_tensors="pt")
        if model_type in ("siglip", "siglip2"):
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = 64
        elif model_type == "clip":
            kwargs["padding"] = True
        return processor(**kwargs)

    @staticmethod
    def _count_text_players(inputs: dict, model_type: str) -> int:
        """
        Count valid text tokens (excluding BOS/EOS/padding).

        CLIP: strips 2 tokens (BOS + EOS).
        SigLIP: counts non-zero input_ids.
        SigLIP2: counts non-zero input_ids minus 1.
        """
        input_ids = inputs["input_ids"][0]  # first sample
        if model_type == "siglip2":
            return input_ids.count_nonzero().item() - 1
        elif model_type == "siglip":
            return (input_ids != 1).count_nonzero().item()
        elif model_type == "clip":
            return input_ids.size(0) - 2
        return 0

    @staticmethod
    def _create_segmenter(config: ImputerConfig) -> BaseSegmenter:
        """Create the appropriate Segmenter based on the shared config."""
        if config.accelerator in ("gradient", "adaptive", "hybrid"):
            raise NotImplementedError(
                f"Accelerator '{config.accelerator}' is not yet implemented. "
                f"Use accelerator=None for baseline PatchSegmenter."
            )

        # Default: PatchSegmenter (baseline for VLMs)
        return PatchSegmenter(config=config)

    @staticmethod
    def _create_masker(accelerator: Optional[str]) -> BaseMasker:
        """Create the appropriate Masker."""
        # For VLMs, CrossModalMeanMasker is the default
        # AttentionMasker would be used for more advanced occlusion
        return CrossModalMeanMasker()

