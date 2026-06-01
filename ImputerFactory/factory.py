"""
ImageImputerFactory — Central assembly line.

Inspects the model, selects optimal defaults, injects segmenters,
and returns a fully wired ImageImputer ready for Shapley evaluation.
"""

from typing import Optional, Any

from .core.imputer import ImageImputer
from .data import ImputerConfig, ProcessorOutput
from .segmenters.base import BaseSegmenter
from .segmenters import get_segmenter
from .maskers.base import BaseMasker
from .maskers import get_masker


class ImageImputerFactory:
    """
    Assembles the ImageImputer pipeline from components.

    Usage:
        factory = ImageImputerFactory()
        imputer = factory.build(model, processor, input_image, input_text)

    Segmenter options:
        - None              → auto: PatchSegmenter for ViT, SLICSegmenter for CNN
        - "patch"           → PatchSegmenter (rigid grid, baseline for ViT)
        - "slic"            → SLICSegmenter (perceptual superpixels, required for CNN)
        - "gradient_guided" → GradientGuidedSegmenter (saliency-guided, opt-in)

    Segmenter kwargs (forwarded to segmenter constructor):
        - For "slic":       optionally override n_segments, compactness, sigma.
        - For "gradient_guided": optionally override model/processor/image/text
          used for the saliency forward pass.
        - For "patch":      no meaningful kwargs.
        The factory auto-fills the required defaults (e.g. image_array for SLIC,
        model/processor/image/text for gradient-guided); you only need to supply
        overrides to change default behavior.

    Masker options:
        - None / "crossmodal_mean" → CrossModalMeanMasker (baseline for VLMs)
        - "vision"           → VisionMeanMasker (image-only)
        - "text"             → TextAttentionMasker (text-only)
    """

    def build(
        self,
        model: Any,
        processor: Any,
        input_image: Any,
        input_text: str,
        segmenter: Optional[str] = None,
        segmenter_kwargs: Optional[dict] = {},
        masker: Optional[str] = None,
        use_amp: bool = False,
    ) -> ImageImputer:
        """
        Build a fully assembled ImageImputer.

        Args:
            model: HuggingFace VLM (CLIPModel, SiglipModel, etc.).
            processor: Corresponding HuggingFace processor.
            input_image: PIL Image or path.
            input_text: Text string.
            segmenter: Optional segmenter strategy ("patch", "slic").
            segmenter_kwargs: Optional dict forwarded to the segmenter
                constructor. Keys depend on the segmenter type:
                - "patch": no meaningful keys.
                - "slic":  "n_segments", "compactness", "sigma", etc.
                - "gradient_guided": "model", "processor", "image", "text".
                The factory injects required defaults before instantiation
                (e.g. image_array for SLIC), so you only need overrides.
            masker: Optional masker strategy ("crossmodal_mean", "vision", "text").
            use_amp: If True, model.forward runs under torch.autocast(fp16)
                on CUDA. Useful for ViT-L/14 with large coalitions.

        Returns:
            Configured ImageImputer ready for forward_1d / forward_crossmodal.
        """
        # ── 1. Infer model type ─────────────────────────────────────────
        model_type = self._infer_model_type(model)

        # ── 2. Extract model dimensions (ViT or CNN backbone) ──────────
        image_size, patch_size, n_channels = self._extract_vision_dims(model)
        is_vit = patch_size > 0
        grid_size = image_size // patch_size if is_vit else 0
        # n_players_image is provisional for ViT (grid²) and resolved later
        # for SLIC (the actual superpixel count comes from skimage).
        n_players_image = grid_size ** 2 if is_vit else 0

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
            segmenter=segmenter,
            masker=masker,
            segmenter_kwargs=segmenter_kwargs,  # populated by segmenter strategies in future
            use_amp=use_amp,
        )

        # ── 5. Select Segmenter ────────────────────────────────────────
        # Default routing: ViT → patch grid, CNN backbone → SLIC superpixels.
        # Caller can override by passing segmenter="patch"/"slic" explicitly.
        if config.segmenter is None:
            config.segmenter = "patch" if is_vit else "slic"
        if config.segmenter == "slic":
            # SLIC needs the raw image to plan superpixels (CPU once).
            config.segmenter_kwargs.setdefault("image_array", input_image)
        if config.segmenter == "gradient_guided":
            # GradientGuidedSegmenter needs model/processor/image/text for
            # the forward+backward pass that extracts the saliency map.
            config.segmenter_kwargs.setdefault("model", model)
            config.segmenter_kwargs.setdefault("processor", processor)
            config.segmenter_kwargs.setdefault("image", input_image)
            config.segmenter_kwargs.setdefault("text", input_text)
        segmenter = self._create_segmenter(config)
        # SLIC determines the real player count at __init__; sync config.
        config.n_players_image = segmenter.get_layout().n_players_image

        # ── 6. Select Masker (default: "crossmodal_mean") ────────────────────
        if config.masker is None:
            config.masker = "crossmodal_mean"
        masker = self._create_masker(config)

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
        config = getattr(model, "config", None)
        candidates = [
            getattr(model, "name_or_path", ""),
            getattr(config, "_name_or_path", ""),
            getattr(config, "name_or_path", ""),
            getattr(config, "model_type", ""),
            type(model).__name__,
            type(config).__name__ if config is not None else "",
        ]
        normalized = [str(value).lower() for value in candidates if value]

        if any("siglip2" in value for value in normalized):
            return "siglip2"
        elif any("siglip" in value for value in normalized):
            return "siglip"
        return "clip"

    @staticmethod
    def _extract_vision_dims(model) -> tuple:
        """
        Return (image_size, patch_size, n_channels).

        ViT backbones expose `patch_size` via `model.vision_model.embeddings`.
        CNN backbones (e.g. CLIP-RN50) do not — fall back to `vision_config`
        and return patch_size=0 to signal "no rigid grid". The Factory uses
        the zero to route to SLICSegmenter automatically.
        """
        vision = model.vision_model
        embeddings = getattr(vision, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "patch_size"):
            return (
                embeddings.image_size,
                embeddings.patch_size,
                embeddings.config.num_channels,
            )
        # CNN fallback
        vc = getattr(model.config, "vision_config", None) or model.config
        image_size = int(getattr(vc, "image_size", 224))
        n_channels = int(getattr(vc, "num_channels", 3))
        return (image_size, 0, n_channels)

    @staticmethod
    def _preprocess(processor, image, text, model_type: str) -> dict:
        """Run processor once to get the input structure."""
        kwargs = dict(images=image, text=text, return_tensors="pt")
        if model_type in ("siglip", "siglip2"):
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = 64
        elif model_type == "clip":
            kwargs["padding"] = True
        outputs = processor(**kwargs)
        # SigLIP tokenizer does not return attention_mask; derive it
        if "attention_mask" not in outputs:
            outputs["attention_mask"] = (outputs["input_ids"] != 1).long()
        return outputs

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
        """Look up and instantiate the Segmenter via registry."""
        cls = get_segmenter(config.segmenter)
        return cls(config=config)

    @staticmethod
    def _create_masker(config: ImputerConfig) -> BaseMasker:
        """Look up and instantiate the Masker via registry."""
        cls = get_masker(config.masker)
        return cls()

