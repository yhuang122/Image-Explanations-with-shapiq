"""
ImageImputerFactory — Central assembly line.

Inspects the model, enriches ``SegmenterConfig`` with model metadata,
selects components, and returns a fully wired ``ImageImputer``.
"""

from typing import Optional, Any

from .core.imputer import ImageImputer
from .data import SegmenterConfig, MaskerConfig, ProcessorOutput
from .segmenters.base import BaseSegmenter
from .segmenters import get_segmenter
from .maskers.base import BaseMasker
from .maskers import get_masker


class ImageImputerFactory:
    """
    Assembles the ImageImputer pipeline from typed configs.

    Usage::

        factory = ImageImputerFactory()
        imputer = factory.build(model, processor, input_image, input_text)

    Or with explicit config::

        seg_cfg = SegmenterConfig(strategy="slic", slic=SlicParams(n_segments=60))
        msk_cfg = MaskerConfig(strategy="crossmodal_mean")
        imputer = factory.build(model, processor, img, txt,
                                segmenter_config=seg_cfg,
                                masker_config=msk_cfg)

    Segmenter strategies (``SegmenterConfig.strategy``):
        - ``"patch"`` (default) — rigid grid, baseline for ViT.
        - ``"slic"`` — perceptual superpixels, required for CNN backbones.
        - ``"gradient_guided"`` — saliency-guided non-uniform layout.

    Masker strategies (``MaskerConfig.strategy``):
        - ``"crossmodal_mean"`` (default) — composite: vision-mean + text-attn.
        - ``"crossmodal_gaussian"`` — composite: vision-gaussian (stub) + text-attn.
        - ``"vision_mean"`` — image-only multiplicative mask.
        - ``"text_attn"`` — text-only attention-mask swap.
    """

    def build(
        self,
        model: Any,
        processor: Any,
        input_image: Any,
        input_text: str,
        segmenter_config: Optional[SegmenterConfig] = None,
        masker_config: Optional[MaskerConfig] = None,
        use_amp: bool = False,
    ) -> ImageImputer:
        """
        Build a fully assembled ImageImputer.

        Args:
            model: HuggingFace VLM (CLIPModel, SiglipModel, etc.).
            processor: Corresponding HuggingFace processor.
            input_image: PIL Image or path.
            input_text: Text string.
            segmenter_config: Optional ``SegmenterConfig``.  ``None`` (or
                strategy ``"patch"``) uses the default rigid-grid segmenter.
                Strategy ``"slic"`` requires the ``image_array`` run-time
                dependency, which the Factory passes automatically.
            masker_config: Optional ``MaskerConfig``.  ``None`` (or strategy
                ``"crossmodal_mean"``) uses the default composite masker.
            use_amp: If True, model.forward runs under ``torch.autocast(fp16)``
                on CUDA.  Useful for ViT-L/14 with large coalitions.

        Returns:
            Configured ``ImageImputer`` ready for ``forward_1d`` /
            ``forward_crossmodal``.
        """
        # ── 0. Default configs (caller may provide overrides) ───────────
        if segmenter_config is None:
            segmenter_config = SegmenterConfig()
        if masker_config is None:
            masker_config = MaskerConfig()

        # ── 1. Infer model type ─────────────────────────────────────────
        model_type = self._infer_model_type(model)

        # ── 2. Extract model dimensions (ViT or CNN backbone) ──────────
        image_size, patch_size, n_channels = self._extract_vision_dims(model)
        is_vit = patch_size > 0
        grid_size = image_size // patch_size if is_vit else 0
        n_players_image = grid_size ** 2 if is_vit else 0  # provisional

        # ── 3. Preprocess once to determine text players ─────────────────
        inputs_dict = self._preprocess(processor, input_image, input_text, model_type)
        n_players_text = self._count_text_players(inputs_dict, model_type)
        text_total_length = inputs_dict["input_ids"].shape[1]

        # ── 4. Enrich SegmenterConfig with model metadata ───────────────
        segmenter_config.model_type = model_type
        segmenter_config.image_size = image_size
        segmenter_config.patch_size = patch_size
        segmenter_config.n_channels = n_channels
        segmenter_config.grid_size = grid_size
        segmenter_config.n_players_image = n_players_image
        segmenter_config.n_players_text = n_players_text
        segmenter_config.text_total_length = text_total_length

        # ── 5. Create Segmenter (per-strategy dispatch) ──────────────────
        strategy = segmenter_config.strategy
        if strategy == "slic":
            segmenter = self._create_segmenter(
                segmenter_config, image_array=input_image,
            )
        elif strategy == "gradient_guided":
            segmenter = self._create_segmenter(
                segmenter_config,
                model=model, processor=processor,
                image=input_image, text=input_text,
            )
        else:
            # "patch" or any unrecognized strategy → defaults to patch
            segmenter = self._create_segmenter(segmenter_config)

        # Sync actual player count from layout (SLIC count may differ)
        segmenter_config.n_players_image = segmenter.get_layout().n_players_image

        # ── 6. Create Masker ────────────────────────────────────────────
        masker = self._create_masker(masker_config)

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
            inputs_original=inputs_original,
            inputs_raw=inputs_dict,
            input_image=input_image,
            input_text=input_text,
            use_amp=use_amp,
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

        ViT backbones expose ``patch_size`` via ``model.vision_model.embeddings``.
        CNN backbones (e.g. CLIP-RN50) do not — fall back to ``vision_config``
        and return patch_size=0 to signal "no rigid grid".  The Factory uses
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
    def _create_segmenter(config: SegmenterConfig, **extra_kwargs) -> BaseSegmenter:
        """Look up and instantiate the Segmenter via registry."""
        cls = get_segmenter(config.strategy)
        return cls(config=config, **extra_kwargs)

    @staticmethod
    def _create_masker(config: MaskerConfig) -> BaseMasker:
        """Look up and instantiate the Masker via registry."""
        cls = get_masker(config.strategy)
        return cls(config=config)

