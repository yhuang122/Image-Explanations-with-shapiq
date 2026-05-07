from .core.imputer import ImageImputer
from .segmenters.patch import PatchSegmenter
# from .segmenters.slic import SLICSegmenter
from .maskers.mean import MeanMasker
from .adapters.torch_ops import TorchOps

class ImageImputerFactory:
    """Factory for assembling the ImageImputer pipeline."""
    
    def build(self, model, processor=None, backend="pytorch", accelerator=None):
        # 1. Select the backend tensor operations
        if backend == "pytorch":
            ops = TorchOps()
        else:
            raise NotImplementedError(f"Backend {backend} is not supported yet.")

        # 2. Introspect the model to determine the baseline components
        model_type = getattr(model.config, "model_type", "").lower()
        
        # Currently explicitly focusing on Vision-Language Models (VLMs)
        if "clip" in model_type or "siglip" in model_type:
            # Baseline for Vision-Language Models
            patch_size = model.vision_model.embeddings.patch_size
            image_size = model.vision_model.embeddings.image_size
            
            segmenter = PatchSegmenter(image_size=image_size, patch_size=patch_size, ops=ops)
            masker = MeanMasker(ops=ops) 
        else:
            raise ValueError(f"Currently only Vision-Language Models (e.g., CLIP, SigLIP) are supported. Unsupported model type: {model_type}")

        # 3. Inject Accelerators (if any)
        if accelerator == "hybrid":
            # Override baseline with a HybridSegmenter
            pass
        elif accelerator == "gradient":
            # Override baseline with a GradientGuidedSegmenter
            pass

        # 4. Assemble and return the Imputer container
        return ImageImputer(
            model=model,
            processor=processor,
            segmenter=segmenter,
            masker=masker,
            tensor_ops=ops
        )
