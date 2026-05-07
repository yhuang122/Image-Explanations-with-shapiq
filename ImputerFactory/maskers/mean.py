from .base import BaseMasker

class MeanMasker(BaseMasker):
    """Injects average pixel values (or zeros if normalized) directly into the input tensor."""
    
    def __init__(self, ops):
        self.ops = ops

    def apply(self, inputs, masks):
        """
        TODO: Apply masks to pixel values, e.g., inputs['pixel_values'] * masks
        """
        pass
