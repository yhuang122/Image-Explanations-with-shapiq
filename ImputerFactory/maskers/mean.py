from .base import BaseMasker

class MeanMasker(BaseMasker):
    """Injects average pixel values (or zeros if normalized) directly into the input tensor."""
    
    def __init__(self, ops):
        self.ops = ops

    def apply(self, inputs, masks):
        """
        Args:
            inputs: dict containing batched model inputs. E.g., inputs['pixel_values'] of 
                shape (batch_size, n_channels, image_size, image_size).
            masks: torch.Tensor, shape (batch_size, n_channels, image_size, image_size).
                Provided by the Segmenter.
                
        Returns:
            dict: Updated inputs dictionary with masked logic applied.
        """
        pass
