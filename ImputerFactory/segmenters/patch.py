from .base import BaseSegmenter

class PatchSegmenter(BaseSegmenter):
    """Rigid grids aligned with ViTs. Translates coalitions to patch-level masks."""
    
    def __init__(self, image_size, patch_size, n_channels, ops):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.ops = ops
        
    def generate_masks(self, coalitions):
        """
        Args:
            coalitions: torch.Tensor or np.ndarray, shape (batch_size, n_players_image).
                Boolean masks indicating active patches.
                
        Returns:
            torch.Tensor: shape (batch_size, n_channels, image_size, image_size).
                Expanded binary masks mapped to the physical image dimensions.
        """
        pass
