from abc import ABC, abstractmethod

class TensorOps(ABC):
    """Abstract interface for tensor manipulations to keep the pipeline framework-agnostic."""
    
    @abstractmethod
    def create_text_attention_masks(self, coalitions_text, model_type, total_length, n_players_text):
        """
        Args:
            coalitions_text: torch.Tensor shape (batch_size, n_players_text).
            model_type: str, e.g., 'clip', 'siglip'.
            total_length: int, total token length expected by the model (e.g. 64).
            n_players_text: int, number of valid text tokens.
            
        Returns:
            torch.IntTensor: shape (batch_size, total_length).
        """
        pass
