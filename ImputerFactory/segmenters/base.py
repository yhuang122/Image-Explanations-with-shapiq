from abc import ABC, abstractmethod

class BaseSegmenter(ABC):
    """Abstract base class for spatial division methods."""
    
    @abstractmethod
    def generate_masks(self, coalitions):
        """Translate a boolean coalition array into spatial masks."""
        pass
