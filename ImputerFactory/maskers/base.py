from abc import ABC, abstractmethod

class BaseMasker(ABC):
    """Abstract base class responsible for feature occlusion."""
    
    @abstractmethod
    def apply(self, inputs, masks):
        """Apply the generated masks to the original inputs."""
        pass
