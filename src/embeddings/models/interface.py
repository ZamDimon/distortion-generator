"""
General interface for embedding models.
"""

from pathlib import Path
import logging
import tensorflow as tf

class EmbeddingModel:
    """
    Embedding model interface
    """
    
    @property
    def raw(self) -> tf.keras.models.Model:
        """Returns the initializes model."""
        pass
    
    def summary(self) -> None:
        """Prints the model summary."""
        pass
    
    def save(self, path: Path) -> None:
        """
        Saves the model.
        
        Arguments:
            - path (Path) - path to the model
        """
        pass
    
    def print_example_predictions(self, 
                                  logger: logging.Logger,
                                  X: tf.Tensor, 
                                  y: tf.Tensor,
                                  predictions_number: int=35) -> None:
        """
        Prints example predictions.
        
        Arguments:
            - logger (logging.Logger) - logger
            - X (tf.Tensor) - images
            - y (tf.Tensor) - labels
            - predictions_number (int) - number of example predictions to print
        """
        pass
    