"""
General interface for embedding models.
"""

import logging
from pathlib import Path

import tensorflow as tf

from src.embeddings.hyperparameters import Hyperparameters

class EmbeddingModel:
    """
    Embedding model interface
    """
    
    def __init__(self, hyperparams: Hyperparameters) -> None:
        """
        Initializes the embedding model.
        
        Arguments:
            - hyperparams (Hyperparameters) - hyperparameters for the model
        """
        pass
    
    def raw(self) -> tf.keras.models.Model:
        """Returns the initializes model."""
        pass
    
    def summary(self) -> None:
        """Prints the model summary."""
        pass
    
    def save(self, base_path: Path) -> None:
        """Saves the model."""
        pass
    
    def print_example_predictions(self, 
                                  logger: logging.Logger,
                                  X: tf.Tensor, 
                                  y: tf.Tensor,
                                  predictions_number: int=35) -> None:
        """Prints example predictions."""
        pass
    