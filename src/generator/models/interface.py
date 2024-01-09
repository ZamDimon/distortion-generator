"""
General interface for embedding models.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

class GeneratorModel:
    """
    Generator model interface
    """
    
    @property
    def raw(self) -> tf.keras.models.Model:
        """Returns the initializes model."""
        pass
    
    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return self.raw._name
    
    def summary(self) -> None:
        """Prints the model summary."""
        self.raw.summary()
    
    def save(self, path: Path) -> None:
        """
        Saves the model.
        
        Arguments:
            - path (Path) - path to the model
        """
        self.raw.save(path)
        
    @classmethod
    def from_path(cls, path: Path, trainable=False) -> GeneratorModel:
        """
        Loads the model.
        
        Arguments:
            - path (Path) - path to the model
            - trainable (bool) - whether the model should be trainable
        """
        
        model = cls.__new__(cls)
        super(GeneratorModel, model).__init__()
        
        generator_model = tf.keras.models.load_model(path)
        generator_model._name = 'generator'
        generator_model.trainable = trainable
        
        model._model = generator_model
        return model
    
    