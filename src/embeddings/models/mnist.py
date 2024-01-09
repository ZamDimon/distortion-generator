"""
Module with the embedding model for the MNIST dataset.
"""

from __future__ import annotations
from pathlib import Path

import tensorflow as tf

from src.embeddings.models.interface import EmbeddingModel
from src.embeddings.hyperparameters import EmbeddingHyperparameters

class MNISTEmbeddingModel(EmbeddingModel):
    """
    Embedding model for the MNIST dataset.
    """
    
    def __init__(self, hyperparams: EmbeddingHyperparameters) -> None:
        """
        Embedding model is a target model that we need to train to make predictions.
        
        Arguments:
            - hyperparams (EmbeddingModelHyperparameters) - hyperparameters for the model
        """
        
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=hyperparams.input_shape),
            tf.keras.layers.Dense(
                hyperparams.hidden_layer_size,
                name="hidden_layer",
                kernel_initializer=tf.keras.initializers.HeUniform(),
                kernel_regularizer=tf.keras.regularizers.l2(hyperparams.regulizer)),
            tf.keras.layers.LeakyReLU(alpha=hyperparams.alpha),
            tf.keras.layers.Dense(
                hyperparams.embedding_size,  
                name="output_layer",
                kernel_initializer=tf.keras.initializers.HeUniform(),
                kernel_regularizer=tf.keras.regularizers.l2(hyperparams.regulizer),
                activation=None
            ),
            tf.keras.layers.UnitNormalization(axis=1)
        ])
        self._hyperparams = hyperparams
    
    @property
    def raw(self) -> tf.keras.models.Model:
        return self._model
    
    @property
    def name(self) -> str:
        return super().name
    
    def summary(self) -> None:
        super().summary()
        
    def save(self, path: Path) -> None:
        super().save(path)
    
    @classmethod
    def from_path(cls, path: Path, trainable=False) -> MNISTEmbeddingModel:
        return super().from_path(path, trainable)
