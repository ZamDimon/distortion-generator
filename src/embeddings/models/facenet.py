"""
Module with the embedding model for the LFW dataset.
"""

from __future__ import annotations
from pathlib import Path

import tensorflow as tf
from deepface import DeepFace

from src.embeddings.models.interface import EmbeddingModel

class FaceNetEmbeddingModel(EmbeddingModel):
    """
    Embedding model for the MNIST dataset.
    """
    
    def __init__(self, trainable: bool = False) -> None:
        """
        Embedding model is a target model that we need to train to make predictions.
        """
        
        facenet = DeepFace.build_model('Facenet512')
        
        # Adding unit normalization layer
        input_tensor = facenet.input
        output_tensor = facenet.layers[-1].output
        output_tensor = tf.keras.layers.UnitNormalization(axis=1)(output_tensor)
        facenet_with_normalization = tf.keras.Model(input_tensor, output_tensor)
        facenet_with_normalization.trainable = trainable
        self._model = facenet_with_normalization
    
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
    def from_path(cls, path: Path, trainable=False) -> FaceNetEmbeddingModel:
        return super().from_path(path, trainable)
