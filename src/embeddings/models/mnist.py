"""
Module with the embedding model for the MNIST dataset.
"""

from __future__ import annotations
from pathlib import Path
import logging

import tensorflow as tf

from src.embeddings.hyperparameters import Hyperparameters

class MNISTEmbeddingModel:
    """
    Embedding model for the MNIST dataset.
    """
    
    def __init__(self, hyperparams: Hyperparameters) -> None:
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
        """
        Returns the model.
        """
        
        return self._model
    
    def summary(self) -> None:
        """
        Prints the model summary.
        """
        
        self._model.summary()
        
    def save(self, path: Path) -> None:
        """
        Saves the model.
        
        Arguments:
            - path (Path) - path to the model
        """
        
        self._model.save(path)
        
    def print_example_predictions(self, 
                                  logger: logging.Logger,
                                  X: tf.Tensor, 
                                  y: tf.Tensor,
                                  predictions_number: int=35) -> None:
        """
        Shows example predictions.
        
        Arguments:
            - logger (logging.Logger) - logger for printing
            - X (tf.Tensor) - images
            - y (tf.Tensor) - labels
            - predictions_number (int) - number of predictions to show
        """
        
        predictions = self._model.predict(X[:predictions_number])
        for (label, prediction) in sorted(zip(y[:predictions_number], predictions), key=lambda x: x[0]):
            logger.info(f'{label}: {prediction}')

