"""
Package responsible for training the generator model.
Contains the Trainer class which, using embedded and generator models,
trains the generator model.
"""

from __future__ import annotations

from typing import Tuple, TypeAlias
from enum import IntEnum

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from src.datasets.interface import DatasetLoader
from src.embeddings.models.interface import EmbeddingModel
from src.generator.models.interface import GeneratorModel
from src.generator.hyperparameters import GeneratorHyperparameters

from src.generator.models.loss import ImageLossType

ImageShape: TypeAlias = Tuple[int, int, int]


class TrainerNetwork:
    """
    Trainer for the generator model.
    """
    
    COMBINED_L1_WEIGHT: float = 0.5
    COMBINED_SSIM_WEIGHT: float = 0.5
    COMBINED_SOBEL_WEIGHT: float = 0.5
    
    def __init__(self,
                 hyperparams: GeneratorHyperparameters,
                 generator: GeneratorModel, 
                 embedding_model: EmbeddingModel) -> None:
        """
        Creates a trainer that trains the generator model.
        
        Parameters:
            - hyperparams (GeneratorHyperparameters) - hyperparameters for the model
            - generator (tf.keras.models.Model) - generator model
            - embedding_model (tf.keras.models.Model) - embedding model
            - input_shape (ImageShape) - shape of the input image
        """
        
        self._hyperparams = hyperparams # Saving hyperparameters
        self._embedding_model = embedding_model # Saving embedding model
        self._generator_model = generator # Saving generator model
        self._history = None # At some point we will save the history of the model
        
        # Define the input layer
        img_input = tf.keras.layers.Input((*hyperparams.input_shape, 3), name='image_input')

        # Define the output image from the generator
        output_image = generator.raw(img_input)
        output_image._name = 'generated_image'

        # Embedding layer applied to the generated image
        embedding_generated = embedding_model.raw(output_image)
        embedding_generated._name = 'generated_embedding'
        
        # Build a model
        self._model = tf.keras.models.Model(inputs=img_input, outputs=[output_image, embedding_generated])    
    
    def _threshold_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for embedded vectors comparison 
        using threshold loss defined in our paper.
        
        Arguments:
            - y_true (tf.Tensor) - true embedded vector
            - y_pred (tf.Tensor) - predicted embedded vector
        """
        
        differences = tf.math.reduce_sum(tf.math.square(y_true - y_pred), axis=1)
        differences = tf.nn.relu(differences - self._hyperparams.threshold)
        return self._hyperparams.pi_emb * tf.reduce_mean(differences) / 4.0
       
    def summary(self) -> None:
        self._model.summary()
        
    def train(self, 
              dataset: DatasetLoader, 
              image_loss: ImageLossType = ImageLossType.MSE,
              ) -> None:
        """
        Trains the model using the provided dataset.
        
        Arguments:
            - dataset (tf.data.Dataset) - dataset to train the model on
            - image_loss (ImageLossType, optional) - type of the image loss to apply while training. Defaults to ImageLossType.MSE
        """
        
        (X_train, _), _ = dataset.get()
        
        y_train_embeddings = self._embedding_model.raw.predict(np.array(X_train, dtype=np.float32), verbose=1)
        y_train_merged = [X_train, y_train_embeddings]
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self._hyperparams.learning_rate)
        self._model.compile(optimizer=optimizer, loss={
            self._generator_model.name: image_loss.as_loss(self._hyperparams.pi_emb),
            self._embedding_model.name: self._threshold_loss
        })
        self._history = self._model.fit(
            X_train,
            y_train_merged,
            batch_size=self._hyperparams.batch_size,
            epochs=self._hyperparams.epochs,
            validation_split=self._hyperparams.validation_split,
            verbose=1
        )
        
    def get_history(self) -> tf.keras.callbacks.History:
        """
        Returns the history of the model.
        """
        
        if self._history is None:
            raise Exception("The model hasn't been trained yet")
        
        return self._history
    