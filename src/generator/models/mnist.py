"""
Package for MNIST generator model
"""

from __future__ import annotations
from typing import Tuple, TypeAlias
from pathlib import Path

import tensorflow as tf

from src.generator.models.interface import GeneratorModel

ImageShape: TypeAlias = Tuple[int, int, int]

class MNISTGeneratorModel(GeneratorModel):
    """
    MNIST Generator model
    """
    
    def __init__(self, input_shape: ImageShape = (28, 28, 1)) -> None:
        """
        Creates a generator that inputs an image and returns yet another image
        
        Parameters:
            - input_shape (ImageShape) - shape of the input image
        """
        
        # Define the input layer
        inputs = tf.keras.layers.Input(input_shape)

        # Encoder (contracting path)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(inputs)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(conv3)

        # Decoder (expansive path)
        up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
        up4 = tf.keras.layers.concatenate([up4, conv2], axis=3)
        conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(up4)
        conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(conv4)

        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = tf.keras.layers.concatenate([up5, conv1], axis=3)
        conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(up5)
        conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeUniform())(conv5)

        # Output layer
        output_image = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='output_image')(conv5)
        
        # Returning a retrieved model
        self._model = tf.keras.models.Model(inputs=inputs, outputs=output_image, name='generator')
        
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
    def from_path(cls, path: Path, trainable=False) -> MNISTGeneratorModel:
        return super().from_path(path, trainable)
    
    