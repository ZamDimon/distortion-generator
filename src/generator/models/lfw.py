"""
Package for LFW generator model
"""

from __future__ import annotations
from typing import Tuple, TypeAlias
from pathlib import Path

import tensorflow as tf

from src.generator.models.interface import GeneratorModel

ImageShape: TypeAlias = Tuple[int, int, int]

class LFWGeneratorModel(GeneratorModel):
    """
    LFW Generator model
    """
    
    @staticmethod
    def _conv_block(
        inputs: tf.keras.layers.Layer, 
        num_filters: int,
    ) -> tf.keras.layers.Layer:
        x = tf.keras.layers.Conv2D(
            num_filters, 
            3, 
            padding="same", 
            kernel_initializer=tf.keras.initializers.HeUniform()
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        x = tf.keras.layers.Conv2D(
            num_filters, 
            3, 
            padding="same", 
            kernel_initializer=tf.keras.initializers.HeUniform()
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        return x

    @staticmethod
    def _encoder_block(
        inputs: tf.keras.layers.Layer, 
        num_filters: int,
        crop: bool = False
    ) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
        x = LFWGeneratorModel._conv_block(inputs, num_filters)
        
        if crop:
            x = tf.keras.layers.Cropping2D(cropping=((1,1), (1,1)))(x)
        
        p = tf.keras.layers.MaxPool2D((2,2))(x)
        return x, p

    @staticmethod
    def _decoder_block(
        inputs: tf.keras.layers.Layer, 
        skip: tf.keras.layers.Layer, num_filters: int) -> tf.keras.layers.Layer:
        x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = LFWGeneratorModel._conv_block(x, num_filters)
        return x

    
    def __init__(self, input_shape: ImageShape = (160, 160, 3)) -> None:
        """
        Creates a generator that inputs an image and returns yet another image
        
        Parameters:
            - input_shape (ImageShape) - shape of the input image
        """
        
        inputs = tf.keras.layers.Input(input_shape)

        s1, p1 = self._encoder_block(inputs, 32) # -> 80 x 80
        s2, p2 = self._encoder_block(p1, 64) # -> 40 x 40
        s3, p3 = self._encoder_block(p2, 128) # -> 20 x 20

        # Bridge
        b1 = self._conv_block(p3, 256) # -> 10 x 10

        # Decoder
        d1 = self._decoder_block(b1, s3, 128) # -> 10 x 10
        d2 = self._decoder_block(d1, s2, 64) # -> 20 x 20
        d3 = self._decoder_block(d2, s1, 32) # -> 40 x 40
        
        outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid", kernel_initializer=tf.keras.initializers.HeUniform())(d3)
        
        # Returning a retrieved model
        self._model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='generator')
        
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
    def from_path(cls, path: Path, trainable=False) -> LFWGeneratorModel:
        return super().from_path(path, trainable)
    
    