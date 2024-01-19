"""
Package for LFW generator model
"""

from __future__ import annotations
from typing import Tuple
from pathlib import Path

import tensorflow as tf

from src.generator.models.interface import GeneratorModel
from src.generator.models.interface import ImageShape

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

    
    def __init__(self, input_shape: ImageShape = (160, 160, 3), grayscale: bool = True) -> None:
        """
        Creates a generator that inputs an image and returns yet another image
        
        Parameters:
            - input_shape (ImageShape) - shape of the input image
            - grayscale (bool) - whether the image is grayscale
        """
        
        inputs = tf.keras.layers.Input(input_shape)

        s1, p1 = self._encoder_block(inputs, 64) # -> 80 x 80
        s2, p2 = self._encoder_block(p1, 128) # -> 40 x 40
        s3, p3 = self._encoder_block(p2, 256) # -> 20 x 20
        s4, p4 = self._encoder_block(p3, 512) # -> 10 x 10
        
        # Bridge
        b1 = self._conv_block(p4, 1024) # -> 5 x 5

        # Decoder
        d1 = self._decoder_block(b1, s4, 512) # -> 10 x 10
        d2 = self._decoder_block(d1, s3, 256) # -> 20 x 20
        d3 = self._decoder_block(d2, s2, 128) # -> 40 x 40
        d4 = self._decoder_block(d3, s1, 64) # -> 80 x 80
        
        outputs = tf.keras.layers.Conv2D(1 if grayscale else 3, 1, padding="same", activation="sigmoid", kernel_initializer=tf.keras.initializers.HeUniform())(d4)
        
        if grayscale:
            outputs = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(outputs)
        
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
    
    