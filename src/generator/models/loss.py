from __future__ import annotations

from enum import IntEnum

import tensorflow as tf
import tensorflow_io as tfio

# Weight coefficients for ImageLossType combined losses
COMBINED_L1_WEIGHT: float = 0.5
COMBINED_SSIM_WEIGHT: float = 0.5
COMBINED_SOBEL_WEIGHT: float = 0.5

class ImageLossType(IntEnum):
    """
    Enum for image loss type.
    """
    
    SSIM = 1
    L1 = 2
    MSE = 3
    SOBEL = 4
    COMBINED_SSIM = 5
    COMBINED_SOBEL = 6
    
    @staticmethod
    def from_str(image_loss: str) -> ImageLossType:
        """
        Returns the image loss type from the string.
        
        Arguments:
            - image_loss (str) - string to parse
        """
        
        losses = {
            'ssim': ImageLossType.SSIM,
            'l1': ImageLossType.L1,
            'mse': ImageLossType.MSE,
            'sobel': ImageLossType.SOBEL,
            'combined_ssim': ImageLossType.COMBINED_SSIM,
            'combined_sobel': ImageLossType.COMBINED_SOBEL
        }
        if image_loss not in losses:
            raise ValueError(f'Image loss type {image_loss} is not supported')
        
        return losses[image_loss]
    
    def _ssim_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for image comparison using structural similarity index.
        
        Arguments:
            - y_true (tf.Tensor) - true image
            - y_pred (tf.Tensor) - predicted image
        """
        
        return -(1.0 - self._pi_emb) * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=7))) / 2.0

    def _l1_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for image comparison using l1 difference error.
        
        Arguments:
            - y_true (tf.Tensor) - true image
            - y_pred (tf.Tensor) - predicted image
        """
        
        return -(1.0 - self._pi_emb) * tf.reduce_mean(tf.math.abs(y_true - y_pred))

    def _mse_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for image comparison using mean squared error.
        
        Arguments:
            - y_true (tf.Tensor) - true image
            - y_pred (tf.Tensor) - predicted image
        """
        
        return -(1.0 - self._pi_emb) * tf.reduce_mean(tf.math.square(y_true - y_pred))

    def _combined_ssim_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for image comparison using structural similarity index and mean squared error.
        
        Arguments:
            - y_true (tf.Tensor) - true image
            - y_pred (tf.Tensor) - predicted image
        """
        
        return (COMBINED_SSIM_WEIGHT * self._ssim_loss(y_true, y_pred) + 
                COMBINED_L1_WEIGHT * self._l1_loss(y_true, y_pred))
        
    def _sobel_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for image comparison using sobel filter.
        
        Arguments:
            - y_true (tf.Tensor) - true image
            - y_pred (tf.Tensor) - predicted image
        """
        
        true_mask = tfio.experimental.filter.sobel(y_true) # Getting a real mask
        diff_masked = true_mask * tf.math.abs(y_pred - y_true) # Calculating the difference under the mask
        return -(1.0 - self._pi_emb) * tf.reduce_mean(diff_masked)
    
    def _combined_sobel_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Loss function for image comparison using structural similarity index and mean squared error.
        
        Arguments:
            - y_true (tf.Tensor) - true image
            - y_pred (tf.Tensor) - predicted image
        """
        
        return (COMBINED_SOBEL_WEIGHT * self._sobel_loss(y_true, y_pred) + 
                COMBINED_L1_WEIGHT * self._l1_loss(y_true, y_pred))   
    
    def as_loss(self, pi_emb: float) -> tf.keras.losses.Loss:
        """
        Returns the loss function by the image loss type.
        
        Arguments:
            - pi_emb (float) - weight of the threshold loss
        """
        
        self._pi_emb = pi_emb
        match self.value:
            case ImageLossType.SSIM:
                return self._ssim_loss
            case ImageLossType.MSE:
                return self._mse_loss
            case ImageLossType.L1:
                return self._l1_loss
            case ImageLossType.SOBEL:
                return self._sobel_loss
            case ImageLossType.COMBINED_SSIM:
                return self._combined_ssim_loss
            case ImageLossType.COMBINED_SOBEL:
                return self._combined_sobel_loss
            case _:
                raise ValueError(f'Image loss type is not supported')
    