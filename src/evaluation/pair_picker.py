import random
from typing import Tuple

import tensorflow as tf
import numpy as np

class PairPicker:
    """
    Class for picking the same digits
    """
    
    def __init__(self, X: list, y: list) -> None:
        """
        Initializes the PairPicker class
        
        Args:
            (X, y) - dataset
        """
        unique_labels = np.unique(y)
        self._X_batches = {}
        for y_unique in unique_labels:
            batch = [X[i] for i in range(len(y)) if y[i] == y_unique]
            self._X_batches[y_unique] = batch
        
    def pick(self, digit: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Picks a pair of the specified digit
        """
        digits = self._X_batches[digit]
        return random.choice(digits), random.choice(digits)
    