"""
Helper class to randomly select pairs of inputs with the same labels
"""

import random
from typing import Tuple, List, Any

import tensorflow as tf
import numpy as np

class PairPicker:
    """
    Class for picking inputs with the same labels
    """
    
    def __init__(self, X: list, y: list) -> None:
        """
        Initializes the PairPicker class
        
        Args:
            (X, y) - dataset consisting of inputs and labels
        """
        
        labels = np.unique(y)
        self._X_batches = {}
        for label in labels:
            batch = [X[i] for i in range(len(y)) if y[i] == label]
            self._X_batches[label] = batch
        
    def pick(self, label: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Picks a pair of the specified label
        """
        
        labels = self._X_batches[label]
        return random.choice(labels), random.choice(labels)
    
class PairValidationGenerator(tf.keras.utils.Sequence):
    """
    Class for generating pairs of inputs for validation
    """
    
    def __init__(self, X: list, y: list, classes_number: int = 3) -> None:
        """
        Initializes pair generator
        """
        assert len(X) == len(y), "The number of images and labels is the same"
        assert 2*classes_number <= len(np.unique(y)), "The number of classes must be less than the number of unique labels"
        
        labels = np.unique(y)
        np.random.shuffle(labels)
        labels = sorted(labels, key=lambda x: len(X[y == x]), reverse=True)
        labels = labels[:2*classes_number]
        
        self._positive_classes = labels[:classes_number]
        self._negative_classes = labels[classes_number:]

        self._X_batches = {}
        for label in labels:
            batch = [X[i] for i in range(len(y)) if y[i] == label]
            self._X_batches[label] = batch
    
    def get_test_classes(self) -> Tuple[List[Any], List[Any]]:
        """
        Returns a list of chosen positive and negative classes
        """
        
        return (self._positive_classes, self._negative_classes)
    
    def pick(self):
        """
        Generate one batch of data
        """
        
        positive_label = random.choice(self._positive_classes)
        negative_label = random.choice(self._negative_classes)
        return random.choice(self._X_batches[positive_label]), random.choice(self._X_batches[negative_label])
    