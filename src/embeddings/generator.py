"""
Package responsible for generating triplets for the triplet loss
using generator (which implements tf.keras.utils.Sequence).
"""

import tensorflow as tf
import numpy as np

from src.embeddings.hyperparameters import Hyperparameters


class TripletGenerator(tf.keras.utils.Sequence):
    """
    Triplets generator for the triplet network.
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 hyperparams: Hyperparameters,
                 embedding_model: tf.keras.models.Model) -> None:
        """
        Initializes triplets generator. 
        
        Args:
            - X (np.ndarray): The images.
            - y (np.ndarray): The labels.
            - batch_size (int): The batch size.
            - embedding_model (tf.keras.models.Model): The embedding model.
        """
        
        self._n = len(X)
        assert self._n == len(y), "The number of images and labels is the same"
        
        self._X = X
        self._y = y
        self._hyperparams = hyperparams
        self._batch_size = hyperparams.batch_size
        self._hard_size = hyperparams.batch_size // 2
        self._model = embedding_model
        
        self._unique_labels = np.unique(y)
        self._X_batches = {}
        for y_unique in self._unique_labels:
            batch = [X[i] for i in range(self._n) if y[i] == y_unique]
            self._X_batches[y_unique] = batch
            assert len(batch) >= 2*hyperparams.batch_size, 'Each batch with the same label must contain at least 2*batch_size of images'
        
        self.on_epoch_end()
        
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        
        np.random.shuffle(self._unique_labels)
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self._n // self._batch_size
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        
        np.random.shuffle(self._unique_labels)
        positive_index, negative_index = self._unique_labels[:2]
        
        positive_images = self._X_batches[positive_index][:2*self._batch_size]
        negative_images = self._X_batches[negative_index][:self._batch_size]
        
        np.random.shuffle(positive_images)
        np.random.shuffle(negative_images)
        
        anchors = np.array([positive_images[2*i] for i in range(self._batch_size)])
        positives = np.array([positive_images[2*i+1] for i in range(self._batch_size)])
        negatives = np.array([negative_images[i] for i in range(self._batch_size)])
        
        # Picking the hardest triplets
        anchor_embeddings = self._model.predict(anchors, verbose=0)
        positive_embeddings = self._model.predict(positives, verbose=0)
        negative_embeddings = self._model.predict(negatives, verbose=0)
        
        ap_distances = np.sum(np.square(anchor_embeddings - positive_embeddings), axis=1)
        an_distances = np.sum(np.square(anchor_embeddings - negative_embeddings), axis=1)
        
        losses = ap_distances - an_distances
        triplets = [(anchors[i], positives[i], negatives[i]) for i in range(self._batch_size)]
        sorted_zip = sorted(zip(losses, triplets), key=lambda x: x[0])
        sorted_triplets = [triplet for _, triplet in sorted_zip]
        sorted_triplets = sorted_triplets[self._hard_size//2:3*self._hard_size//2]
        
        # Unpacking sorted triplets
        hard_anchors, hard_positives, hard_negatives = [], [], []
        for triplet in sorted_triplets:
            hard_anchors.append(triplet[0])
            hard_positives.append(triplet[1])
            hard_negatives.append(triplet[2])
            
        hard_anchors, hard_positives, hard_negatives = np.array(hard_anchors), np.array(hard_positives), np.array(hard_negatives)
        return ([hard_anchors, hard_positives, hard_negatives], 
                np.ones((self._batch_size, 3*self._hyperparams.embedding_size)))