from typing import TypeAlias, Dict

import tensorflow as tf
import numpy as np

UserId: TypeAlias = int

class AuthenticationSystem:
    """
    Mock of a face recognition system
    """
    
    LOGIN_FAILED: UserId = -1
    
    def __init__(self, generator: tf.keras.Model, embedding: tf.keras.Model, threshold: float) -> None:
        """
        Creates a simple implementation of a digit recognition system
        
        Args:
            generator - model of the generator
            threshold - threshold above which two images are considered to be different
        """
        
        self._threshold = threshold
        self._generator = generator
        self._embedding = embedding
        
        # Variables for storage
        self._storage: Dict[UserId, tf.Tensor] = {}
        self._last_key: UserId = 0
     
    def set_threshold(self, new_threshold: float) -> None:
        """
        Sets a new threshold for classification
        
        Parameters:
            new_threshold - new threshold for classification
        """
        
        self._threshold = new_threshold
    
    def register(self, photo: tf.Tensor) -> UserId:
        """
        Register a digit with the specified photo after distorting it
        
        Args:
            photo - digit photo
            
        Returns:
            User ID assigned to the specified photo
        """
        
        self._last_key = self._last_key + 1
        distorted_img = self._generator(np.expand_dims(photo, axis=0))
        self._storage[self._last_key] = self._embedding(distorted_img)
        return self._last_key
        
    @staticmethod
    def _distance(x: tf.Tensor, y: tf.Tensor) -> float:
        """
        Finds distance between two embeddings
        
        Parameters:
            x - first embedding
            y - second embedding
        """
        
        return tf.square(tf.norm(y - x))
    
    def login(self, photo: tf.Tensor) -> UserId:
        """
        Tries to login into the system having a photo
        
        Parameters:
            photo - photo to login with
        """
        
        embedding_input = self._embedding(tf.expand_dims(photo, axis=0))
        return self.login_via_embedding(embedding_input)
    
    def login_via_embedding(self, photo_embedding: tf.Tensor) -> UserId:
        """
        Tries to login into the system having an embedding. 
        Note that this method is used for faster execution when evaluating
        the system. In the real world, we would not have an embedding, so
        we would have to use login method instead.
        
        Parameters:
            photo_embedding - embedding of the photo to login with
        """
        
        if np.shape(photo_embedding)[0] != 1:
            photo_embedding = tf.expand_dims(photo_embedding, axis=0)
        
        for user_id, storage_embedding in self._storage.items():
            distance = self._distance(photo_embedding, storage_embedding)
            if distance < self._threshold:
                return user_id
            
        return AuthenticationSystem.LOGIN_FAILED

class DebugAuthenticationSystem:
    """
    Mock of a face recognition system
    """
    
    LOGIN_FAILED: UserId = -1
    
    def __init__(self, generator: tf.keras.Model, embedding: tf.keras.Model, threshold: float) -> None:
        """
        Creates a simple implementation of an authentication system
        
        Args:
            generator - model of the generator
            threshold - threshold above which two images are considered to be different
        """
        self._threshold = threshold
        self._generator = generator
        self._embedding = embedding
        
        # Variables for storage
        self._storage: Dict[UserId, tf.Tensor] = {}
        self._last_key: UserId = 0
     
    def set_threshold(self, new_threshold: float) -> None:
        self._threshold = new_threshold

    def register(self, photo: tf.Tensor) -> UserId:
        """
        Register a digit with the specified photo after distorting it
        
        Args:
            photo - digit photo
            
        Returns:
            User ID assigned to the specified photo
        """
        self._last_key = self._last_key + 1
        distorted_img = self._generator(np.expand_dims(photo, axis=0))
        self._storage[self._last_key] = self._embedding(distorted_img)
        return self._last_key
    
    def register_directly(self, photo: tf.Tensor) -> UserId:
        """
        Register a digit with the specified photo without distorting it
        
        Args:
            photo - digit photo
            
        Returns:
            User ID assigned to the specified photo
        """
        self._last_key = self._last_key + 1
        self._storage[self._last_key] = self._embedding(np.expand_dims(photo, axis=0))
        return self._last_key
        
    @staticmethod
    def _distance(x: tf.Tensor, y: tf.Tensor) -> float:
        """
        Finds distance between two embeddings
        """
        return tf.square(tf.norm(y - x))
    
    def login(self, embedding: tf.Tensor) -> UserId:
        if np.shape(embedding)[0] != 1:
            embedding = tf.expand_dims(embedding, axis=0)
        
        for user_id, storage_embedding in self._storage.items():
            distance = self._distance(embedding, storage_embedding)
            if distance < self._threshold:
                return user_id
            
        return DebugAuthenticationSystem.LOGIN_FAILED