"""
Class for evaluating accuracy of a generator model.
"""

import os
from pathlib import Path
from typing import List, Tuple, Any

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

from src.embeddings.models.interface import EmbeddingModel
from src.generator.hyperparameters import GeneratorHyperparameters
from src.generator.models.interface import GeneratorModel
from src.datasets.interface import DatasetLoader

class GeneratorEvaluator:
    """
    Class for evaluating the generator model.
    """
    
    RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    
    def __init__(self, 
                 generator_model: GeneratorModel,
                 embedding_model: EmbeddingModel,
                 dataset_loader: DatasetLoader,
                 hyperparams: GeneratorHyperparameters) -> None:  
        """
        Initializes the generator evaluation.
        
        Arguments:
            - embedding_model (EmbeddingModel) - embedding model to evaluate
            - dataset_loader (DatasetLoader) - dataset loader
            - hyperparams (Hyperparameters) - hyperparameters for the model
        """
        
        self._generator_model = generator_model
        self._embedding_model = embedding_model
        self._dataset_loader = dataset_loader
        self._hyperparams = hyperparams
    
    def save_example_images(self, 
                             base_path: Path, 
                             images_to_save: int = 10,
                             image_shape: Tuple[int, int] = (256,256),
                             labels: List[Any] = None) -> None:
        """
        Saves example images.
        
        Arguments:
            - base_path (Path) - path to the directory where to save images
            - images_to_save (int) - number of images to save. Defaults to 10
            - image_shape (Tuple[int, int]) - shape of the image to save. Defaults to (256,256)
            - labels (List[Any]) - list of labels to save. If None, all labels will be saved. Defaults to None
        """
        
        assert len(image_shape) == 2, 'Image shape must be a tuple of 2 elements'
        
        _, (X_test, y_test) = self._dataset_loader.get()
        y_uniques = np.unique(y_test)
        labels_to_save = y_uniques if labels is None else labels
        
        for label in labels_to_save:
            # Making sure we have a place to save images in
            directory_path = base_path / str(label)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            
            # Preparing a set of images to save
            X = X_test[y_test == label]
            X = X[:images_to_save]
            X_generated = self._generator_model.raw.predict(np.array(X), verbose=0)
     
            shape_to_resize = [image_shape[0],image_shape[1]]
            X = tf.image.resize(X, shape_to_resize, method=GeneratorEvaluator.RESIZE_METHOD)
            X_generated = tf.image.resize(X_generated, shape_to_resize, method=GeneratorEvaluator.RESIZE_METHOD)
            
            # Saving images
            for i, (x, x_generated) in enumerate(zip(X, X_generated)):
                real_img_name, generated_img_name = f'{i}_real.png', f'{i}_generated.png'
                pyplot.imsave(directory_path / real_img_name, np.squeeze(x, axis=-1), cmap='gray')
                pyplot.imsave(directory_path / generated_img_name, np.squeeze(x_generated, axis=-1), cmap='gray')