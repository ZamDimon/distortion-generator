"""
Class for evaluating accuracy of a generator model.
"""

# General-purpose imports
import os, random
from pathlib import Path
from typing import List, Tuple, Dict, Any

# For printing
from rich.console import Console
from rich.table import Table

# ML imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

# Internal imports
from src.embeddings.models.interface import EmbeddingModel
from src.generator.hyperparameters import GeneratorHyperparameters
from src.generator.models.interface import GeneratorModel
from src.datasets.interface import DatasetLoader
from src.evaluation.pair_picker import PairPicker, PairValidationGenerator
from src.evaluation.authentication_system import DebugAuthenticationSystem
from src.evaluation.classification import get_statistics
from src.display.roc import ROCPlotter

class GeneratorEvaluator:
    """
    Class for evaluating the generator model.
    """
    
    RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    MAX_LABELS_TO_ANALYZE = 10
    IMAGES_TO_ANALYZE = 1000
    
    def __init__(self, 
                 generator_model: GeneratorModel,
                 embedding_model: EmbeddingModel,
                 dataset_loader: DatasetLoader,
                 hyperparams: GeneratorHyperparameters = None) -> None:  
        """
        Initializes the generator evaluation.
        
        Arguments:
            - generator_model (GeneratorModel) - generator model to evaluate
            - embedding_model (EmbeddingModel) - embedding model to evaluate
            - dataset_loader (DatasetLoader) - dataset loader
            - hyperparams (Hyperparameters, optional) - hyperparameters for the model if needed. Defaults to None
        """
        
        self._generator_model = generator_model
        self._embedding_model = embedding_model
        self._hyperparams = hyperparams
        
        (X_test, y_test), _ = dataset_loader.get()
        self._X_test = X_test
        self._y_test = y_test
    
    def save_example_images(self, 
                             base_path: Path, 
                             images_to_save: int = 10,
                             image_shape: Tuple[int, int] = (256,256),
                             labels: List[Any] = None,
                             grayscale: bool = True) -> None:
        """
        Saves example images.
        
        Arguments:
            - base_path (Path) - path to the directory where to save images
            - images_to_save (int) - number of images to save. Defaults to 10
            - image_shape (Tuple[int, int]) - shape of the image to save. Defaults to (256,256)
            - labels (List[Any]) - list of labels to save. If None, all labels will be saved. Defaults to None
            - grayscale (bool) - whether to save images in grayscale. Defaults to True
        """
        
        assert len(image_shape) == 2, 'Image shape must be a tuple of 2 elements'
        
        y_uniques = np.unique(self._y_test)
        labels_to_save = y_uniques if labels is None else labels
        
        for label in labels_to_save:
            # Making sure we have a place to save images in
            directory_path = base_path / str(label)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            
            # Preparing a set of images to save
            X = self._X_test[self._y_test == label]
            X = X[:images_to_save]
            X_generated = self._generator_model.raw.predict(np.array(X), verbose=0)
     
            shape_to_resize = [image_shape[0], image_shape[1]]
            X = tf.image.resize(X, shape_to_resize, method=GeneratorEvaluator.RESIZE_METHOD)
            X_generated = tf.image.resize(X_generated, shape_to_resize, method=GeneratorEvaluator.RESIZE_METHOD)
            
            # Saving images
            for i, (x, x_generated) in enumerate(zip(X, X_generated)):
                real_img_name, generated_img_name = f'{i}_real.png', f'{i}_generated.png'
                x = x.numpy()
                x_generated = x_generated.numpy()
                
                if grayscale:
                    pyplot.imsave(directory_path / real_img_name, np.squeeze(x, axis=-1), cmap='gray')
                    pyplot.imsave(directory_path / generated_img_name, np.squeeze(x_generated, axis=-1), cmap='gray')
                    continue
                
                pyplot.imsave(directory_path / real_img_name, x)
                pyplot.imsave(directory_path / generated_img_name, x_generated)   
                
                
    @staticmethod
    def _mse_img_distance(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Distance function for image comparison
        """
        
        return tf.reduce_mean(tf.math.square(y_true - y_pred))

    def evaluate_image_distances(self) -> None:
        """
        Image distance evaluation. Calculates Euclidean distances between real and generated images,
        real and real images, and generated and generated images. Then, prints the results using rich library.
        """
        
        X_generated = self._generator_model.raw.predict(self._X_test, verbose=1)
        
        labels = np.unique(self._y_test)
        labels = sorted(labels, key=lambda x: len(self._X_test[self._y_test == x]), reverse=True)
        labels = labels[:GeneratorEvaluator.MAX_LABELS_TO_ANALYZE]
        
        X_batches_real: Dict[Any, tf.Tensor] = {}
        X_batches_generated: Dict[Any, tf.Tensor] = {}
        for label in labels:
            batch_real = np.array([self._X_test[i] for i in range(len(self._y_test)) if self._y_test[i] == label])
            batch_generated = np.array([X_generated[i] for i in range(len(self._y_test)) if self._y_test[i] == label])
            
            X_batches_real[label] = batch_real
            X_batches_generated[label] = batch_generated
        
        # Calculating Euclidean distances between real and generated images
        real_generated_distances = [self._mse_img_distance(X_batches_real[label], X_batches_generated[label]) for label in labels]
        
        pair_picker_real = PairPicker(self._X_test, self._y_test)
        pair_picker_generated = PairPicker(X_generated, self._y_test)
        
        real_real_distances, generated_generated_distance = [], []
        
        # Calculating Euclidean distances between real and real images
        for label in labels:
            avg_distance = np.mean([self._mse_img_distance(*pair_picker_real.pick(label)) for _ in range(GeneratorEvaluator.IMAGES_TO_ANALYZE)])
            real_real_distances.append(avg_distance)
            
        # Calculating Euclidean distances between generated and generated images
        for digit in labels:
            avg_distance = np.mean([self._mse_img_distance(*pair_picker_generated.pick(digit)) for _ in range(GeneratorEvaluator.IMAGES_TO_ANALYZE)])
            generated_generated_distance.append(avg_distance)
            
        # Printing results
        table = Table(title='Image distances')
        table.add_column('Label', justify='center', style='green', no_wrap=True)
        table.add_column('Real - Generated', justify='center', style='green', no_wrap=True)
        table.add_column('Real - Real', justify='center', style='green', no_wrap=True)
        table.add_column('Generated - Generated', justify='center', style='green', no_wrap=True)
        
        for label, real_generated, real_real, generated_generated in zip(labels, real_generated_distances, real_real_distances, generated_generated_distance):
            table.add_row(str(label), f'{real_generated:.3f}', f'{real_real:.3f}', f'{generated_generated:.3f}')
        
        console = Console()
        console.print(table)
    
    def build_roc(self, roc_save_path: Path, classes_to_test: int = 3) -> None:
        """
        This function builds the ROC curve based on the testing 
        authentication system
        
        Parameters:
            - classes_to_test (int, optional) - a number of classes to test with
        """
        
        # Creating a pair generator
        X_embeddings_real = self._embedding_model.raw.predict(self._X_test, verbose=1)
        pairs_generator = PairValidationGenerator(X_embeddings_real, self._y_test, classes_number=classes_to_test)
        
        positive_classes, _ = pairs_generator.get_test_classes()
        
        # Registering the real recognition system
        authentication_system_real = DebugAuthenticationSystem(self._generator_model.raw, self._embedding_model.raw, 0.25)
        for positive_class in positive_classes:
            login_photo = random.choice(self._X_test[self._y_test == positive_class])
            authentication_system_real.register_directly(login_photo)
        
        # Defining a statistics
        statistics_real = get_statistics(authentication_system_real, pairs_generator, pairs_number=1200, threshold_split=100, threshold_to=2.5)
        
        # Registering the recognition system using generated instances
        authentication_system_generated = DebugAuthenticationSystem(self._generator_model.raw, self._embedding_model.raw, 0.25)
        for positive_class in positive_classes:
            login_photo = random.choice(self._X_test[self._y_test == positive_class])
            authentication_system_generated.register_directly(login_photo)
        
        # Defining a statistics
        statistics_generated = get_statistics(authentication_system_generated, pairs_generator, pairs_number=1200, threshold_split=100, threshold_to=2.5)
        plotter = ROCPlotter()
        plotter.plot_from_statistics(
            statistics_generated=statistics_generated, 
            statistics_real=statistics_real, 
            save_path=roc_save_path)
        
        best_result_real = statistics_real.pick_best_f1()
        best_result_real.draw_summary()

        print(f'TP={best_result_real._true_positive}')
        print(f'TN={best_result_real._true_negative}')
        print(f'FP={best_result_real._false_positive}')
        print(f'FN={best_result_real._false_negative}')
        
        best_result_generated = statistics_generated.pick_best_f1()
        best_result_generated.draw_summary()

        print(f'TP={best_result_generated._true_positive}')
        print(f'TN={best_result_generated._true_negative}')
        print(f'FP={best_result_generated._false_positive}')
        print(f'FN={best_result_generated._false_negative}')
        