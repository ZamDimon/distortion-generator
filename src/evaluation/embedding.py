"""
Class for evaluating accuracy of embedding model.
"""

from typing import Any
from rich.console import Console
from rich.table import Table

import numpy as np

from src.embeddings.models.interface import EmbeddingModel
from src.embeddings.hyperparameters import EmbeddingHyperparameters
from src.datasets.interface import DatasetLoader

class EmbeddingEvaluator:
    """
    Class for evaluating embedding model.
    """
    
    EXAMPLE_CLASSES_NUMBER = 4
    EXAMPLE_SAMPLES_NUMBER = 8
    EXAMPLE_FEATURES_NUMBER = 8
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 dataset_loader: DatasetLoader,
                 hyperparams: EmbeddingHyperparameters) -> None:  
        """
        Initializes the embedding evaluation.
        
        Arguments:
            - embedding_model (EmbeddingModel) - embedding model to evaluate
            - dataset_loader (DatasetLoader) - dataset loader
            - hyperparams (EmbeddingHyperparameters) - hyperparameters for the model
        """
        
        self._embedding_model = embedding_model
        self._dataset_loader = dataset_loader
        self._hyperparams = hyperparams
        
        # Creating console for printing
        self._console = Console()
    
    def print_example_predictions(self) -> None:
        """
        Prints example predictions.
        """
        
        # Taking test data
        _, (X_test, y_test) = self._dataset_loader.get()
        # Finding all different classes
        y_uniques = np.unique(y_test)
        # Taking only first EmbeddingEvaluation.EXAMPLE_CLASSES_NUMBER classes
        y_uniques = y_uniques[:EmbeddingEvaluator.EXAMPLE_CLASSES_NUMBER]
        
        # Printing table for each class
        for y_unique in y_uniques:
            X_unique = X_test[y_test == y_unique]
            X_unique = X_unique[:EmbeddingEvaluator.EXAMPLE_SAMPLES_NUMBER]
            y_predicted = self._embedding_model.raw.predict(X_unique)
            self._print_feature_vector_table(y_unique, y_predicted)

    def _print_feature_vector_table(self, y_unique: Any, y_predicted: np.ndarray) -> None:
        """
        Prints feature vector table.
        
        Arguments:
            - y_unique (Any) - class of the feature vectors
            - y_predicted (np.ndarray) - predicted values
        """
        
        table = Table(title=f"Feature vector v for class {str(y_unique)}")

        for i in range(EmbeddingEvaluator.EXAMPLE_FEATURES_NUMBER):
            table.add_column(f"v{i}", justify="center", style="green", no_wrap=True)
        
        for i in range(EmbeddingEvaluator.EXAMPLE_SAMPLES_NUMBER):
            y_predicted_str = ["{:.3f}".format(y) for y in y_predicted[i]]
            table.add_row(*y_predicted_str)

        self._console.print(table)