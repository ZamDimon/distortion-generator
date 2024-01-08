"""
Package responsible for training the embedding models
"""

import logging
from pathlib import Path

from src.embeddings.models.interface import EmbeddingModel
from src.embeddings.models.siamese import TripletNetwork
from src.embeddings.hyperparameters import EmbeddingHyperparameters

from src.datasets.interface import DatasetLoader
from src.display.history import Historian
from src.display.pca import PCAPlotter
from src.evaluation.embedding import EmbeddingEvaluator

class EmbeddingTrainer:
    """
    Embedding models training
    """
    
    def __init__(self, 
                 logger: logging.Logger,
                 embedding_model: EmbeddingModel,
                 dataset_loader: DatasetLoader,
                 hyperparams: EmbeddingHyperparameters) -> None:
        """
        Initializes the embedding trainer.
        
        Arguments:
            - embedding_model (EmbeddingModel) - embedding model to train
            - dataset_loader (DatasetLoader) - dataset loader
        """
        
        self._logger = logger
        self._embedding_model = embedding_model
        self._dataset_loader = dataset_loader
        self._hyperparams = hyperparams
        
    def train(self, 
              model_save_path: Path = Path('./models/embedding'),
              history_save_path: Path = Path('./images/embedding'),
              pca_save_path: Path = Path('./images/embedding/pca.png')
              ) -> None:
        """
        Trains the embedding model.
        
        Arguments:
            - model_save_path (Path) - path to the model file. If None, the model will not be saved. Defaults to None
            - history_save_path (Path) - path to the history file. If None, the history will not be saved. Defaults to None
        """
        
        triplet_network = TripletNetwork(self._embedding_model.raw, self._hyperparams)
        
        self._logger.info('Training the triplet network...')
        triplet_network.train(self._dataset_loader)
        self._logger.info('Training is successful.')
        
        if model_save_path is not None:
            self._logger.info('Saving the model...')
            self._embedding_model.save(model_save_path)        
            self._logger.info('Successfully saved the model.')
            
        if history_save_path is not None:
            self._logger.info('Displaying the model history...')
            history = triplet_network.get_history()
            historian = Historian(history)
            historian.display('Embedding model training', 
                            keys_legend=['Training loss', 'Validation loss'], 
                            save_path=history_save_path)
            self._logger.info('Successfully displayed the model history.')
        
        if pca_save_path is not None:
            self._logger.info('Displaying PCA plot...')
            _, (X_test, y_test) = self._dataset_loader.get()
            X_test_embedding = self._embedding_model.raw.predict(X_test)
            pca_plotter = PCAPlotter(X_test_embedding, y_test, labels_to_display=[i for i in range(5,10)])
            pca_plotter.plot(save_path=pca_save_path)
            
            self._logger.info('Successfully displayed the PCA plot.')
        
        self._logger.info('Displaying example predictions...')
        evaluator = EmbeddingEvaluator(self._embedding_model, self._dataset_loader, self._hyperparams)
        evaluator.print_example_predictions()
        