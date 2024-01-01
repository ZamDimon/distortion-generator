"""
Package responsible for training the embedding models
"""

import logging
from pathlib import Path

from src.embeddings.models.interface import EmbeddingModel
from src.embeddings.models.siamese import TripletNetwork
from src.embeddings.hyperparameters import Hyperparameters

from src.datasets.interface import DatasetLoader
from src.display.history import Historian
from src.display.pca import PCAPlotter

class EmbeddingTrainer:
    """
    Embedding models training
    """
    
    def __init__(self, 
                 logger: logging.Logger,
                 embedding_model: EmbeddingModel,
                 dataset_loader: DatasetLoader,
                 hyperparams: Hyperparameters) -> None:
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
              model_save_path: Path = Path('./models'),
              history_save_path: Path = Path('./images/embedding'),
              ) -> None:
        """
        Trains the embedding model.
        
        Arguments:
            - model_save_path (Path) - path to the model file. If None, the model will not be saved. Defaults to None
            - history_save_path (Path) - path to the history file. If None, the history will not be saved. Defaults to None
        """
        
        triplet_network = TripletNetwork(self._embedding_model.raw(), self._hyperparams)
        self._logger.info('Training the triplet network...')
        triplet_network.train(self._dataset_loader)
        
        self._logger.info('Training is successful. Saving the model...')
        if model_save_path is not None:
            self._embedding_model.save(self._hyperparams.model_save_path)        
        
        self._logger.info('Successfully saved the model. Displaying the model history...')
        history = triplet_network.get_history()
        historian = Historian(history)
        historian.display('Embedding model training', 
                          keys_legend=['Training loss', 'Validation loss'], 
                          save_path=history_save_path)
        
        self._logger.info('Successfully displayed the model history. Displaying PCA plot...')
        _, (X_test, y_test) = self._dataset_loader.get()
        pca_plotter = PCAPlotter(X_test, y_test)
        pca_plotter.plot(save_path=self._hyperparams.pca_plot_save_path)
        
        