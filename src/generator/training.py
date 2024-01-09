"""
Package responsible for training the embedding models
"""

import logging
from pathlib import Path

from src.embeddings.models.interface import EmbeddingModel

from src.generator.models.interface import GeneratorModel
from src.generator.models.trainer import TrainerNetwork, ImageLossType
from src.generator.hyperparameters import GeneratorHyperparameters
from src.evaluation.generator import GeneratorEvaluator

from src.datasets.interface import DatasetLoader
from src.display.history import Historian

class GeneratorTrainer:
    """
    Generator models training
    """
    
    def __init__(self, 
                 logger: logging.Logger,
                 generator_model: GeneratorModel,
                 embedding_model: EmbeddingModel,
                 dataset_loader: DatasetLoader,
                 hyperparams: GeneratorHyperparameters) -> None:
        """
        Initializes the generator trainer.
        
        Arguments:
            - logger (logging.Logger) - logger to print info with
            - generator_model (GeneratorModel) - generator model to train
            - embedding_model (EmbeddingModel) - embedding model to use for training
            - dataset_loader (DatasetLoader) - dataset loader
            - hyperparams (Hyperparameters) - hyperparameters for the model
        """
        
        self._logger = logger
        self._generator_model = generator_model
        self._embedding_model = embedding_model
        self._dataset_loader = dataset_loader
        self._hyperparams = hyperparams
        
    def train(self, 
              model_save_path: Path = Path('./models/generator'),
              history_save_path: Path = Path('./images/embedding'),
              image_save_base_path: Path = Path('./images/generator'),
              grayscale: bool = False
            ) -> None:
        """
        Trains the generator model.
        
        Arguments:
            - model_save_path (Path) - path to the model file. If None, the model will not be saved. Defaults to None
            - history_save_path (Path) - path to the history file. If None, the history will not be saved. Defaults to None
            - image_save_base_path (Path) - path to the directory where to save images. If None, the images will not be saved. Defaults to None
        """
        
        trainer_network = TrainerNetwork(
            hyperparams=self._hyperparams,
            generator=self._generator_model,
            embedding_model=self._embedding_model,
        )
        
        self._logger.info('Launching the trainer network...')
        trainer_network.train(
            self._dataset_loader, 
            image_loss=ImageLossType.MSE)
        self._logger.info('Training is successful.')
        
        if model_save_path is not None:
            self._logger.info('Saving the model...')
            self._generator_model.save(model_save_path)
            self._logger.info('Successfully saved the model')        
        
        if history_save_path is not None:
            self._logger.info('Displaying the model history...')
            history = trainer_network.get_history()
            historian = Historian(history)
            historian.display('Generator model training', 
                keys_to_display=history.history.keys(),
                keys_legend=['Train full', 'Test full', 'Train generator', 'Test generator', 'Train embedding', 'Test embedding'], 
                save_path=history_save_path)
            self._logger.info('Successfully displayed the model history.')
            
        if image_save_base_path is not None:
            self._logger.info('Saving example images...')
            evaluator = GeneratorEvaluator(
                generator_model=self._generator_model,
                embedding_model=self._embedding_model,
                dataset_loader=self._dataset_loader,
                hyperparams=self._hyperparams
            )
            evaluator.save_example_images(image_save_base_path, grayscale=grayscale)
            self._logger.info('Successfully saved example images.')
       