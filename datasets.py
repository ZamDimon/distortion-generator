"""
Package responsible for logging configuration.
"""

from pathlib import Path
from enum import IntEnum

from src.datasets.interface import DatasetLoader
from src.datasets.mnist import MNISTLoader
from src.datasets.lfw_cropped import LFWCroppedLoader

from src.embeddings.models.interface import EmbeddingModel
from src.embeddings.models.mnist import MNISTEmbeddingModel
from src.embeddings.models.facenet import FaceNetEmbeddingModel
from src.embeddings.hyperparameters import EmbeddingHyperparameters

from src.generator.models.interface import GeneratorModel
from src.generator.models.mnist import MNISTGeneratorModel
from src.generator.models.lfw import LFWGeneratorModel

# For now, unused
_LFW_FUNNELED_BASE_PATH = Path('./dataset/lfw_funneled')

# Base path for the cropped LFW dataset
LFW_CROPPED_BASE_PATH = Path('./dataset/lfwcrop_color/faces')

class DatasetHandler(IntEnum):
    """
    Enum for logging verbosity
    """
    
    MNIST = 0
    LFW = 1
    
    def as_str(self) -> str:
        """
        Returns the enum as a string
        """
        
        match self.value:
            case 0:
                return 'mnist'
            case 1:
                return 'lfw'
    
    def dataset_loader(self, grayscale: bool = False) -> DatasetLoader:
        """
        Returns the dataset loader for the enum
        
        Parameters:
            - grayscale (bool, optional) - whether the images should be loaded in grayscale (if needed). Defaults to False
        """
        
        match self.value:
            case 0:
                return MNISTLoader(expand_dims=True)
            case 1:
                return LFWCroppedLoader(LFW_CROPPED_BASE_PATH, 
                                        portion=1.0, 
                                        grayscale=grayscale)
    
    def empty_embedding_model(self, hyperparams_path: Path = None) -> EmbeddingModel:
        """
        Returns the embedding model architecture for the enum
        
        Parameters:
            - hyperparams_path (Path) - path to the hyperparameters file if needed. 
            If None, the default hyperparameters will be used. Defaults to None
        """
        
        match self.value:
            case 0:
                return MNISTEmbeddingModel(EmbeddingHyperparameters(hyperparams_path))
            case 1:
                return FaceNetEmbeddingModel()
        
    def pretrained_embedding_model(self, model_path: Path = None, trainable: bool = False) -> EmbeddingModel:
        """
        Returns the embedding model architecture for the enum
        
        Parameters:
            - model_path (Path, optional) - path to the model file if needed. Defaults to None.
            - trainable (bool, optional) - whether the model should be trainable. Defaults to False
        """
        
        match self.value:
            case 0:
                return MNISTEmbeddingModel.from_path(model_path, trainable=trainable)
            case 1:
                return FaceNetEmbeddingModel(trainable=trainable)
        
    def generator_model(self, grayscale: bool = False) -> GeneratorModel:
        """
        Returns the generator model architecture for the enum
        
        Parameters:
            - grayscale (bool, optional) - whether the model should be grayscale. Defaults to False
        """
        
        match self.value:
            case 0:
                return MNISTGeneratorModel()
            case 1:
                return LFWGeneratorModel(grayscale=grayscale)

    def pretrained_generator_model(self, 
                                   model_path: Path, 
                                   trainable: bool = False) -> GeneratorModel:
        """
        Returns the generator model architecture for the enum
        
        Parameters:
            - grayscale (bool, optional) - whether the model should be grayscale. Defaults to False
        """
        
        match self.value:
            case 0:
                return MNISTGeneratorModel.from_path(model_path, trainable=trainable)
            case 1:
                return LFWGeneratorModel.from_path(model_path, trainable=trainable)
    