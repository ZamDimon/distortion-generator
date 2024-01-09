"""
File responsible for loading the LFW dataset.
"""

import os, random
from typing import List, Tuple, Any
from pathlib import Path

from rich.progress import Progress

import numpy as np
from PIL import Image
from matplotlib import pyplot

from src.datasets.interface import DatasetLoader, Dataset
from src.utils.image import preprocess_images

class LFWLoader(DatasetLoader):
    """
    LFW dataset loader.
    """
    
    # All images in the LFW dataset have the same shape of (160, 160, 3)
    IMAGE_SHAPE = (160, 160, 3)
    # The portion of the dataset to be used for testing
    TESTING_PORTION = 0.2
    
    def __init__(self, base_dataset_path: Path, portion: float = 1.0) -> None:
        """ Initializes an LFW loader

        Args:
            - base_dataset_path (Path): Path to the base dataset folder
            - portion (float, optional): The portion of the dataset to be used. Defaults to 1.0.
        """
        
        # Load the LFW dataset
        X, y = LFWLoader._load_from_folders(base_dataset_path, portion=portion)
        
        # Shuffle the dataset
        joined = list(zip(X, y))
        random.shuffle(joined)
        X, y = zip(*joined)
        
        # Split the dataset into training and testing
        training_portion = int(len(joined) * (1.0 - LFWLoader.TESTING_PORTION))
        self._X_train = np.array(X[:training_portion])
        self._y_train = np.array(y[:training_portion])
        self._X_test = np.array(X[training_portion:])
        self._y_test = np.array(y[training_portion:])
    
    @staticmethod
    def _load_from_folders(path: Path, portion: float = 1.0) -> Tuple[np.ndarray, List[Any]]:
        """
        Loads images from a folder by iterating through all subfolders.
        Subfolders must contain images ONLY. Otherwise, an error will be raised.
        
        Args:
            path (Path): Path to the folder containing subfolders with images
            portion (float, optional): The portion of the dataset to be used. Defaults to 1.0.
        
        Returns:
            Tuple[np.ndarray, List[Any]]: A tuple of images and labels
        """
        
        fs = [f for f in os.scandir(path) if f.is_dir()]
        n = int(len(fs) * portion)
        
        subfolders = [f.path for f in fs][:n]
        labels = [f.name for f in fs][:n]
        
        # Displaying a progress bar
        images: List[np.ndarray] = []
        with Progress() as progress:
            task = progress.add_task("[green]Loading images...", total=len(subfolders))
            for folder in subfolders:
                progress.advance(task, advance=1)
                images.append(LFWLoader._load_images_from_folder(folder))
        
        # Flatenning the images
        flatenned_labels = [[label]*len(images[i]) for i, label in enumerate(labels)]
        flatenned_labels = [item for sublist in flatenned_labels for item in sublist]
        flatenned_images = [item for sublist in images for item in sublist]
        return np.array(flatenned_images), flatenned_labels
    
    @staticmethod
    def _load_images_from_folder(folder: Path) -> np.ndarray:
        """
        Loads images from a folder by iterating through all subfiles.
        Folder must contain images ONLY. Otherwise, an error will be raised.
        
        Args:
            folder (Path): Path to the folder containing images
            
        Returns:
            np.ndarray: Array of images in the numpy format
        """
        
        files = os.listdir(folder)
        images = np.empty((len(files), *LFWLoader.IMAGE_SHAPE))
        
        for i, file in enumerate(files):
            images[i,:] = (Image
                           .open(os.path.join(folder, file))
                           .resize(LFWLoader.IMAGE_SHAPE[:2]))
            images[i,:] = preprocess_images(images[i,:])
        
        return images
    
    def get(self) -> Dataset:
        return (self._X_train, self._y_train), (self._X_test, self._y_test) 
    
    def show_examples(self) -> Dataset:
        """
        Shows 3 example images of the dataset displayed on a 3x1 grid.
        """
        
        pyplot.style.use('ggplot')

        _, axs = pyplot.subplots(nrows=1, ncols=3, figsize=(9,3))
        for i, ax in enumerate(axs.flatten()):
            pyplot.sca(ax)
            pyplot.imshow(self._X_train[i])
            pyplot.title(f'Person: {self._y_train[i]}')

        pyplot.suptitle('Example images from the lfw dataset')
        pyplot.show()