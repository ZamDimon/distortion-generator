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

class LFWCroppedLoader(DatasetLoader):
    """
    LFW dataset loader.
    """
    
    # All images in the LFW dataset have the same shape of (160, 160, 3)
    IMAGE_SHAPE = (160, 160, 3)
    # The portion of the dataset to be used for testing
    TESTING_PORTION = 0.2
    
    def __init__(self, base_dataset_path: Path, portion: float = 1.0, grayscale: bool = False) -> None:
        """ Initializes an LFW cropped loader

        Args:
            - base_dataset_path (Path): Path to the base dataset folder
            - portion (float, optional): The portion of the dataset to be used. Defaults to 1.0.
            - grayscale (bool, optional): Whether to load the images in grayscale. Defaults to False.
        """
        
        # Load the LFW dataset
        X, y = LFWCroppedLoader._load_images_from_folder(base_dataset_path, portion=portion, grayscale=grayscale)
        
        # Shuffle the dataset
        joined = list(zip(X, y))
        random.shuffle(joined)
        X, y = zip(*joined)
        
        # Split the dataset into training and testing
        training_portion = int(len(joined) * (1.0 - LFWCroppedLoader.TESTING_PORTION))
        self._X_train = np.array(X[:training_portion])
        self._y_train = np.array(y[:training_portion])
        self._X_test = np.array(X[training_portion:])
        self._y_test = np.array(y[training_portion:])
    
    @staticmethod
    def _load_images_from_folder(folder: Path, portion: float = 1.0, grayscale: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Loads images from a folder by iterating through all subfiles.
        Folder must contain images ONLY. Otherwise, an error will be raised.
        
        Args:
            folder (Path): Path to the folder containing images
            
        Returns:
            np.ndarray: Array of images in the numpy format
        """
        
        files = os.listdir(folder)
        n = int(len(files) * portion)
        files = files[:n]
        
        labels = [f"{file.split('_')[0]}_{file.split('_')[1]}" for file in files]
        images = np.empty((len(files), *LFWCroppedLoader.IMAGE_SHAPE))
        
        with Progress() as progress:
            task = progress.add_task("[blue]Loading images...", total=len(files))
            for i, file in enumerate(files):
                img = (Image
                            .open(os.path.join(folder, file))
                            .resize(LFWCroppedLoader.IMAGE_SHAPE[:2]))
                if grayscale:
                    img = img.convert('L')
                    img = np.expand_dims(img, axis=-1)
                    img = np.repeat(img, 3, axis=-1)
                    
                img = np.array(img)
                img = preprocess_images(img)
                images[i,:] = img
                progress.update(task, advance=1)
                
        return images, labels
    
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