"""
Module responsible for loading the datasets.
"""

from keras.datasets import mnist
from matplotlib import pyplot

from src.datasets.interface import DatasetLoader, Dataset
from src.utils.image import preprocess_images

class MNISTLoader(DatasetLoader):
    """
    MNIST dataset loader.
    """
    
    def __init__(self) -> None:
        # Load the MNIST dataset and normalize it
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self._X_train = preprocess_images(X_train)
        self._y_train = y_train
        self._X_test = preprocess_images(X_test)
        self._y_test = y_test
    
    def get(self) -> Dataset:
        return (self._X_train, self._y_train), (self._X_test, self._y_test) 
    
    def show_examples(self) -> Dataset:
        """
        Shows 9 example images of the dataset displayed on a 3x1 grid.
        """
        
        pyplot.style.use('ggplot')

        _, axs = pyplot.subplots(nrows=1, ncols=3, figsize=(9,3))
        for i, ax in enumerate(axs.flatten()):
            pyplot.sca(ax)
            pyplot.imshow(self._X_train[i], cmap=pyplot.get_cmap('gray'))
            pyplot.title(f'Number {self._y_train[i]}')

        pyplot.suptitle('Example images from the MNIST dataset')
        pyplot.show()