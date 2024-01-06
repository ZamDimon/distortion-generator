"""
Package responsible for PCA analysis
"""

from typing import List
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot

import numpy as np
import sklearn
from sklearn.decomposition import PCA

class PCAPlotter:
    """
    Class responsible for plotting PCA analysis
    """
    
    CMAP_NAME = 'rainbow'
    PYPLOT_THEME = 'default'
    FIG_SIZE = (10,8)
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 labels_to_display: np.ndarray = None,
                 colors_to_display: List[str] = None) -> None:
        """
        Creates an instance of PCAPlotter. 
        
        Args:
            - X (np.ndarray): The embedding vectors.
            - y (np.ndarray): The labels.
            - labels_to_display (np.ndarray): The labels to display. If None, all labels will be displayed. Defaults to None
            - colors_to_display (List[str]): The colors to display. If None, the default colors will be used. Defaults to None
        """
        
        assert len(X) == len(y), 'The number of images and labels must be the same'
        
        # Saving data        
        self._X = X
        self._y = y
        self._colors_to_display = colors_to_display
        self._labels_to_display = np.unique(y) if labels_to_display is None else labels_to_display
        
        X_batches, color_indeces = [], []
        for i, label_to_display in enumerate(self._labels_to_display):
            X_batch = [x for x, label in zip(X, y) if label == label_to_display]
            X_batches.append(X_batch)
            color_indeces.append([i] * len(X_batch))
        
        self._init_cmap(len(self._labels_to_display))
        self._X_batches = X_batches
        self._X_flatenned = [item for batch in X_batches for item in batch]
        self._color_indeces = color_indeces
    
    def _init_cmap(self, number: int) -> None:
        """
        Returns a list of colors for each label.
        """
        
        self._cmap = pyplot.cm.rainbow(np.linspace(0, 1, number))
        
    def _get_color(self, index: int) -> any:
        """
        Returns a color for a given index.
        
        Args:
            - index (int): The index of the color.
        """
        if self._colors_to_display is None:
            return self._cmap[index]    
        
        return self._colors_to_display[index]
    
    def plot(self, save_path: Path = None) -> None:
        """
        Plots and saves the PCA analysis plot.
        
        Arguments:
            - save_path (Path) - path to save the plot
        """

        mpl.rcParams['figure.dpi'] = 300 # For high resolution
        
        # Launching PCA
        pca = PCA(n_components=3)
        pca = sklearn.decomposition.PCA(n_components=3)
        batch_scaled = sklearn.preprocessing.StandardScaler().fit_transform(self._X_flatenned)
        pca_features = pca.fit_transform(batch_scaled)

        # Getting scaled features
        x_data = pca_features[:,0]
        y_data = pca_features[:,1]
        z_data = pca_features[:,2]
        
        # Plot 3D plot
        pyplot.style.use(PCAPlotter.PYPLOT_THEME)
        _ = pyplot.figure(figsize=PCAPlotter.FIG_SIZE)
        ax = pyplot.axes(projection='3d')

        _from, _to = 0, len(self._X_batches[0])
        for i in range(len(self._X_batches)):
            ax.scatter3D(x_data[_from:_to], y_data[_from:_to], z_data[_from:_to], 
                         c=self._get_color(i), label=f'{self._labels_to_display[i]}')
            
            if i == len(self._X_batches) - 1: break
            
            _from = _to
            _to = _to + len(self._X_batches[i+1])

        ax.legend()
        if save_path is not None:
            pyplot.savefig(save_path)
            
        pyplot.show()
