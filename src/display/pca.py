"""
Package responsible for PCA analysis
"""

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
    
    def __init__(self, X, y) -> None:
        """
        Creates an instance of PCAPlotter. 
        
        Args:
            - X (np.ndarray): The embedding vectors.
            - y (np.ndarray): The labels.
        """
        
        # Saving data        
        self._X = X
        self._y = y
        self._y_unique = np.unique(y)
        
        X_batches, color_indeces = [], []
        for i, y_unique in enumerate(self._y_unique):
            X_batch = [X for X, y in zip(X, y) if y_unique == y]
            X_batches.append(X_batch)
            color_indeces.append([i] * len(X_batch))
        
        self._init_cmap(len(self._y_unique))
        self._X_batches = X_batches
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
        
        return self._cmap[index]
    
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
        batch_scaled = sklearn.preprocessing.StandardScaler().fit_transform(self._X)
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
                         c=self._get_color(i), label=f'Number {i}')
            
            if i == len(self._X_batches) - 1: break
            
            _from = _to
            _to = _to + len(self._X_batches[i+1])

        ax.legend()
        if save_path is not None:
            pyplot.savefig(save_path)
            
        pyplot.show()
