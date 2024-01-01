"""
Package responsible for displaying the training history
"""

from typing import List
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot

class Historian:
    """
    Object for storing and displaying the training history.
    """
    
    def __init__(self, history: tf.keras.callbacks.History) -> None:
        """
        Initializes the historian.
        
        Arguments:
            - history (tf.keras.callbacks.History) - history of the model
        """
        
        self._history = history
        
    def display(self, 
                title='Training',
                keys_to_display: List[str] = ['loss', 'val_loss'],
                keys_legend: List[str] = None,
                save_path: Path = None) -> None:
        """
        Displays and optionally saves the training history.
        
        Arguments:
            - title (str, optional) - title of the plot. Defaults to 'Training'.
            - keys_to_display (List[str], optional) - keys to plot from the history. Defaults to ['loss', 'val_loss'].
            - keys_legend (List[str], optional) - keys for a legend. If None, uses keys_to_display.
            - save_path (Path, optional) - path to save the history. Defaults to None.
        """
        
        mpl.rcParams['figure.dpi'] = 300 # For high resolution
        
        # If keys_legend is not provided, use keys_to_display
        if keys_legend is None:
            keys_legend = keys_to_display
            
        assert len(keys_to_display) == len(keys_legend), 'The number of keys to display and keys for legend must be the same'
        
        # Summarize history for loss in a plot
        pyplot.style.use('default')
        pyplot.grid()
        
        # Plotting each curve        
        cmap = pyplot.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(keys_to_display))]

        for key, color in zip(keys_to_display, colors):
            pyplot.plot(self._history.history[key], linewidth=2.5, color=color)

        pyplot.title(title)
        pyplot.ylabel('Loss')
        pyplot.xlabel('Epoch')
        pyplot.legend(keys_legend, loc='upper right')
        
        if save_path is not None:
            pyplot.savefig(save_path)
        pyplot.show()
        