from typing import TypeAlias, Tuple
import numpy as np

# Type aliases
Dataset: TypeAlias = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

class DatasetLoader:
    """
    Interface that dataset loaders must implement.
    """
    
    def get(self) -> Dataset:
        """
        Retrieves the dataset from the class internal variables.
        """
        pass
        
    def show_examples(self) -> Dataset:
        """
        Shows some examples of the dataset.
        """
        pass