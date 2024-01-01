"""
Package responsible for the hyperparameters of the embedding model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

class Hyperparameters:
    """
    Class containing the hyperparameters for the embedding model.
    """
    
    _PARAMS_LIST: List[str] = [
        'meta',
        'input_shape',
        'hidden_layer_size',
        'batch_size',
        'epochs',
        'alpha',
        'regulizer',
        'theta',
        'epsilon',
        'learning_rate',
        'embedding_size',
    ]
    
    def __init__(self, json_path: Path) -> None:
        """
        Initializes the hyperparameters and asserts that the file is properly configures.
        
        Args:
            json_path (Path): Path to the JSON file containing the hyperparameters.
        """
        
        with open(str(json_path), 'r') as json_file:
            self._dictionary = json.loads(json_file.read())
            metadata = Metadata(self._dictionary['meta']) if 'meta' in self._dictionary else Metadata.default()
            self._dictionary['meta'] = metadata
    
    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Hyperparameters:
        hyperparams = cls.__new__(cls)
        super(Hyperparameters, hyperparams).__init__()
        cls._dictionary = dictionary
        return cls
            
    def raw(self) -> Dict[str, Any]:
        """
        Returns the raw dictionary.
        """
        
        return self._dictionary
    
    def __getattr__(self, name: str) -> Any:
        """
        Returns the value of the attribute.
        
        Args:
            name (str): The name of the attribute.
        """
        
        if name not in Hyperparameters._PARAMS_LIST:
            raise AttributeError(f'Attribute {name} not found')
        
        if name in self._dictionary:
            return self._dictionary[name]
        
        default_hyperparams = Hyperparameters.default()
        return default_hyperparams._dictionary[name]
    
    @staticmethod
    def default() -> Hyperparameters:
        """
        Returns the default hyperparameters.
        """
        
        return Hyperparameters.from_dictionary({
            'meta': Metadata.default(),
            'input_shape': (28,28),
            'hidden_layer_size': 2048,
            'batch_size': 1028,
            'val_batch_size': 128,
            'epochs': 100,
            'alpha': 0.01,
            'regulizer': 1e-3,
            'theta': 0.2,
            'epsilon': 1e-8,
            'learning_rate': 5e-5,
            'embedding_size': 2,
        })
        
class Metadata:
    """
    Embedding model metadata
    """
    
    _FIELDS = [
        'version',
        'subversion',
    ]
    
    def __init__(self, dictionary: Dict[str, Any]) -> None:
        """
        Initializes the metadata.
        
        Args:
            dictionary (Dict): The dictionary containing the metadata.
        """
        
        self._dictionary = dictionary
        
    def raw(self) -> Dict[str, Any]:
        """
        Returns the raw dictionary.
        """
        
        return self._dictionary
        
    def __getattr__(self, name: str) -> None:
        """
        Returns the value of the attribute.
        
        Args:
            name (str): The name of the attribute.
        """
        if name not in Metadata._FIELDS:
            raise AttributeError(f'Attribute {name} not found')
        
        return self._dictionary.get(name, Metadata.default()._dictionary[name])
    
    @staticmethod
    def default() -> Metadata:
        """
        Returns the default metadata.
        """
        
        return Metadata({
            'version': 1,
            'subversion': 1,
        })