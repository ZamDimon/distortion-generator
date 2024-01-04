"""
Package responsible for the hyperparameters of the embedding model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

class GeneratorHyperparameters:
    """
    Class containing the hyperparameters for the embedding model.
    """
    
    _PARAMS_LIST: List[str] = [
        'meta',
        'input_shape',
        'threshold',
        'pi_emb',
        'learning_rate',
        'epochs',
        'batch_size',
        'validation_split'
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
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> GeneratorHyperparameters:
        hyperparams = cls.__new__(cls)
        super(GeneratorHyperparameters, hyperparams).__init__()
        cls._dictionary = dictionary
        return cls
            
    def raw(self) -> Dict[str, Any]:
        """
        Returns the raw dictionary.
        """
        
        return self._dictionary
    
    def save(self, path: Path) -> None:
        """
        Saves the hyperparameters to the JSON file.
        
        Args:
            path (Path): The path to the JSON file.
        """
        
        with open(str(path), 'w') as json_file:
            dictionary_to_save = self._dictionary.copy()
            dictionary_to_save['meta'] = self._dictionary['meta'].raw()
            json_file.write(json.dumps(dictionary_to_save, indent=4))
    
    def __getattr__(self, name: str) -> Any:
        """
        Returns the value of the attribute.
        
        Args:
            name (str): The name of the attribute.
        """
        
        if name not in GeneratorHyperparameters._PARAMS_LIST:
            raise AttributeError(f'Attribute {name} not found')
        
        if name in self._dictionary:
            return self._dictionary[name]
        
        default_hyperparams = GeneratorHyperparameters.default()
        return default_hyperparams._dictionary[name]
    
    @staticmethod
    def default() -> GeneratorHyperparameters:
        """
        Returns the default hyperparameters.
        """
        
        return GeneratorHyperparameters.from_dictionary({
            'meta': Metadata.default().raw(),
            'threshold': 0.5,
            'lambda': 0.95,
            'learning_rate': 1e-4,
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2
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