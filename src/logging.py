"""
Package responsible for logging configuration.
"""

from enum import IntEnum

import logging
from rich.logging import RichHandler
import absl.logging

class VerboseMode(IntEnum):
    """
    Enum for logging verbosity
    """
    
    WARNING = 0
    INFO = 1
    DEBUG = 2
    
    def log_level(self) -> str:
        """
        Returns the string representation of the enum
        """
        return { 0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}.get(self.value, 'INFO')


def create_logger(mode: VerboseMode=VerboseMode.INFO) -> logging.Logger:
    """
    Configures the logging, and returns the logger instance

    ### Args:
    - level (str, optional): Logging level. Defaults to 'INFO'.

    ### Returns:
        logging.Logger: Instance of the logger
    """
    
    logging.basicConfig(
        level=mode.log_level(), format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    absl.logging.set_verbosity(absl.logging.ERROR) # Disabling the TensorFlow warnings
    
    return logging.getLogger("rich")