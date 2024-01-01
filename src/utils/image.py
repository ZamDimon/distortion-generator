import numpy as np

MAX_PIXEL_INTENSITY = 255.0

def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    Preprocesses an image to be used as input for the model.
    
    Args:
        image: The image to be preprocessed.
    
    Returns:
        The preprocessed image.
    """
    
    return images / MAX_PIXEL_INTENSITY