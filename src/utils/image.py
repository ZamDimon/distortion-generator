import numpy as np

MAX_PIXEL_INTENSITY = 255.0

def preprocess_images(images: np.ndarray, expand_dims=False) -> np.ndarray:
    """
    Preprocesses an image to be used as input for the model.
    
    Args:
        image (np.ndarray): The image to be preprocessed.
        expand_dims (bool, optional): Whether to expand the dimensions of the image. Defaults to True.
    
    Returns:
        The preprocessed image.
    """
    if expand_dims:
        images = np.expand_dims(images, axis=-1)
        
    return images / MAX_PIXEL_INTENSITY