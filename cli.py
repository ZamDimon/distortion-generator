"""
Package for launching training and validation flows
"""

# Creating a CLI client
import typer
app = typer.Typer(help=('CLI for interacting with', 
                  'implementations of models described',
                  'in the research paper'))

import numpy as np
import random
from typing_extensions import Annotated
from pathlib import Path

from matplotlib import pyplot

from src.logging import create_logger, VerboseMode
from src.datasets.mnist import MNISTLoader
from src.display.pca import PCAPlotter

from src.utils.color import lighten_color

from src.embeddings.training import EmbeddingTrainer
from src.embeddings.hyperparameters import EmbeddingHyperparameters
from src.embeddings.models.mnist import MNISTEmbeddingModel

from src.generator.hyperparameters import GeneratorHyperparameters
from src.generator.models.mnist import MNISTGeneratorModel
from src.generator.training import GeneratorTrainer

@app.command()
def train_embedding_model(
    hyperparams_path: Annotated[Path, typer.Option(help='Path to the hyperparameters file')] = Path('./hyperparams_embedding.json'),
    model_save_path: Annotated[Path, typer.Option(help='Path to the model file')] = None,
    history_path: Annotated[Path, typer.Option(help='Path to the history file')] = None,
    pca_save_path: Annotated[Path, typer.Option(help='Path to the PCA plot file')] = None,
    verbose: Annotated[int, typer.Option(help='Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing model summary and debug')] = 1,
) -> None:
    verbose = VerboseMode(verbose)
    logger = create_logger(verbose)
    
    # Getting the dataset
    logger.info('Loading the MNIST dataset...')
    mnist_loader = MNISTLoader(expand_dims=False)
    logger.info('Successfully loaded the MNIST dataset')
    
    if verbose == VerboseMode.DEBUG:
        logger.info('Showing example images from the MNIST dataset...')
        mnist_loader.show_examples()
    
    # Getting hyperparameters
    logger.info('Loading hyperparameters for the embedding model...')
    hyperparams = EmbeddingHyperparameters(json_path=hyperparams_path)
    
    # Creating the embedding model
    logger.info('Successfully loaded the hyperparameters.')
    embedding_model = MNISTEmbeddingModel(hyperparams)
    logger.info('Using the following embedding model:')
    embedding_model.summary()
    
    logger.info('Launching the trainer...')
    trainer = EmbeddingTrainer(logger, embedding_model, mnist_loader, hyperparams)
    
    # Setting the save paths if not provided
    if model_save_path is None:
        model_save_path = Path(f'./models/embedding/v{hyperparams.meta.version}.{hyperparams.meta.subversion}')
    if history_path is None:
        history_path = Path(f'./images/embedding/v{hyperparams.meta.version}.{hyperparams.meta.subversion}/history.png')
    if pca_save_path is None:
        pca_save_path = Path(f'./images/embedding/v{hyperparams.meta.version}.{hyperparams.meta.subversion}/pca.png')

    # Creating the directories if they do not exist
    model_save_path.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pca_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Training the model
    trainer.train(
        model_save_path=model_save_path, 
        history_save_path=history_path,
        pca_save_path=pca_save_path)
    hyperparams.save(model_save_path / 'hyperparams.json')
    
    logger.info('Loader successfully finished the training process. Exiting...')

@app.command()
def train_generator_model(
    embedding_model_path: Annotated[Path, typer.Option(help='Path to the embedding model file')],
    hyperparams_path: Annotated[Path, typer.Option(help='Path to the hyperparameters file')] = Path('./hyperparams_generator.json'),
    model_save_path: Annotated[Path, typer.Option(help='Path to the model file')] = None,
    history_path: Annotated[Path, typer.Option(help='Path to the history file')] = None,
    image_base_path: Annotated[Path, typer.Option(help='Path where to save images')] = None,
    verbose: Annotated[int, typer.Option(help='Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing model summary and debug')] = 1,
) -> None:
    verbose = VerboseMode(verbose)
    logger = create_logger(verbose)
    
    # Getting the dataset
    logger.info('Loading the MNIST dataset...')
    mnist_loader = MNISTLoader(expand_dims=True)
    logger.info('Successfully loaded the MNIST dataset')
    
    # Getting hyperparameters
    logger.info('Loading hyperparameters for the generator model...')
    hyperparams = GeneratorHyperparameters(json_path=hyperparams_path)
    
    # Creating the generator model
    logger.info('Successfully loaded the hyperparameters.')
    generator_model = MNISTGeneratorModel()
    logger.info('Using the following generator model:')
    generator_model.summary()
    
    # Loading the embedding model
    logger.info('Loading the embedding model...')
    embedding_model = MNISTEmbeddingModel.from_path(embedding_model_path, trainable=False)
    logger.info('Successfully loaded the embedding model. Its summary:')
    embedding_model._model.summary()
    
    logger.info('Setting the trainer...')
    trainer = GeneratorTrainer(logger=logger, 
                               generator_model=generator_model,
                               embedding_model=embedding_model, 
                               dataset_loader=mnist_loader, 
                               hyperparams=hyperparams)
    
    # Setting the save paths if not provided
    if model_save_path is None:
        model_save_path = Path(f'./models/generator/v{hyperparams.meta.version}.{hyperparams.meta.subversion}')
    if history_path is None:
        history_path = Path(f'./images/generator/v{hyperparams.meta.version}.{hyperparams.meta.subversion}/history.png')
    if image_base_path is None:
        image_base_path = Path(f'./images/generator/v{hyperparams.meta.version}.{hyperparams.meta.subversion}')
    
    # Creating the directories if they do not exist
    model_save_path.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    image_base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Training the model
    trainer.train(
        model_save_path=model_save_path, 
        history_save_path=history_path,
        image_save_base_path=image_base_path)
    hyperparams.save(model_save_path / 'hyperparams.json')
    
    logger.info('Loader successfully finished the training process. Exiting...')

@app.command()
def pca_generator_embedding(
    embedding_model_path: Annotated[Path, typer.Option(help='Path to the embedding model file')],
    generator_model_path: Annotated[Path, typer.Option(help='Path to the generator model file')],
    pca_save_path: Annotated[Path, typer.Option(help='Path to the PCA plot file')] = None,
    classes_to_display: Annotated[int, typer.Option(help='Number of classes to display. Defaults to 3')] = 3,
    verbose: Annotated[int, typer.Option(help='Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing model summary and debug')] = 1,
) -> None:
    verbose = VerboseMode(verbose)
    logger = create_logger(verbose)
    
    # Getting the dataset
    logger.info('Loading the MNIST dataset...')
    mnist_loader = MNISTLoader(expand_dims=True)
    logger.info('Successfully loaded the MNIST dataset')
    
    # Loading the embedding model
    logger.info('Loading the embedding model...')
    embedding_model = MNISTEmbeddingModel.from_path(embedding_model_path, trainable=False)
    logger.info('Successfully loaded the embedding model. Its summary:')
    embedding_model.summary()
    
    # Loading the generator model
    logger.info('Loading the generator model...')
    generator_model = MNISTGeneratorModel.from_path(generator_model_path, trainable=False)
    logger.info('Successfully loaded the generator model. Its summary:')
    generator_model.summary()
    
    embedding_hyperparams = EmbeddingHyperparameters(embedding_model_path / 'hyperparams.json')
    generator_hyperparams = GeneratorHyperparameters(generator_model_path / 'hyperparams.json')
    
    # Setting the save paths if not provided
    if pca_save_path is None:
        pca_save_path = Path(f'./images/evaluation/pca_embedding_{embedding_hyperparams.meta.version}.{embedding_hyperparams.meta.subversion}_generator_{generator_hyperparams.meta.version}.{generator_hyperparams.meta.subversion}.png')
    
    # Loading the dataset
    _, (X_test, y_test) = mnist_loader.get()
    y_uniques = np.unique(y_test)
    random.shuffle(y_uniques)
    y_selected = y_uniques[:classes_to_display]
    X_selected = [X_test[y_test == label] for label in y_selected]
    num_selected = [len(X_selected[i]) for i in range(len(X_selected))]
    X_selected = np.array([item for batch in X_selected for item in batch], dtype=np.float32)
    
    # Finding embeddings of real images
    y_batches_real = [[f'{label} (Real)'] * num_selected[i] for i, label in enumerate(y_selected)]
    y_batches_real = [item for batch in y_batches_real for item in batch]
    X_real = embedding_model.raw.predict(X_selected)

    # Finding embeddings of generated images
    y_predicted_batches = [[f'{label} (Generated)'] * num_selected[i] for i, label in enumerate(y_selected)]
    y_predicted_batches = [item for batch in y_predicted_batches for item in batch]
    X_generated_images = generator_model.raw.predict(X_selected)
    X_generated = embedding_model.raw.predict(X_generated_images)
    
    # Picking colors
    cmap = pyplot.cm.rainbow(np.linspace(0, 1, classes_to_display))
    color_pairs = [(cmap[i], lighten_color(cmap[i], 0.5)) for i in range(classes_to_display)]
    colors_to_display = [color for pair in color_pairs for color in pair]
    
    # Applying PCA
    logger.info('Launching PCA...')
    pca = PCAPlotter(
        X=np.array([*X_real, *X_generated], dtype=np.float64),
        y=[*y_batches_real, *y_predicted_batches],
        colors_to_display=colors_to_display
    )
    pca.plot(save_path=pca_save_path)
    logger.info('Successfully launched PCA. Plotting...')


if __name__ == '__main__':
    app()
