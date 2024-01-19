# :framed_picture: :interrobang: Distortion Generator

Neural network for creating distortion while keeping embeddings as close as possible. Part of the
research paper _"Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings."_

The code is written in [_TensorFlow v2.12_](https://www.tensorflow.org/).

![Example Generations](images/meta/example_generations.png)

## :file_folder: Structure

The project is structured as follows:
| File/Folder | Description |
| ----------- | ----------- |
| [`cli.py`](cli.py) | CLI for running training or evaluation |
| [`src`](src) | All source files for training and evaluating the models | 
| [`images`](images) | Images with example generations, evaluation plots etc. | 
| [`hyperparams_embedding.json`](hyperparams_embedding.json) | Hyperparameters for training the embedding model |
| [`hyperparams_generator.json`](hyperparams_generator.json) | Hyperparameters for training the generator model | 
