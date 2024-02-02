<h1 align="center"> :framed_picture: :interrobang: Distortion Generator </h1>

<p align="center">
  | <a href="https://arxiv.org/abs/2401.15048">[arXiv]</a> | <a href="https://paperswithcode.com/paper/unrecognizable-yet-identifiable-image">[Papers with Code]</a> | <a href="citation">[Citation]</a>  |

Neural network for creating distortion while keeping embeddings as close as possible. Part of the
research paper _"Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings."_

The code is written in [_TensorFlow v2.12_](https://www.tensorflow.org/).

![Example Generations](images/meta/example_generations.png)

## ℹ️ Paper info

> [**Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings**](https://arxiv.org/abs/2401.15048)
> 
> [Dmytro Zakharov](https://scholar.google.com/citations?user=WL-8aoAAAAAJ&hl=en)<sup>1</sup>, [Oleksandr Kuznetsov](https://scholar.google.com/citations?user=DUI-bncAAAAJ&hl=en)<sup>1,2</sup>, [Emanuele Frontoni](https://scholar.google.com/citations?user=Vgi8nAcAAAAJ&hl=en)<sup>2</sup>
> 
> <sup>1</sup> V. N. Karazin Kharkiv National University, Ukraine
> 
> <sup>2</sup> University of Macerata, Italy
> 
> _Preprint at arXiv_
</p>

## :file_folder: Structure

The project is structured as follows:
| File/Folder | Description |
| ----------- | ----------- |
| [`cli.py`](cli.py) | CLI for running training or evaluation |
| [`src`](src) | All source files for training and evaluating the models | 
| [`images`](images) | Images with example generations, evaluation plots etc. | 
| [`hyperparams_embedding.json`](hyperparams_embedding.json) | Hyperparameters for training the embedding model |
| [`hyperparams_generator.json`](hyperparams_generator.json) | Hyperparameters for training the generator model | 
| [`dataset`](dataset) | Dataset which was used for training (actually, the portion of it since we do not want to put everything into the repository) |
| [`models`](models) | Models' weights after the training. Just so you know, the newest versions of the generator are not included since they weigh too much for GitHub to handle. |

<a name="citation"></a>
## 🗒️ Citation
```bib
@misc{zakharov2024unrecognizable,
      title={Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings}, 
      author={Dmytro Zakharov and Oleksandr Kuznetsov and Emanuele Frontoni},
      year={2024},
      eprint={2401.15048},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
