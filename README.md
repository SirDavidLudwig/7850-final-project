# 7850 Final Project

This repository contains the notebooks and models for the project: Exploring Transformer-based Generative Adversarial Network Methods for Set Generation. This project explores the implementation of generative adversarial network variants, AC-GAN and Cycle-GAN models in use with unstructured data. Furthermore, it also explores the incorporation of high-dimensional data in the form of pre-trained DNA embeddings.

This project is based on the paper: [Generative Adversarial Set Transformers](https://www.ml.informatik.tu-darmstadt.de/papers/stelzner2020ood_gast.pdf), and the unofficial implementation located in the public Git repository: [tf-gast](https://github.com/DLii-Research/tf-gast).

**Note:** This repository does not contain the components of the baseline model as this model was obtained via the [tf-gast](https://github.com/DLii-Research/tf-gast) repository mentioned above.

## Development/Deployment Instructions

Before you can run or train any of these models, be sure your environment meets the required dependencies.

**Dependencies**

- [Cython](https://cython.org/)
- [Tensorflow 2.6](https://www.tensorflow.org/)
- [Tensorflow Addons](https://www.tensorflow.org/addons)
- [Numpy](https://numpy.org/)
- [Set Transformers](https://github.com/DLii-Research/tf-set-transformer)

### Demo

The `Demo.ipynb` notebook within this repository deploys each of the successful models using pre-trained weights. Simply clone this repository and run through the demo notebook to see each of the models in action with plots.

These pre-trained weights are supplied by default through this repository. If you'd like to train your own model to obtain your own weights, see the Models and Training section described below.

### Models and Training

Each of the models within this project are located in their own self-conatined Jupyter Notebooks. This project is organized such that each model is defined and fully trained within its own notebook. If you'd like to examine our training procedures, or even train your own version of the model, simply run the corresponding model's notebook. These notebooks are designed to run end-to-end, defining the model architecture, hyperparameters, training procedure, and evaluation.