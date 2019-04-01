# progressive-resizing

This repository contains the code for building a convolutional neural network machine learning classifier in three parts. It is the companion repo for the article ["Boost your CNN performance with progressive resizing in Keras"](https://medium.com/@aleksey.bilogur/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20).

The three components are:

* A first model that works on 48x48 images.
* A second model that works on 96x96 images.
* A final model that works on 192x192 images.

The resultant model is a "three layer cake": each larger-scale model subsumes the previous smaller-scale model layers and weights in its architecture.

This approach is meant to demonstrate a workflow and technique for building neural networks known as "progressive resizing". Progressive resizing has been used to good effect by Jeremy Howard, who used to achieve a top 10% finish in the [Planet Kaggle Competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space), and he uses it throughout his fast.ai course ["Practical Deep Learning for Coders"](https://course.fast.ai/).

## The data

This project uses the "Open Fruit" dataset, a dataset of fruit images taken from Google's [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html).

You can download the data for yourself from the source data package on Quilt T4:

```bash
$ pip install -U t4
$ python -c "import t4; t4.Package.install('s3://quilt-example', 'quilt/open_fruit', './')"
```

Alternatively, to build this data from source again (warning: this takes a long time!):

```bash
$ git clone https://github.com/quiltdata/open-images.git
$ cd open-images/
$ conda env create -f environment.yml
$ source activate quilt-open-images-dev
$ pip install -e ./src/openimager/
$ python -c "import openimager; openimager.download(['Apple', 'Grape', 'Orange', 'Pomegranate', 'Banana', 'Grapefruit', 'Peach', 'Strawberry', 'Cantaloupe', 'Lemon', 'Pear', 'Tomato', 'Common fig', 'Mango', 'Pineapple', 'Watermelon'])"
```

The run the code in the [`build-dataset.ipynb`](https://github.com/ResidentMario/progressive-resizing/blob/master/notebooks/build-dataset.ipynb) notebook to generate the final cropped images.

To learn more about using Google Open Images for building new datasets, see ["How to classify photos in 600 classes using nine million Open Images"](https://medium.freecodecamp.org/how-to-classify-photos-in-600-classes-using-nine-million-open-images-65847da1a319).

## The models

The [`build-models.ipynb`](https://github.com/ResidentMario/progressive-resizing/blob/master/notebooks/build-models.ipynb) notebook is the notebook where I work through the model definitions. The actual model builds were executed using the [`fahr`](https://residentmario.github.io/fahr/) remote training CLI, which I have been building side-by-side with this project. To model resources are in the various `resnet*` folders in this repository.

Note that you can execute `setup.sh` to install the required resources (a recent version of `keras-preprocessing`, `fahr`, etc.).