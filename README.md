# SVHN-ML
This project is part of my personal journey thought Machine Learning and Deep Learning. I explore the SVHN dataset with different models and techniques I learn along the way.

## SVHN Dataset
"SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images." - [Street View House Numbers (SVHN) Dataset Home Page](http://ufldl.stanford.edu/housenumbers)

The dataset characteristics are:
- 10 classes, 1 for each digit.
- 73257 digits for training
- 26032 digits for testing
- 531131 additional samples

It comes in two formats:
1. Original images with character level bounding boxes.
2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

In this project, I'm using the second format.

## Utilities
### Pre-processing
_preproc.py_ provides a loading data function and some utilities. If run as main, it transforms the dataset into a more convinient format for my machine learning models.

### Logging
_mylogger.py_ contains a auxiliary class for based on python's default _logging_ to encapsulate the logging system I'm using.

## Models
### Artificial Neural Network
_ann.py_ contains a simple ANN model with two hidden layers and Adam as optimizer.
