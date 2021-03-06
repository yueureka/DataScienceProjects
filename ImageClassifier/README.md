# Project: Image Classifier with Deep Neural Network
In this project, an image classifier is trained to recognize different species of flowers.This [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) is used with total 102 flower categories.

The project is broken down into multiple steps:

1. Load and preprocess the image dataset
2. Train the image classifier on your dataset
3. Use the trained classifier to predict image content

The deep learning model is transfer-learnt from Vgg11, and I've rebuilt fully connected layers and implemented prediction functions, and optimized the training parameters such as learning rate, layer size, error function, dropout rate, epoch number, and optimizer. 

# Installation
The code runs in Python 3, Python libraries such as pandas, numpy, seaborn and matplotlib are used in the code. Pytorch is used to train the model, and GPU mode is applied during the training process. 
