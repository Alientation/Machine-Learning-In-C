# A neural network framework in c
learned a lot about the math behind neural networks (gradient descent/backprop)
really just an application of material taught in multivariable and lin alg.

## Details
Simple, really not too optimized multilayer perceptron model. A model is structured with a 
series of layers explicitly defined by the user.

## Progress
XOR trained model\
digit recognizer trained on handdrawn dataset (only 100 examples, 10 per digit, which explains the model's low performance)
model visualizer in raylib
drawing panel for models that take in images, to create dataset files\
dataset file loader\

## Goals
- Train a model on MNIST
- Optimize to improve training speed
- Add momentum, regularization, dropout, and warmup training
- Download and use pretrained model
- Add more types of layers like convolution, maxpooling, etc
- ..possibly use third party linear algebra libraries, but trying not to.. don't want to use any for this project other than raylib

## Some Images
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/neuralnet_vis.PNG)
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/drawingpanel.PNG)
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/digit_predictor.PNG)
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/drawing_digit_predictor.PNG)
