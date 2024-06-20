# A neural network framework in c
learned a lot about the math behind neural networks (gradient descent/backprop)
really just an application of what i learned in multivariable and lin alg.

## Details
Simple, really not too optimized multilayer perceptron model. A model is structured with a 
series of layers explicitly defined by the user, where dense layers are separate from the activation functions
that would normally go together just to be more verbose.

## Progress
XOR trained model
digit recognizer trained on handdrawn dataset (only 100 examples, 10 per digit, which explains the model's low performance)
model visualizer in raylib
drawing panel for models that take in images, to create dataset files
dataset file loader

## Goals
- Train a model on MNIST
- Optimize, currently sitting at 1.6 million examples a second for a model (2 -> 2 -> 1)
     - though it appears that after having the model run all the tests after each training epoch, the total time went up
     marginally by only a couple hundred ms
- Add momentum, regularization, dropout, and warmup training
- Download and use pretrained model
- multi dimensional matrices (for convolution layers)
- Add more types of layers like convolution, maxpooling, etc
- Support mini batching
- ..possibly use third party linear algebra libraries, but trying not to.. don't want to use any for this project other than raylib

## Some Images
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/neuralnet_vis.PNG)
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/drawingpanel.PNG)
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/digit_predictor.PNG)
![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/github_images/drawing_digit_predictor.PNG)
