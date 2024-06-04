# A neural network framework in c
learned a lot about the math behind neural networks (gradient descent/backprop)
really just an application of what i learned in multivariable and lin alg.

![alt text](https://github.com/Alientation/Machine-Learning-In-C/blob/master/visualizer.PNG)

## Details
Simple, really not too optimized multilayer perceptron model. A model is structured with a 
series of layers explicitly defined by the user, where dense layers are separate from the activation functions
that would normally go together just to be more verbose.

## Progress
XOR trained model
Basic visualizer in raylib

## Goals
- Train a model on MNIST
- Optimize, currently sitting at 1.6 million examples a second for a model (2 -> 2 -> 1)
     - though it appears that after having the model run all the tests after each training epoch, the total time went up
     marginally by a couple hundred ms
- Improve UI of visualizer and add more features like manually tuning weights or giving inputs to the model
- Add momentum, regularization, dropout, and warmup training
- Download and use pretrained model
- Support mini batching
- ..possibly use third party linear algebra libraries, but trying not to.. don't want to use any for this project other than raylib
