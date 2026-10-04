# Week 02 - Batch Gradient Descent and SGD

## Program Title

Comparison of Batch Gradient Descent and Stochastic Gradient Descent using Neural Networks

## Aim

To implement and compare Batch Gradient Descent and Stochastic Gradient Descent for training a neural network.

## Dataset Used

The Two-Moons dataset was generated using the `make_moons()` function from Scikit-learn.

n_samples = 400  
noise = 0.2  
random_state = 1

## Program 2(a)

A neural network was implemented from scratch using NumPy.

It uses:

- ReLU activation function
- Sigmoid activation function
- Forward propagation
- Backpropagation
- Batch Gradient Descent
- Stochastic Gradient Descent
- 200 training epochs

The neural network consists of 2 input neurons, two hidden layers with 16 neurons each, and 1 output neuron.

## Program 2(b)

A neural network was implemented using Keras.

It uses:

- Two hidden layers with 16 neurons each
- ReLU activation
- Sigmoid output layer
- Binary cross-entropy loss
- SGD optimizer
- Learning rate of 0.5
- 200 training epochs

Both Batch Gradient Descent and SGD were used for training.

## Results

Program 2(a) achieved:

Batch GD accuracy: 0.965  
SGD accuracy: 0.9675

Program 2(b) achieved:

Batch GD accuracy: 0.9625  
SGD accuracy: 0.9725

Both methods successfully trained the neural network on the Two-Moons dataset.

## Files

- Program2(a)BatchGD_SGD.py - Neural network implementation using NumPy
- Program2(b)Keras_BatchGD_SDG.py - Neural network implementation using Keras
- Output Screenshots/ - Screenshots of the program outputs