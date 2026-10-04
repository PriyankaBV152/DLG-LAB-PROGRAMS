# Week 01 - Neural Network Implementation for XOR

## Program Title

Implementation of XOR using a Neural Network

## Aim

To implement the XOR operation using a neural network using NumPy and Keras.

## Dataset Used

XOR truth table.

| Input 1 | Input 2 | Expected Output |
|---------|---------|-----------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

## Program 1(a)

A neural network was implemented from scratch using NumPy.

It uses:
- Sigmoid activation function
- Forward propagation
- Backpropagation
- Weight and bias updates
- 10,000 epochs

## Program 1(b)

A neural network was implemented using Keras.

It uses:
- Input layer with 2 inputs
- Hidden layer with 8 neurons
- ReLU activation
- Sigmoid output layer
- SGD optimizer
- 1000 epochs

## Results

Program 1(a) produced predictions close to:

```text
0
1
1
0