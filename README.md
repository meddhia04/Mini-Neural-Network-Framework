# Mini Neural Network Framework

A lightweight neural network built from scratch using NumPy. Implements forward/backward propagation and gradient descent for multi-class classification.

## Architecture
- **Input layer**: 2 neurons
- **Hidden layer**: 5 neurons (sigmoid activation)
- **Output layer**: 3 neurons (softmax activation)
## Requirements
- Python 3.6+
- NumPy
- Matplotlib

Install: `pip install numpy matplotlib`

## Usage
```python
# Create and train
nn = SimpleNeuralNetwork(2, 5, 3, 0.5)
nn.train(X_train, y_train, 2000)

# Evaluate
accuracy = nn.compute_accuracy(X_test, y_test)
