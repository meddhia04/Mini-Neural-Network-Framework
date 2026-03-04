import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network with random weights and biases
        
        Parameters:
        input_size: number of input features
        hidden_size: number of neurons in hidden layer
        output_size: number of output classes
        learning_rate: step size for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Initialize weights and biases with small random values
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # For tracking loss
        self.losses = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        X: input data (n_samples, input_size)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Cross-entropy loss
        y_true: true labels (one-hot encoded)
        y_pred: predicted probabilities
        """
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def backward(self, X, y_true, y_pred):
        """
        Backward propagation (compute gradients)
        """
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = y_pred - y_true  # Derivative of cross-entropy with softmax
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        """
        Update weights and biases using gradient descent
        """
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        X: training data
        y: training labels (one-hot encoded)
        epochs: number of training iterations
        """
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = self.compute_accuracy(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """Predict class probabilities"""
        y_pred = self.forward(X)
        return y_pred
    
    def predict_classes(self, X):
        """Predict class labels"""
        y_pred = self.predict(X)
        return np.argmax(y_pred, axis=1)
    
    def compute_accuracy(self, X, y):
        """Compute classification accuracy"""
        y_pred = self.predict_classes(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)


# Let's create some sample data to test our network
def create_sample_data():
    """Create a simple classification dataset"""
    np.random.seed(42)
    
    # Generate three clusters of points
    n_samples = 300
    X = np.zeros((n_samples, 2))
    y = np.zeros((n_samples, 3))
    
    # Class 0
    X[:100] = np.random.randn(100, 2) * 0.5 + [-1, -1]
    y[:100, 0] = 1
    
    # Class 1
    X[100:200] = np.random.randn(100, 2) * 0.5 + [1, -1]
    y[100:200, 1] = 1
    
    # Class 2
    X[200:] = np.random.randn(100, 2) * 0.5 + [0, 1]
    y[200:, 2] = 1
    
    return X, y

# Create and prepare the data
X, y = create_sample_data()

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create and train the neural network
nn = SimpleNeuralNetwork(
    input_size=2,
    hidden_size=5,
    output_size=3,
    learning_rate=0.5
)

print("Training Neural Network...")
nn.train(X_train, y_train, epochs=2000, verbose=True)

# Evaluate on test data
test_accuracy = nn.compute_accuracy(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Plot the training loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(nn.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot decision boundary
plt.subplot(1, 2, 2)

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict on mesh grid
Z = nn.predict_classes(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.tight_layout()
plt.show()