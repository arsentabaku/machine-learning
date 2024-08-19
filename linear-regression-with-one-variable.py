import numpy as np
import matplotlib.pyplot as plt

# Define the training data
# x_train: size of the house in 1000 square feet
# y_train: price of the house in 1000s of dollars
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"Training data: x_train = {x_train}, y_train = {y_train}")

# Determine the number of training examples
m = len(x_train)
print(f"Number of training examples: {m}")

# Select training examples
for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"Training example {i}: (x^{i}, y^{i}) = ({x_i}, {y_i})")

# Initialize the model's parameters
w = 200  # weight
b = 100  # bias

# Compute the cost for a house of size 1.2 (1200 square feet)
x_new = 1.2
cost_1200sqft = w * x_new + b
print(f"The estimated cost for a 1200 square feet house is ${cost_1200sqft:.0f} thousand dollars")

# Define a function to compute the model's prediction
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray): Data, m examples
      w (scalar): Model parameter (weight)
      b (scalar): Model parameter (bias)
    Returns:
      ndarray: Model prediction
    """
    return w * x + b

# Compute the model's prediction for the training data
predicted_prices = compute_model_output(x_train, w, b)

# Plot the model's predictions and the actual values
plt.plot(x_train, predicted_prices, c='b', label='Model Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()
