
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

activations = {
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "ReLU": relu,
    "Leaky ReLU": leaky_relu
}

# 1. Randomly generate network structure
n_inputs = np.random.randint(3, 7)  # [3, 6]
n_hidden_layers = np.random.randint(1, 4)  # [1, 3]
neurons_per_layer = [np.random.randint(2, 6) for _ in range(n_hidden_layers)]  # [2, 5] per layer

# 2. Randomly generate input features, weights, and biases
inputs = np.random.uniform(-10, 10, n_inputs)

weights = []
biases = []

layer_input_size = n_inputs
for n_neurons in neurons_per_layer:
    w = np.random.uniform(-1, 1, (n_neurons, layer_input_size))
    b = np.random.uniform(-1, 1, n_neurons)
    weights.append(w)
    biases.append(b)
    layer_input_size = n_neurons

# Output layer: 1 neuron
w_out = np.random.uniform(-1, 1, (1, layer_input_size))
b_out = np.random.uniform(-1, 1, 1)
weights.append(w_out)
biases.append(b_out)

# Print network structure
print("Network Structure:")
print(f"  Number of input features: {n_inputs}")
print(f"  Number of hidden layers: {n_hidden_layers}")
print(f"  Neurons per hidden layer: {neurons_per_layer}")
print(f"  Input values: {np.round(inputs, 2).tolist()}")
for i in range(n_hidden_layers):
    print(f"  Layer {i+1} weights:\n{np.round(weights[i], 2)}")
    print(f"  Layer {i+1} biases: {np.round(biases[i], 2)}")
print(f"  Output layer weights:\n{np.round(weights[-1], 2)}")
print(f"  Output layer bias: {np.round(biases[-1], 2)}")

# 3. Forward pass for each activation function
final_outputs = {}

for name, activation in activations.items():
    a = inputs.copy()
    for i in range(n_hidden_layers):
        z = np.dot(weights[i], a) + biases[i]
        a = activation(z)
    # Output layer (always 1 neuron)
    z_out = np.dot(weights[-1], a) + biases[-1]
    a_out = activation(z_out)
    final_outputs[name] = float(a_out[0])

# 4. Print and plot results
print("\nFinal Output for Each Activation Function:")
for name, value in final_outputs.items():
    print(f"  {name}: {round(value, 4)}")

plt.figure(figsize=(7, 5))
plt.bar(final_outputs.keys(), final_outputs.values(), color=['#4e79a7', '#f28e2b', '#76b7b2', '#e15759'])
plt.ylabel("Final Output Value")
plt.title("Neural Network Output by Activation Function")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
