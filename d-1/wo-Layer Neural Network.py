import random
import math

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Generate 3 random input values
inputs = [round(random.uniform(-1, 1), 2) for _ in range(3)]

# First layer: 2 neurons, each with 3 random weights and 1 random bias
hidden_weights = [
    [round(random.uniform(-1, 1), 2) for _ in range(3)],
    [round(random.uniform(-1, 1), 2) for _ in range(3)]
]
hidden_biases = [round(random.uniform(-1, 1), 2) for _ in range(2)]

# Calculate hidden layer outputs
hidden_outputs = []
for i in range(2):
    z = sum(inputs[j] * hidden_weights[i][j] for j in range(3)) + hidden_biases[i]
    out = sigmoid(z)
    hidden_outputs.append(round(out, 2))

# Second layer: 1 neuron with 2 random weights and 1 random bias
output_weights = [round(random.uniform(-1, 1), 2) for _ in range(2)]
output_bias = round(random.uniform(-1, 1), 2)

# Calculate final output
z_out = sum(hidden_outputs[j] * output_weights[j] for j in range(2)) + output_bias
final_output = sigmoid(z_out)

# Print all values and intermediate outputs
print(f"Inputs: {inputs}")
print(f"Hidden layer weights: {hidden_weights}")
print(f"Hidden layer biases: {hidden_biases}")
print(f"Hidden outputs: {hidden_outputs}")
print(f"Output layer weights: {output_weights}")
print(f"Bias: {output_bias}")
print(f"Final Output: {round(final_output, 3)}")
