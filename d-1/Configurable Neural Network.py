import random
import math

# Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

# Get user input
n = int(input("Enter number of inputs: "))
h = int(input("Enter number of hidden neurons: "))
activation_choice = input("Enter activation (sigmoid/relu): ").strip().lower()

# Choose activation function
if activation_choice == "sigmoid":
    activation = sigmoid
    act_name = "Sigmoid"
elif activation_choice == "relu":
    activation = relu
    act_name = "ReLU"
else:
    print("Invalid activation function. Use 'sigmoid' or 'relu'.")
    exit()

# Generate random inputs
inputs = [round(random.uniform(-1, 1), 2) for _ in range(n)]

# Generate random weights and biases for hidden layer
hidden_weights = []
hidden_biases = []
for _ in range(h):
    weights = [round(random.uniform(-1, 1), 2) for _ in range(n)]
    bias = round(random.uniform(-1, 1), 2)
    hidden_weights.append(weights)
    hidden_biases.append(bias)

# Calculate hidden layer outputs
hidden_outputs = []
for i in range(h):
    z = sum(inputs[j] * hidden_weights[i][j] for j in range(n)) + hidden_biases[i]
    out = activation(z)
    hidden_outputs.append(round(out, 2))

# Output layer: 1 neuron with h weights and 1 bias
output_weights = [round(random.uniform(-1, 1), 2) for _ in range(h)]
output_bias = round(random.uniform(-1, 1), 2)

# Calculate final output (no activation on output, as in example)
z_out = sum(hidden_outputs[j] * output_weights[j] for j in range(h)) + output_bias
final_output = round(z_out, 3)

# Print all values and intermediate outputs
print(f"Inputs: {inputs}")
print(f"Hidden layer weights: {hidden_weights}")
print(f"Hidden biases: {hidden_biases}")
print(f"Hidden outputs ({act_name}): {hidden_outputs}")
print(f"Output layer weights: {output_weights}")
print(f"Bias: {output_bias}")
print(f"Final Output: {final_output}")
