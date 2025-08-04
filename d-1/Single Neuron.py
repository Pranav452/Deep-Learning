import math

# Get user input for x1 and x2
x1, x2 = map(float, input("Enter x1, x2: ").split())

# Get user input for w1 and w2
w1, w2 = map(float, input("Enter w1, w2: ").split())

# Get user input for bias
b = float(input("Enter bias: "))

# Calculate z
z = x1 * w1 + x2 * w2 + b

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Calculate output
output = sigmoid(z)

# Print output rounded to 3 decimal places
print(f"Neuron output: {round(output, 3)}")
