# custom_ann_hidden_layer.py

import torch
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate and save dataset (skip if you already have binary_data.csv)
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=1)
df = pd.DataFrame(X, columns=['f1', 'f2'])
df['label'] = y
df.to_csv('binary_data.csv', index=False)

# 2. Load dataset
data = pd.read_csv('binary_data.csv')
X = data[['f1', 'f2']].values
y = data['label'].values

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 4. Initialize weights and biases for 2-4-1 network
torch.manual_seed(0)
W1 = torch.randn(2, 4, requires_grad=True, dtype=torch.float32)
b1 = torch.zeros(1, 4, requires_grad=True, dtype=torch.float32)
W2 = torch.randn(4, 1, requires_grad=True, dtype=torch.float32)
b2 = torch.zeros(1, 1, requires_grad=True, dtype=torch.float32)

# 5. Binary cross-entropy loss
def binary_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()

# 6. Training loop
lr = 0.1
epochs = 30

for epoch in range(1, epochs + 1):
    # Forward pass
    Z1 = X_train @ W1 + b1         # (batch, 4)
    A1 = torch.relu(Z1)            # ReLU activation
    Z2 = A1 @ W2 + b2              # (batch, 1)
    y_pred = torch.sigmoid(Z2)     # Sigmoid output

    # Loss
    loss = binary_cross_entropy(y_pred, y_train)

    # Backward pass
    loss.backward()

    # Manual update
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad

    # Zero gradients
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    # Print loss at start and end
    if epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 7. Evaluate on test set
with torch.no_grad():
    Z1 = X_test @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2
    y_pred = torch.sigmoid(Z2)
    predictions = (y_pred >= 0.5).float()
    accuracy = (predictions == y_test).float().mean().item() * 100
    print(f"Accuracy: {accuracy:.1f}%")
