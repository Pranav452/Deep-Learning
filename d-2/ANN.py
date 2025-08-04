# custom_ann.py

import torch
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate dataset and save to CSV (skip if you already have a CSV)
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

# 4. Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 5. Initialize weights and bias
torch.manual_seed(0)
W = torch.randn((2, 1), requires_grad=True, dtype=torch.float32)
b = torch.zeros(1, requires_grad=True, dtype=torch.float32)

# 6. Sigmoid function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# 7. Binary cross-entropy loss
def binary_cross_entropy(y_pred, y_true):
    # Clamp y_pred to avoid log(0)
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()

# 8. Training loop
lr = 0.1
epochs = 30

for epoch in range(1, epochs + 1):
    # Forward pass
    logits = X_train @ W + b
    y_pred = sigmoid(logits)
    loss = binary_cross_entropy(y_pred, y_train)

    # Backward pass
    loss.backward()

    # Update weights and bias manually (gradient descent)
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad

    # Zero gradients
    W.grad.zero_()
    b.grad.zero_()

    # Print loss at some intervals
    if epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 9. Evaluate on test set
with torch.no_grad():
    logits_test = X_test @ W + b
    y_pred_test = sigmoid(logits_test)
    predictions = (y_pred_test >= 0.5).float()
    accuracy = (predictions == y_test).float().mean().item() * 100
    print(f"Accuracy on test set = {accuracy:.1f}%")
