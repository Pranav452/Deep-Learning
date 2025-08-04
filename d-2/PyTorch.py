# tensor_operations.py

# 1. Import torch and check for GPU availability
import torch

# 2. Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Create tensor A of shape (3, 2) with random values from a normal distribution
A = torch.randn(3, 2)
print("A:", A)

# 4. Create tensor B of shape (2, 3) with random values from a normal distribution
B = torch.randn(2, 3)
print("B:", B)

# 5. Matrix multiplication: C = A @ B
C = A @ B
print("C (A @ B):", C)

# 6. Element-wise addition: D = A + torch.ones_like(A)
D = A + torch.ones_like(A)
print("D (A + ones):", D)

# 7. Move result C to GPU if available
C = C.to(device)
print("C is on device:", C.device)
