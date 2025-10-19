import torch

# Create a sample tensor
t = torch.tensor([[[1, 2, 3], [4, 5, 6]],
                  [[7, 8, 9], [10, 11, 12]]])
print("Original Tensor:\n", t)
print("Original Tensor Shape:", t.shape)

# Flatten the entire tensor
flattened_t = t.flatten()

print("\nFlattened Tensor (entire):\n", flattened_t)
print("Flattened Tensor Shape:", flattened_t.shape)

# Flatten specific dimensions (e.g., from dimension 0 to 1)
partially_flattened_t = t.flatten(1)
print("\nPartially Flattened Tensor (dim 0 to 1):\n", partially_flattened_t)
print("Partially Flattened Tensor Shape:", partially_flattened_t.shape)
