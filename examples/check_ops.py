import torch
import torch_int

print(dir(torch_int))
M, N, K = (100, 120, 140)

A = torch.randint(-20, 20, size=(M, K), dtype=torch.int8)
B = torch.randint(-20, 20, size=(K, N), dtype=torch.int8)
D = torch.randint(-20, 20, size=(M, N), dtype=torch.int8)

output = torch_int.linear_relu_abde_i8(
    A, B, D, 0.1, 0.1
)
print(output)