#include <torch/types.h>
#include <torch/torch.h>
torch::Tensor linear_relu_abde_i8(
			torch::Tensor A_,
			torch::Tensor B_,
			torch::Tensor D_,
			float alpha,
			float beta);