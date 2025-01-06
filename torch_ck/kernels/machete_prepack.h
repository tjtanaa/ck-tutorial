#include <torch/types.h>
#include <torch/torch.h>

void prepackB(at::Tensor& B, at::Tensor& Bprepacked);
void prepackB_cpu(at::Tensor& B, at::Tensor& Bprepacked);