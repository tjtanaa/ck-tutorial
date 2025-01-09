#include <torch/types.h>
#include <torch/torch.h>

void prepackB(at::Tensor& B, at::Tensor& Bprepacked);
void prepackB_cpu(at::Tensor& B, at::Tensor& Bprepacked);

// at::Tensor machete_mm_out(
//     at::Tensor XQ,
//     at::Tensor WQ,
//     at::Tensor x_scale,
//     at::Tensor w_scale,
//     at::Tensor Y);
    
// at::Tensor machete_mm(
//     at::Tensor XQ,
//     at::Tensor WQ,
//     at::Tensor x_scale,
//     at::Tensor w_scale);

at::Tensor machete_mm_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    int op_id);