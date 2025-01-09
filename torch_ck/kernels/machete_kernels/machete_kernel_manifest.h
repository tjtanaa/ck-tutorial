#include <cstdlib>

#include <ATen/ATen.h>


// at::Tensor machete_256_16x64x256_16x16_1x1_16x16_intrawave_v3_2(
//     at::Tensor XQ,
//     at::Tensor WQ,
//     at::Tensor x_scale,
//     at::Tensor w_scale,
//     at::Tensor Y
// );

at::Tensor f8f8bf16_pshuffleb_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y
);

// at::Tensor machete_256_16x64x256_16x16_16x16_1x1_16x16_intrawave_v3_2(
//     at::Tensor XQ,
//     at::Tensor WQ,
//     at::Tensor x_scale,
//     at::Tensor w_scale,
//     at::Tensor Y
// );