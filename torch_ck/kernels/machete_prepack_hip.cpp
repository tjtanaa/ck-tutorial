// !!! This is a file automatically generated by hipify!!!
#include "hip/hip_runtime.h"
#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <ATen/ATen.h>
#ifdef USE_ROCM
#include <c10/hip/HIPStream.h>
#else
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#endif
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle_hip.hpp"
// #include "ck/tensor_operation/gpu/element/element_wise_operation_hip.hpp"

// #include "ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle_hip.hpp"

// #include "ck/library/utility/check_err_hip.hpp"
#include "ck/library/utility/device_memory_hip.hpp"
#include "ck/library/utility/host_tensor.hpp"
// #include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
// #include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"


template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using F32  = float;

namespace machete {


template <typename InOutDataType>
void preShuffleBufferCPU(const InOutDataType* src,
                      InOutDataType* dst,
                      int N,
                      int K,
                      int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int N0 = N / NLane;
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> K0 N0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = k0 * KPack * NLane * KLane * N0 + n0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];
        }
    }
}


template <int threads, typename Element>
static __global__ void preShuffleBufferKernel(const Element* src, Element* dst, int N, int K, int NXdl) {
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int N0 = N / NLane;

    // Calculate the global thread index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Iterate over the elements each thread is responsible for
    for (int idx = global_idx; idx < N * K; idx += total_threads) {
        int n = idx / K;
        int k = idx % K;

        int n0 = n / NLane;
        int n1 = n % NLane;

        int k0 = k / (KLane * KPack);
        int tempk = k % (KLane * KPack);
        int k1 = tempk / KPack;
        int k2 = tempk % KPack;

        int outputIndex = k0 * KPack * NLane * KLane * N0 + n0 * KPack * NLane * KLane +
                          k1 * KPack * NLane + n1 * KPack + k2;

        dst[outputIndex] = src[n * K + k];
    }
}

template <typename Element>
static void prepackB_launcher(hipStream_t stream, const Element* src, Element* dst, int N, int K, int NXdl) {
    // Define the number of threads per block
    const int threads_per_block = 128;

    // Calculate the number of blocks needed
    int total_elements = N * K;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
   hipLaunchKernelGGL(( machete::preShuffleBufferKernel<threads_per_block, Element>), dim3(blocks), dim3(threads_per_block), 0, stream, src, dst, N, K, NXdl);
}

} // namespace machete


void prepackB(at::Tensor& B, at::Tensor& Bprepacked) {

  const int N = B.size(0);
  const int K = B.size(1);
  const int NXdl = 16;
  
  auto stream = at::cuda::getCurrentHIPStream().stream();

  machete::prepackB_launcher(stream, 
  reinterpret_cast<FP8*>(B.data_ptr()), 
  reinterpret_cast<FP8*>(Bprepacked.data_ptr()), 
  N, K, NXdl);

  hipError_t err = hipGetLastError();
  if (hipSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

void prepackB_cpu(at::Tensor& B, at::Tensor& Bprepacked) {
  const int N = B.size(0);
  const int K = B.size(1);
  const int NXdl = 16;

  auto B_ptr = reinterpret_cast<FP8*>(B.data_ptr());
  auto Bprepacked_ptr = reinterpret_cast<FP8*>(Bprepacked.data_ptr());
  machete::preShuffleBufferCPU(B_ptr, Bprepacked_ptr, N, K, NXdl);

}