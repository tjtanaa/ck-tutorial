#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <ATen/ATen.h>
#ifdef USE_ROCM
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
// #include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// #include "ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle.hpp"

// #include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
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


// template <typename InOutDataType>
// void preShuffleBufferCPU(const InOutDataType* src,
//                       InOutDataType* dst,
//                       int N,
//                       int K,
//                       int NXdl)
// {
//     int KPack = 16;
//     int NLane = NXdl;
//     int KLane = 64 / NLane;

//     int N0 = N / NLane;
//     // K -> K0 KLane KPack
//     // N -> N0 NLane
//     // N, K -> K0 N0 KLane NLane KPack
//     int tempk;
//     for(int n = 0; n < N; ++n)
//     {
//         for(int k = 0; k < K; ++k)
//         {
//             int n0 = n / NLane;
//             int n1 = n % NLane;

//             int k0 = k / (KLane * KPack);
//             tempk  = k % (KLane * KPack);
//             int k1 = tempk / KPack;
//             int k2 = tempk % KPack;

//             int outputIndex = k0 * KPack * NLane * KLane * N0 + n0 * KPack * NLane * KLane +
//                               k1 * KPack * NLane + n1 * KPack + k2;

//             dst[outputIndex] = src[n * K + k];
//         }
//     }
// }


// template <int threads, typename Element>
// static __global__ void preShuffleBufferKernel(const Element* src, Element* dst, int N, int K, int NXdl) {
//     int KPack = 16;
//     int NLane = NXdl;
//     int KLane = 64 / NLane;

//     int N0 = N / NLane;

//     // Calculate the global thread index
//     int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_threads = gridDim.x * blockDim.x;

//     // Iterate over the elements each thread is responsible for
//     for (int idx = global_idx; idx < N * K; idx += total_threads) {
//         int n = idx / K;
//         int k = idx % K;

//         int n0 = n / NLane;
//         int n1 = n % NLane;

//         int k0 = k / (KLane * KPack);
//         int tempk = k % (KLane * KPack);
//         int k1 = tempk / KPack;
//         int k2 = tempk % KPack;

//         int outputIndex = k0 * KPack * NLane * KLane * N0 + n0 * KPack * NLane * KLane +
//                           k1 * KPack * NLane + n1 * KPack + k2;

//         dst[outputIndex] = src[n * K + k];
//     }
// }


template <typename InOutDataType>
void preShuffleBufferCPU(const InOutDataType* src, InOutDataType* dst, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    // printf("CPU, outputIndex, sourceIndex\n");
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

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];
            // if( n < 4 ){
            //     printf("CPU,%d,%d\n", outputIndex, n * K + k);
            // }
        }
    }
}



template <typename Element>
static __global__ void preShuffleBufferKernel(const Element* src, Element* dst, int N, int K, int NXdl) {
    const int KPack = 16;
    const int NLane = NXdl;
    const int KLane = 64 / NLane;
    const int K0 = K / (KLane * KPack);

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    // printf("GPU, outputIndex, sourceIndex\n");

    for (int idx = global_idx; idx < N * K; idx += total_threads) {
        int n = idx / K;
        int k = idx % K;

        if (n < N && k < K)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            int tempk = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                            k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[idx];  // This is equivalent to src[n * K + k]
            // dst[outputIndex] = src[n * K + k];  // This is equivalent to src[n * K + k]
            
            // if( n < 4 ){
            //     printf("GPU,%d,%d\n", outputIndex, idx);
            // }
        }
    }
}


template <typename InOutDataType>
void preShuffleBufferCPUDebug(const InOutDataType* src, InOutDataType* dst, int* srcIndices, int* dstIndices, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    // printf("CPU, outputIndex, sourceIndex\n");
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

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];
            srcIndices[n * K + k] = n * K + k;
            dstIndices[n * K + k] = outputIndex;

            // if( n < 4 ){
            //     printf("CPU,%d,%d\n", outputIndex, n * K + k);
            // }
        }
    }
}



template <typename Element>
static __global__ void preShuffleBufferKernelDebug(const Element* src, Element* dst, int* srcIndices, int* dstIndices, int N, int K, int NXdl) 
{
    const int KPack = 16;
    const int NLane = NXdl;
    const int KLane = 64 / NLane;
    const int K0 = K / (KLane * KPack);

    // int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int total_threads = gridDim.x * blockDim.x;
    // // printf("GPU, outputIndex, sourceIndex\n");

    // for (int idx = global_idx; idx < N * K; idx += total_threads) {
    //     int n = idx / K;
    //     int k = idx % K;

    //     if (n < N && k < K)
    //     {
    //         int n0 = n / NLane;
    //         int n1 = n % NLane;

    //         int k0 = k / (KLane * KPack);
    //         int tempk = k % (KLane * KPack);
    //         int k1 = tempk / KPack;
    //         int k2 = tempk % KPack;

    //         int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
    //                         k1 * KPack * NLane + n1 * KPack + k2;

    //         dst[outputIndex] = src[idx];  // This is equivalent to src[n * K + k]
    //         // dst[outputIndex] = src[n * K + k];  // This is equivalent to src[n * K + k]
            
    //         srcIndices[idx] = idx;
    //         dstIndices[idx] = outputIndex;
    //         // if( n < 4 ){
    //         //     printf("GPU,%d,%d\n", outputIndex, idx);
    //         // }
    //     }
    // }

    // Calculate global 2D indices
    // int n = blockIdx.x * blockDim.x + threadIdx.x;
    // int k = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (n < N && k < K) {
        int n0 = n / NLane;
        int n1 = n % NLane;

        int k0 = k / (KLane * KPack);
        int tempk = k % (KLane * KPack);
        int k1 = tempk / KPack;
        int k2 = tempk % KPack;

        int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                        k1 * KPack * NLane + n1 * KPack + k2;

        // dst[outputIndex] = src[idx];  // This is equivalent to src[n * K + k]
        dst[outputIndex] = src[n * K + k];  // This is equivalent to src[n * K + k]
        
        srcIndices[n * K + k] = n * K + k;
        dstIndices[n * K + k] = outputIndex;
        // if( n < 4 ){
        //     printf("GPU,%d,%d\n", outputIndex, idx);
        // }
    }

}

template <typename Element>
static void prepackB_launcher(cudaStream_t stream, const Element* src, Element* dst, int N, int K, int NXdl) {
    // Define the number of threads per block
    const int threads_per_block = 256;

    // Calculate the number of blocks needed
    int total_elements = N * K;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    machete::preShuffleBufferKernel<Element><<<blocks, threads_per_block, 0, stream>>>(src, dst, N, K, NXdl);
}

template <typename Element>
static void prepackB_launcherDebug(cudaStream_t stream, const Element* src, Element* dst, int* srcIndices, int* dstIndices, int N, int K, int NXdl) {
    // // Define the number of threads per block
    // const int threads_per_block = 256;

    // // Calculate the number of blocks needed
    // int total_elements = N * K;
    // int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // // Launch the kernel
    // machete::preShuffleBufferKernelDebug<Element><<<blocks, threads_per_block, 0, stream>>>(src, dst, srcIndices, dstIndices, N, K, NXdl);
    
    // Define the number of threads per block in 2D
    const int threads_per_block_x = 16;
    const int threads_per_block_y = 16;
    dim3 threads_per_block(threads_per_block_x, threads_per_block_y);

    // Calculate the number of blocks needed in 2D
    int blocks_x = (K + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_y = (N + threads_per_block_y - 1) / threads_per_block_y;
    dim3 num_blocks(blocks_x, blocks_y);

    // Launch the kernel with 2D grid and block dimensions
    machete::preShuffleBufferKernelDebug<Element><<<num_blocks, threads_per_block, 0, stream>>>(src, dst, srcIndices, dstIndices, N, K, NXdl);
}

} // namespace machete


void prepackB(at::Tensor& B, at::Tensor& Bprepacked, const int NXdl=32) {

    TORCH_CHECK(
        (B.dtype() == at::kFloat8_e4m3fnuz) &&
            (Bprepacked.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");

    const int N = B.size(0);
    const int K = B.size(1);
    // const int NXdl = NXdl;
    
    auto stream = at::cuda::getCurrentHIPStream().stream();

    machete::prepackB_launcher(stream, 
        reinterpret_cast<FP8*>(B.data_ptr()), 
        reinterpret_cast<FP8*>(Bprepacked.data_ptr()), 
        N, K, NXdl
    );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

void prepackB_cpu(at::Tensor& B, at::Tensor& Bprepacked, const int NXdl=32) {

    TORCH_CHECK(
        (B.dtype() == at::kFloat8_e4m3fnuz) &&
            (Bprepacked.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");
        
    const int N = B.size(0);
    const int K = B.size(1);
    // const int NXdl = 32;

    auto B_ptr = reinterpret_cast<FP8*>(B.data_ptr());
    auto Bprepacked_ptr = reinterpret_cast<FP8*>(Bprepacked.data_ptr());
    machete::preShuffleBufferCPU(B_ptr, Bprepacked_ptr, N, K, NXdl);

}


void prepackBDebug(at::Tensor& B, at::Tensor& Bprepacked, at::Tensor& srcIndices, at::Tensor& dstIndices, const int NXdl=32) {

    TORCH_CHECK(
        (B.dtype() == at::kFloat8_e4m3fnuz) &&
            (Bprepacked.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");

    const int N = B.size(0);
    const int K = B.size(1);
    // const int NXdl = NXdl;
    
    auto stream = at::cuda::getCurrentHIPStream().stream();

    machete::prepackB_launcherDebug(stream, 
        reinterpret_cast<FP8*>(B.data_ptr()), 
        reinterpret_cast<FP8*>(Bprepacked.data_ptr()), 
        reinterpret_cast<int*>(srcIndices.data_ptr()), 
        reinterpret_cast<int*>(dstIndices.data_ptr()), 
        N, K, NXdl
    );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

void prepackB_cpuDebug(at::Tensor& B, at::Tensor& Bprepacked, at::Tensor& srcIndices, at::Tensor& dstIndices, const int NXdl=32) {

    TORCH_CHECK(
        (B.dtype() == at::kFloat8_e4m3fnuz) &&
            (Bprepacked.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");
        
    const int N = B.size(0);
    const int K = B.size(1);
    // const int NXdl = 32;

    auto B_ptr = reinterpret_cast<FP8*>(B.data_ptr());
    auto Bprepacked_ptr = reinterpret_cast<FP8*>(Bprepacked.data_ptr());
    machete::preShuffleBufferCPUDebug(
        B_ptr, 
        Bprepacked_ptr,
        reinterpret_cast<int*>(srcIndices.data_ptr()), 
        reinterpret_cast<int*>(dstIndices.data_ptr()),  
        N, 
        K, 
        NXdl);

}


template <typename InOutDataType>
void CompareFP8CPU(const InOutDataType* src, InOutDataType* dst, int* flag, int N, int K, float atol, float rtol)
{
    // this is a function to perform elementwise equality comparison for FP8 datatype
    // src and dst are tensors of (N, K) shape
    
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            // Calculate the difference between the tensors
            float diff = std::abs(static_cast<float>(src[n * K + k]) - static_cast<float>(dst[n * K + k]));

            // Determine the tolerance
            float tolerance = atol + rtol * std::abs(static_cast<float>(dst[n * K + k]));

            flag[n * K + k] = diff > tolerance;
            // // Check if the difference exceeds the tolerance
            // if (diff > tolerance)
            // {
            //     flag = 1;
            //     // // Handle the mismatch (e.g., log the mismatch, throw an error, etc.)
            //     // // For now, we'll just print the mismatch
            //     // std::cout << "Mismatch at (" << n << ", " << k << "): "
            //     //           << "src = " << static_cast<float>(src[n * K + k]) << ", "
            //     //           << "dst = " << static_cast<float>(dst[n * K + k]) << ", "
            //     //           << "diff = " << diff << ", "
            //     //           << "tolerance = " << tolerance << std::endl;
            // }
        }
    }
}

void compare_fp8_cpuDebug(at::Tensor& A, at::Tensor& B, at::Tensor& Flag, float atol, float rtol) {

    TORCH_CHECK(
        (A.dtype() == at::kFloat8_e4m3fnuz) &&
            (B.dtype() == at::kFloat8_e4m3fnuz),
        "Inputs must be type float8_e4m3fnuz.");
        
    const int N = B.size(0);
    const int K = B.size(1);
    // const int NXdl = 32;

    auto A_ptr = reinterpret_cast<FP8*>(A.data_ptr());
    auto B_ptr = reinterpret_cast<FP8*>(B.data_ptr());
    auto Flag_ptr = reinterpret_cast<int*>(Flag.data_ptr());
    CompareFP8CPU(
        A_ptr, 
        B_ptr,
        Flag_ptr,
        N,
        K,
        atol,
        rtol);

}