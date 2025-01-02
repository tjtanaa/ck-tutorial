#include <torch/types.h>
#include <torch/torch.h>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/literals.hpp"

#include <bmm.h>

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using I8   = int8_t;
using F32  = float;
using I32   = int32_t;

struct ScaleScaleAddRelu {
    template <typename T1, typename T2, typename T3>
    __host__ __device__ constexpr void
    operator()(T1& e, const T2& c, const T3& d) const;

    F32 alpha;
    F32 beta;
};

template <>
__host__ __device__ constexpr void
ScaleScaleAddRelu::operator()<I8, I32, I8>(I8& e, const I32& c, const I8& d) const
{
    // Scale AxB result with alpha
    const F32 c_scale = ck::type_convert<F32>(c) * alpha;

    // Scale D with beta
    const F32 d_scale = ck::type_convert<F32>(d) * beta;

    // Perform addition operation
    F32 temp = c_scale + d_scale;
    
    // Perform RELU operation
    temp = temp > 0 ? temp : 0;

    // Perform rounding operation 
    temp = temp > 127 ? 127 : temp;
    
    // Return to E
    e = ck::type_convert<I8>(temp);
};

// Function input and output 
torch::Tensor linear_relu_abde_i8(
			torch::Tensor A_,
			torch::Tensor B_,
			torch::Tensor D_,
			float alpha,
			float beta)
{
    // Convert torch::Tensor A_ (M, K) to torch::Tensor A (1, M, K) 
    auto A = A_.unsqueeze(0);

    // Convert torch::Tensor B_ (K, N) to torch::Tensor A (1, K, N) 
    auto B = B_.unsqueeze(0);

    // Return the batch count from the size of dimension 0
    int batch_count = A.size(0);

    // Return the M, N, K from the size of dimension 1 & 2
    int M = A.size(1);
    int N = B.size(1);
    int K = A.size(2);

    // Initialize the stride size for A, B, D and E
    int stride_A = K;
    int stride_B = K;
    int stride_D0 = N;
    int stride_E = N;

    // Initialize the stride size for batched A, B, D and E
    long long int batch_stride_A = M * K;
    long long int batch_stride_B = K * N;
    long long int batch_stride_D0 = M * N;
    long long int batch_stride_E = M * N;

    // // Initialize the stride size for batched A, B, D and E
    // int batch_stride_A = M * K;
    // int batch_stride_B = K * N;
    // int batch_stride_D0 = M * N;
    // int batch_stride_E = M * N;

    // Convert the tensor of 2-D to 3-D	
    auto D = D_.view({1,-1}).repeat({M, 1});

    // Allocate memory for E
    auto E = torch::empty(	{batch_count, M, N}, 
                torch::dtype(torch::kInt8).device(A.device()));
                // Data precision 
    using ADataType        = I8;
    using BDataType        = I8;
    using AccDataType      = I32;
    using CShuffleDataType = I32;
    using D0DataType 	      = I8;
    using DsDataType       = ck::Tuple<D0DataType>;
    using EDataType        = I8;
    // Specify tensor order
    using ALayout  = RowMajor;
    using BLayout  = ColumnMajor;
    using D0Layout = RowMajor;
    using DsLayout = ck::Tuple<D0Layout>;
    using ELayout  = RowMajor;  
    
    // No operations bound to the elements of A and B 
    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;

    // Operations bound to the elements of C, D and E
    using CDEElementOp = ScaleScaleAddRelu;


    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl< 
        // Tensor layout
        ALayout, BLayout, DsLayout, ELayout, 
        // Tensor data type
        ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  
        // Tensor operation
        AElementOp,  BElementOp, CDEElementOp,  
        // Padding strategy  
        GemmDefault,
        // Tunable parameters 
        1,   
        256,   
        256,   
        128,    
        64,  
        16,  
        16,   
        32,   
        32,    
        4,    
        2,     
        S<4, 64, 1>,     
        S<1, 0, 2>,     
        S<1, 0, 2>,              
        2,             
        16,             
        16,         
        1,     
        S<4, 64, 1>,     
        S<1, 0, 2>,     
        S<1, 0, 2>,             
        2,             
        16,             
        16,         
        1,           
        1,           
        1,               
        S<1, 64, 1, 4>,              
        16>;

    auto A_ref = A.data_ptr<ADataType>();
    auto B_ref = B.data_ptr<BDataType>();
    auto D0_ref = D.data_ptr<D0DataType>();
    auto E_ref = E.data_ptr<EDataType>();

    auto device_op    = DeviceOpInstance{};
    auto invoker = device_op.MakeInvoker();
    auto argument = device_op.MakeArgument(
            A_ref, B_ref, {D0_ref}, E_ref,
            M, N, K,
            batch_count,
            stride_A,	stride_B,	{stride_D0}, stride_E,
            batch_stride_A, batch_stride_B, {batch_stride_D0}, batch_stride_E,
            AElementOp{}, BElementOp{}, CDEElementOp{alpha, beta});

    invoker.Run(argument, StreamConfig{nullptr, 0});

    // Convert (1, M, N) to (M, N) 
    return E.squeeze(0);
}