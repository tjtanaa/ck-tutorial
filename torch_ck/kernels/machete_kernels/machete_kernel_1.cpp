// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "machete_common.h"

#include <iostream>
#include <stdexcept>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"


// using F8  = ck::f8_t;
// using F16 = ck::half_t;
// using F32 = float;

// using Row = ck::tensor_layout::gemm::RowMajor;
// using Col = ck::tensor_layout::gemm::ColumnMajor;

// template <ck::index_t... Is>
// using S = ck::Sequence<Is...>;

// using PassThrough      = ck::tensor_operation::element_wise::PassThrough;
// // using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

// static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
// static constexpr auto GemmKPadding   = ck::tensor_operation::device::GemmSpecialization::KPadding;
// static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
// static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// static constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
// static constexpr auto Interwave = ck::BlockGemmPipelineScheduler::Interwave;


// clang-format off
        //################################| ALayout| BLayout|         DsLayout| ELayout|AData| BData|          DsData| EData| AccData| Cshuffle|           A|           B|              C|          GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|                         Block-wiseGemm|               Block-wiseGemm|
        //################################|        |        |                 |        | Type|  Type|            Type|  Type|    Type|     Type| Elementwise| Elementwise|    Elementwise|Specialization|  Size| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|                               Pipeline|                     Pipeline|
        //################################|        |        |                 |        |     |      |                |      |        |         |   Operation|   Operation|      Operation|              |      |      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|                              Scheduler|                     Verision|
        //################################|        |        |                 |        |     |      |                |      |        |         |            |            |               |              |      |      |      |      |    |    |    |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |                                       |                             |

        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,   256,    128,   128,  16,  16,  32,   32,    8,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,   128,    128,   128,  16,  16,  32,   32,    4,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,    64,    128,   128,  16,  16,  32,   32,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,    32,    128,   128,  16,  16,  32,   32,    1,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // // N 256
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,   256,    256,   128,  16,  16,  32,   32,    8,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,   128,    256,   128,  16,  16,  32,   32,    4,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,    64,    256,   128,  16,  16,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,    32,    256,   128,  16,  16,  32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // // N 512
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,    64,    512,   128,  16,  16,  32,   32,    2,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>,
        // DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, F16,  F32,     F32,     PassThrough, PassThrough, MultiplyMultiply,    GemmSpec,   256,    32,    512,   128,  16,  16,  32,   32,    1,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, F8>
        // // clang-format on
        // >;


// using MacheteGemmInstances = std::tuple<
//     DeviceGemmHelper<256, 256, 256, 128, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1, ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 128, 128, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 64, 128, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 32, 128, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 256, 256, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 128, 256, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 64, 256, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 32, 256, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 64, 512, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>,
//     DeviceGemmHelper<256, 256, 32, 512, 16, 16, 32, 32, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, GemmDefault, 16, 16>
// >;

// // Create an instance of the tuple
// MacheteGemmInstances gemmInstances;

at::Tensor machete_mm_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    int op_id) {
    // Get input dimensions.
    // int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    // int N = WQ.size(0);
    // int K = WQ.size(1);

    // Retrieve the appropriate GemmInstance based on (M, N, K).
    try {
        if (op_id == 1) {
            //  M	N	K	Latency (us),KBatch,TFlops,GB/s,Kernel Instance
            // 32	8192	1024	8.423, 1 ,63.737,1062.02,"DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 32x128x256, WaveTile: 32x32, WaveMap: 1x1, VmemReadVec: 16x16, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, BlkGemmPipelinePrefetchStages: 2"
            // 32	8192	3584	17.501, 1 ,107.369,1714.15,"DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 32x128x256, WaveTile: 32x32, WaveMap: 1x1, VmemReadVec: 16x16, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, BlkGemmPipelinePrefetchStages: 2"
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,   128,    256,
                32,   32, 
                1, 1, 
                S<16, 16, 1>, 
                S<16, 16, 1>, 
                S<1, 16, 1, 16>, 
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);
        } else if (op_id == 2) {
            //  M	N	K	Latency (us),KBatch,TFlops,GB/s,Kernel Instance
            // 32	7168	8192	31.469, 1 ,119.422,1888.88,"DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 32x128x512, WaveTile: 32x32, WaveMap: 1x1, VmemReadVec: 16x16, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, BlkGemmPipelinePrefetchStages: 2"
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,   128,    512,
                32,   32, 
                1, 1, 
                S<16, 16, 1>, 
                S<16, 16, 1>, 
                S<1, 16, 1, 16>, 
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);
        } else if (op_id == 3) {
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                256,    128,   128,
                32,   32,    
                8,    1,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y); 
        } else if (op_id == 4){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                128,    128,   128,
                32,   32,    
                4,    1,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 5){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    128,   128,
                32,   32,    
                2,    1,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 6){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    128,   128,
                32,   32,    
                1,    1,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 7){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                256,    256,   128,
                32,   32,    
                8,    2,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 8){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                128,    256,   128,
                32,   32,    
                4,    2,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 9){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    256,   128,
                32,   32,    
                2,    2,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 10){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    256,   128,
                32,   32,    
                1,    2,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 11){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    512,   128,
                32,   32,    
                2,    4,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 12){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    512,   128,
                32,   32,    
                1,    4,
                S<8, 32, 1>, 
                S<8, 32, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 13){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                128,    128,   256,
                32,   32,    
                4,    1,
                S<16, 16, 1>,
                S<16, 16, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 14){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    128,   256,
                32,   32,    
                2,    1,
                S<16, 16, 1>,
                S<16, 16, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 15){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    128,   256,
                32,   32,    
                1,    1,
                S<16, 16, 1>,
                S<16, 16, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 16){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    128,   512,
                32,   32,    
                2,    1,
                S<32, 8, 1>, 
                S<32, 8, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 17){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    128,   512,
                32,   32,    
                1,    1,
                S<32, 8, 1>, 
                S<32, 8, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 18){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                128,    256,   256,
                32,   32,    
                4,    2,
                S<16, 16, 1>, 
                S<16, 16, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 19){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    256,   256,
                32,   32,    
                2,    2,
                S<16, 16, 1>, 
                S<16, 16, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 20){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    256,   256,
                32,   32,    
                1,    2,
                S<16, 16, 1>, 
                S<16, 16, 1>, 
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 21){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    256,   512,
                32,   32,    
                2,    2,
                S<32, 8, 1>, 
                S<32, 8, 1>,
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 22){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    256,   512,
                32,   32,    
                1,    2,
                S<32, 8, 1>, 
                S<32, 8, 1>,
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 23){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                64,    512,   256,
                32,   32,    
                2,    4,
                S<16, 16, 1>, 
                S<16, 16, 1>,
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 24){
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                32,    512,   256,
                32,   32,    
                1,    4,
                S<16, 16, 1>, 
                S<16, 16, 1>,
                S<1, 32, 1, 8>,     
                S<8, 8, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 25){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    64,   256,
                16,   16,    
                1,    1,
                S<16, 16, 1>, 
                S<16, 16, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 26){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    128,   256,
                16,   16,    
                1,    2,
                S<16, 16, 1>, 
                S<16, 16, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 27){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    256,   256,
                16,   16,    
                1,    4,
                S<16, 16, 1>, 
                S<16, 16, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else if (op_id == 28){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    512,   256,
                16,   16,    
                1,    8,
                S<16, 16, 1>, 
                S<16, 16, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }  else if (op_id == 29){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    64,   512,
                16,   16,    
                1,    1,
                S<32, 8, 1>, 
                S<32, 8, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }   else if (op_id == 30){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    128,   512,
                16,   16,    
                1,    2,
                S<32, 8, 1>, 
                S<32, 8, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        }   else if (op_id == 31){
            // 16 
            return f8f8bf16_pshuffleb_rowwise_impl<
                DeviceGemmHelper<256, 
                16,    256,   512,
                16,   16,    
                1,    4,
                S<32, 8, 1>, 
                S<32, 8, 1>,
                S<1, 16, 1, 16>,    
                S<4, 4, 1>,
                1, 
                1, 
                ck::BlockGemmPipelineScheduler::Intrawave, 
                ck::BlockGemmPipelineVersion::v3, 
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>>(XQ, WQ, x_scale, w_scale, Y);                     
        } else {
            throw std::runtime_error("Unsupported (M, N, K) combination");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}