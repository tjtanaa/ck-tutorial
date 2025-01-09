#include "machete_common.h"

//  M	N	K	Latency (us),KBatch,TFlops,GB/s,Kernel Instance
// 32	1280	8192	12.528, 4 ,53.567,864.445,"DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x64x256, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 16x16, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, BlkGemmPipelinePrefetchStages: 2"


at::Tensor machete_256_16x64x256_16x16_1x1_16x16_intrawave_v3_2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y
){
//   // A kernel that seems to work well on mid sized tensors.

//   // Check if this input needs to be padded.
//   int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
//   int N = WQ.size(0);
//   int K = WQ.size(1);
//   bool pad = (K % 128 != 0);

    // using DeviceGemmInstance = DeviceGemmHelper<
    //     256,
    //     16, 64, 256,
    //     16, 16,
    //     1, 1,
    //     S<16, 16, 1>,
    //     S<16, 16, 1>,
    //     S<1, 16, 1, 16>,
    //     S<8, 8, 1>,
    //     1,
    //     1,
    //     ck::BlockGemmPipelineScheduler::Intrawave,
    //     ck::BlockGemmPipelineVersion::v3,
    //     ck::tensor_operation::device::GemmSpecialization::Default,
    //     16,
    //     16>;
    // // Run kernel instance.
    // return f8f8bf16_pshuffleb_rowwise_impl<DeviceGemmInstance>(
    //     XQ, WQ, x_scale, w_scale, Y);
    
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        16, 64, 256,
        16, 16,
        1, 1,
        S<16, 16, 1>,
        S<16, 16, 1>,
        S<1, 16, 1, 16>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default,
        16,
        16>;
    // Run kernel instance.
    return f8f8bf16_pshuffleb_rowwise_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);


}