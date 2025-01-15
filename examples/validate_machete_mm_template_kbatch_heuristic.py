import time
import random
from collections import OrderedDict
from typing import List, Callable
import os
from functools import partial
import copy
import numpy as np
import argparse
import torch
import torch_ck

from typing import Tuple

from weight_shapes import WEIGHT_SHAPES

# M = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
# N = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
# K = [1024, 1280, 2048, 4096, 7168, 8192, 16384]

COMPUTE_FLOPS=True

def get_flops(ave_time, M, N, K):
    flop = 2 * M * N * K

    tflops = flop / 1e9 / ave_time

    return tflops


def get_bandwidth(ave_time, M, N, K, A_datatype_size=2, B_datatype_size=2, E_datatype_size=4):

    num_btype = A_datatype_size * M * K + B_datatype_size * K * N + E_datatype_size * M * N;

    gb_per_sec = num_btype / 1e6 / ave_time

    return gb_per_sec


MNK_List = [
    (3840, 64,64),
    (3840, 128,128),
    (3840, 256,256),
    (3840, 4096, 4096),
    (3840, 8192, 8192),
    (3840, 16384, 16384),
    (56, 8192, 7392),
    (32, 1024, 8192),
    (32, 1280, 8192),
    (32, 8192, 1024),
    (32, 7168, 8192),
    (32, 8192, 3584),
    (5000, 1280, 8192),
    (5000, 8192, 1024),
    (5000, 7168, 8192),
    (5000, 8192, 3584),
]

# Add UC interested weight shape
for b in range(7):

    B = 2**b
    mnk = [
        (B, 1280, 8192),
        (B, 8192, 1024),
        (B, 7168, 8192),
        (B, 8192, 3584),
        (B, 6144, 4096),
        (B, 4096, 4096),
        (B, 28672, 4096),
        (B, 4096, 14336),
        (B, 2560, 8192),
        (B, 8192, 2048),
        (B, 14336, 8192),
        (B, 8192, 7168),
        (B, 3072, 4096),
        (B, 4096, 2048),
        (B, 2560, 8192),
        (B, 14336, 4096),
        (B, 4096, 7168),
    ]

    MNK_List.extend(mnk)

# # add model related weight shape
# for b in range(7):

#     B = 2**b
#     for model_name, shape_list in WEIGHT_SHAPES.items():

#         for KN, tp_split_dim in shape_list:
#             for tp_size in [1, 2, 4, 8]:
#                 if KN[tp_split_dim] % tp_size == 0:
#                     KN[tp_split_dim] = KN[tp_split_dim] // tp_size
#                     K, N = KN
#                     MNK_List.append(
#                         (B, N, K)
#                     )

def getHeuristicSearchFunction(M, N, K):
    OpID = 32
    Kbatch = 2
    if M < 32 and N >= 4096 and K >= 1024:
        OpID = 33
        Kbatch = 2 // 2
    elif M >= 32 and N >= 4096 and K >= 1024:
        OpID = 32
        Kbatch = 2 // 2
    elif M >= 16 and N >= 8192 and K >= 2048:
        OpID = 25
        Kbatch = 8 // 2
    elif M >= 64 and N >= 8192 and K >= 4096:
        OpID = 26
        Kbatch = 8 // 2
    elif M >= 5000:
        OpID = 46
        Kbatch = 2 // 2
    elif N >= 14336 or K >= 14336:
        OpID = 64
        Kbatch = 16 // 2
    elif M >= 1 and N >= 1280 and K >= 8192:
        OpID = 33
        Kbatch = 16 // 2
    elif M >= 1 and N >= 2560 and K >= 8192:
        OpID = 26
        Kbatch = 16 // 2
    elif M >= 1 and N >= 7168 and K >= 8192:
        OpID = 33
        Kbatch = 16 // 2
    elif M >= 1 and N >= 14336 and K >= 8192:
        OpID = 64
        Kbatch = 8 // 2
    elif M >= 1 and N >= 4096 and K >= 14336:
        OpID = 25
        Kbatch = 16 // 2
    elif M >= 3840 and N >= 16384 and K >= 16384:
        OpID = 68
        Kbatch = 4
    else:
        OpID = 32
        Kbatch = 2 // 2
    
    return partial(call_kernel, opid=OpID, kbatch=Kbatch)          

def select_kernel_configurationGPT4o(M, N, K):
    """
    Select the best kernel configuration (OpID, Kbatch) based on M, N, and K dimensions.

    Parameters:
        M (int): Dimension M
        N (int): Dimension N
        K (int): Dimension K

    Returns:
        tuple: (OpID, Kbatch)
    """
    # Rule 1: Small M (M <= 16)
    if M <= 16:
        if N == 8192 and K == 1024:
            return 33, 2
        elif N == 8192 and K == 3584:
            return 32, 8
        elif N == 8192 and K == 7168:
            return 25, 8
        elif N == 14336 and K == 8192:
            return 64, 8
        elif N == 4096 and K == 14336:
            return 25, 16
        else:
            return 32, 8  # Default for small M

    # Rule 2: Medium M (16 < M <= 64)
    elif 16 < M <= 64:
        if N == 8192 and K == 1024:
            return 32, 2
        elif N == 8192 and K == 3584:
            return 25, 2
        elif N == 8192 and K == 7168:
            return 26, 8
        elif N == 14336 and K == 8192:
            return 64, 8
        elif N == 4096 and K == 14336:
            return 32, 16
        else:
            return 32, 4  # Default for medium M

    # Rule 3: Large M (M > 64)
    elif M > 64:
        if N == 8192 and K == 1024:
            return 46, 2
        elif N == 8192 and K == 3584:
            return 25, 2
        elif N == 8192 and K == 7168:
            return 64, 8
        elif N == 14336 and K == 8192:
            return 68, 8
        elif N == 16384 and K == 16384:
            return 68, 4
        else:
            return 46, 2  # Default for large M

    # Rule 4: High N and K (N, K >= 8192)
    if N >= 8192 and K >= 8192:
        if M <= 16:
            return 64, 8
        elif M <= 64:
            return 32, 8
        else:
            return 68, 4

    # Default case
    return 32, 8  # Default configuration if no specific rule matches

def getHeuristicSearchFunctionGPT4o(M, N, K):
    OpID, Kbatch = select_kernel_configurationGPT4o(M, N, K)
    return partial(call_kernel, opid=OpID, kbatch=Kbatch) 

def select_kernel_configSonnet(M, N, K):
    # Default values
    OpID = 32
    Kbatch = 2

    # Rule 1: When M is small (1-16)
    if 1 <= M <= 16:
        OpID = 32 if M > 8 else 33
        Kbatch = 16 if K >= 4096 else 2

    # Rule 2: When M is medium (32-64)
    elif 32 <= M <= 64:
        OpID = 25 if N >= 8192 else 32
        Kbatch = 8 if K >= 3584 else 2

    # Rule 3: When M is large (>1000)
    elif M > 1000:
        OpID = 46
        Kbatch = 2

    # Rule 4: When N is large (≥8192)
    if N >= 8192:
        OpID = 25 if OpID not in [46, 68] else OpID
        Kbatch = min(Kbatch, 8)

    # Rule 5: When K is small (≤1024)
    if K <= 1024:
        OpID = 33 if M <= 16 else 32
        Kbatch = 2

    # Rule 6: When K is large (≥4096)
    elif K >= 4096:
        if OpID not in [46, 68]:
            OpID = 64 if M >= 64 or N >= 14336 else 26
        Kbatch = max(Kbatch, 8)

    # Rule 7: For square matrices (M = N = K and large)
    if M == N == K and M >= 4096:
        OpID = 68 if M > 8192 else 46
        Kbatch = 4 if M > 8192 else 2

    # Rule 8: When all dimensions are large
    if M >= 64 and N >= 8192 and K >= 4096:
        OpID = 68
        Kbatch = 8

    return OpID, Kbatch

def getHeuristicSearchFunctionSonnet(M, N, K):
    OpID, Kbatch = select_kernel_configSonnet(M, N, K)
    return partial(call_kernel, opid=OpID, kbatch=Kbatch) 


def rand_data(shape, dtype=torch.float16, scale=1):
    return (scale * torch.rand(shape, device="cuda") - 0.3).to(dtype)


def call_kernel(
        A,
        B,
        a_scale,
        b_scale,
        output,
        opid,
        kbatch):

    return torch_ck.machete_mm_out(
        A,
        B,
        a_scale,
        b_scale,
        output,
        opid,
        kbatch
    )

KERNELS = [getHeuristicSearchFunction, getHeuristicSearchFunctionGPT4o, getHeuristicSearchFunctionSonnet]
NUM_OPS=82
# NUM_OPS=0
# for kbatch in [1, 2, 4, 8]:
#     for opid in range(0, NUM_OPS):
        
#         KERNELS.append(
#             partial(call_kernel, opid=opid+1, kbatch=kbatch)
#         )

@torch.inference_mode()
def main(save_file: str,
         seed: int = 0,
         do_profile: bool = False,
         num_warmup_iters: int = 5,
         num_iters: int = 100) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device("cuda")

    identity = torch.ones(1, dtype=torch.float32)

    def reference_scaled_mm_kernel(xq, wq, x_scale, w_scale, dummy1=None, dummy2=True, out_dtype=torch.float32):
        output = torch._scaled_mm(xq, wq, scale_a=identity, scale_b=identity, out_dtype=out_dtype)
        output = output * x_scale * w_scale.t()
        # output = output.to(dtype=torch.bfloat16)
        return output

    def real_reference_scaled_mm_kernel(xq, wq, x_scale, w_scale, dummy1=None, dummy2=True, out_dtype=torch.float32):
        output = torch._scaled_mm(xq, wq, scale_a=identity, scale_b=identity, out_dtype=torch.float32)
        output = output * x_scale * w_scale.t()
        output = output.to(dtype=torch.bfloat16)
        return output

    def run_cuda_benchmark(kernel: Callable,
                           xqs: List[torch.Tensor],
                           wqs: List[torch.Tensor],
                           x_scales: List[torch.Tensor],
                           w_scales: List[torch.Tensor],
                           num_warmup_iters: int, 
                           num_iters: int,
                           profile: bool = False) -> float:
        # shuffle data
        random.shuffle(xqs)
        random.shuffle(wqs)
        random.shuffle(x_scales)
        random.shuffle(w_scales)
        output = torch.zeros((xqs[0].size(0), wqs[0].size(0)), dtype=torch.float16).cuda()
        
        # warmup
        torch.cuda.synchronize()
        for i in range(num_warmup_iters):
            # _ = kernel(xqs[i], wqs[i], x_scales[i], w_scales[i], None, True, out_dtype=torch.bfloat16)
            _ = kernel(xqs[i], wqs[i], x_scales[i], w_scales[i], output)

        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        # start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        # end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_warmup_iters, num_warmup_iters + num_iters):
            # start_events[i-num_warmup_iters].record()
            # _ = kernel(xqs[i], wqs[i], x_scales[i], w_scales[i], None, True, out_dtype=torch.bfloat16)
            _ = kernel(xqs[i], wqs[i], x_scales[i], w_scales[i], output)
            # end_events[i-num_warmup_iters].record()
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters
        # times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # assert len(times) == num_iters
        # return float(np.mean(times) / 1000.0)


    benchmark_kernels = KERNELS

    if os.path.exists(save_file):
        os.remove(save_file)
    with open(save_file, "w") as f:
        f.write("dimension, " + ", ".join([(str( (opid % NUM_OPS ) + 1 ) + "k" + str( 2**((opid // NUM_OPS)) )) for opid, _ in enumerate(benchmark_kernels)]) + ", ref_scaled_mm" + "\n")

    # Benchmark
    dim_seq = []
    run_benchmark = run_cuda_benchmark
    x_dt = torch.float8_e4m3fnuz
    w_dt = torch.float8_e4m3fnuz
    scale_dt = torch.float32
    # for m in M:
    #     for n in N:
    #         for k in K:
    for m, n, k in MNK_List:
        print("===============================")
        dim_seq.append((m, n, k))
        xqs = [rand_data((m, k), x_dt, 5) for _ in range(num_iters + num_warmup_iters)]
        wqs = [rand_data((n, k), w_dt, 5) for _ in range(num_iters + num_warmup_iters)]
        x_scales = [rand_data((m, 1), scale_dt, 5) for _ in range(num_iters + num_warmup_iters)]
        w_scales = [rand_data((n, 1), scale_dt, 5) for _ in range(num_iters + num_warmup_iters)]

        latencies = []

        for opid, kernel in enumerate(benchmark_kernels):
            if do_profile:
                latency = run_benchmark(
                    kernel(m, n, k), xqs, wqs, x_scales, w_scales, 
                    num_warmup_iters=num_warmup_iters, num_iters=1, profile=True)
            else:
                
                try:
                    latency = run_benchmark(
                        kernel(m, n, k), xqs, wqs, x_scales, w_scales, 
                        num_warmup_iters=num_warmup_iters, num_iters=num_iters, profile=False)
                except Exception as e:
                    print(str(e))
                    torch.cuda.synchronize()
                    latencies.append(-1.0)
                    continue

            xq = torch.rand((m, k))
            xq = xq.to(dtype=torch.float8_e4m3fnuz)
            wq = torch.rand((n, k))
            wq = wq.to(dtype=torch.float8_e4m3fnuz)
            x_scale = torch.rand((m,1), dtype=torch.float32)
            w_scale = torch.rand((n,1), dtype=torch.float32)
            output = torch.zeros((m, n), dtype=torch.float16).cuda()

            # output = kernel(xq, wq, x_scale, w_scale, None, True, out_dtype=torch.bfloat16).t()
            try:
                output = kernel(m, n, k)(xq, wq, x_scale, w_scale, output)
            except Exception as e:
                print(str(e))
                continue

            torch.cuda.synchronize()

            try:
                ref_output = torch._scaled_mm(xq, wq.t(), scale_a=identity, scale_b=identity, out_dtype=torch.float32)
                ref_output = ref_output * x_scale * w_scale.t()
                ref_output = ref_output.to(dtype=torch.bfloat16)

                if not torch.allclose(output, ref_output,rtol=1e-3, atol=5e-8):
                    # print("!!!!!! [{}, {}, {}] - {} Failed correctness test".format(m, n, k, kernel.__name__))
                    latency = 1000.0
            except RuntimeError:
                pass

            # Not part of the validation
            latencies.append(latency)
            print("[{}, {}, {}] - {}: {:.3f}us".format(m, n, k, opid, latency * 1000000))

        # reference
        # wqs_t = [wq.t().contiguous() for wq in wqs]
        wqs_t = [wq.t() for wq in wqs]
        # try:
        latency = run_benchmark(
            reference_scaled_mm_kernel, xqs, wqs_t, x_scales, w_scales, 
            num_warmup_iters=num_warmup_iters, num_iters=1 if do_profile else num_iters, profile=do_profile)
        # except RuntimeError:
        #     latency = -1

        latencies.append(latency)
        print("[{}, {}, {}] - {}: {:.3f}us".format(m, n, k, "reference_scaled_mm_kernel", latency * 1000000))

        with open(save_file, "a") as f:
            # if COMPUTE_FLOPS:
            #     f.write("{}_{}_{}, ".format(m, n, k) 
            #         + ", ".join([str(l*1000000.0) for l in latencies]) + ", "
            #         + ", ".join([str(get_flops(l*1000.0, m, n, k)) if l > 0 else str(-1000000) for l in latencies]) + ", "
            #         + ", ".join([str(get_bandwidth(l*1000.0, m, n, k, 2, 2, 4)) if l > 0 else str(-1000000) for l in latencies])
            #         + "\n")
            # else:
            f.write("{}_{}_{}, ".format(m, n, k) + ", ".join([str(l*1000000.0) for l in latencies]) + "\n")



    # print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported dtype: {dt}")

    parser = argparse.ArgumentParser(
        description="Benchmark the ROCm fp8 scaled mm kernel.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters",
                        type=int,
                        default=200,
                        help="Number of benchmark iterations. "
                        "If --profile is set, this number is ignored")
    
    parser.add_argument("-o", "--output", type=str, 
                        default="benchmark_fp8_machete_mm_rocm_kbatch_heuristic.csv",
                        help="Path to write the results to")

    args = parser.parse_args()
    print(args)

    main(args.output,
         seed=args.seed,
         do_profile=args.profile,
         num_warmup_iters=args.num_warmup_iters,
         num_iters=args.num_iters)
    