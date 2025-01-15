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
import json

from typing import Tuple

from weight_shapes import WEIGHT_SHAPES

COMPUTE_FLOPS=True

def get_flops(ave_time, M, N, K):
    flop = 2 * M * N * K

    tflops = flop / 1e9 / ave_time

    return tflops


def get_bandwidth(ave_time, M, N, K, A_datatype_size=2, B_datatype_size=2, E_datatype_size=4):

    num_btype = A_datatype_size * M * K + B_datatype_size * K * N + E_datatype_size * M * N;

    gb_per_sec = num_btype / 1e6 / ave_time

    return gb_per_sec

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

KERNELS = []
NUM_OPS=296
# NUM_OPS=0
for kbatch in [1, 2, 4, 8]:
    for opid in range(0, NUM_OPS):
        
        KERNELS.append(
            partial(call_kernel, opid=opid+1, kbatch=kbatch)
        )

@torch.inference_mode()
def main(
        MNK_List: List[List[int]],
        save_file: str,
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
        f.write("dimension, " + ", ".join([(str( (opid % NUM_OPS ) + 1 ) + "k" + str( 2**((opid // NUM_OPS)) )) + "v" + str( (opid // NUM_OPS // 4) + 1) for opid, _ in enumerate(benchmark_kernels)]) + ", ref_scaled_mm" + "\n")

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
                    kernel, xqs, wqs, x_scales, w_scales, 
                    num_warmup_iters=num_warmup_iters, num_iters=1, profile=True)
            else:
                
                try:
                    latency = run_benchmark(
                        kernel, xqs, wqs, x_scales, w_scales, 
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
                output = kernel(xq, wq, x_scale, w_scale, output)
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
    parser.add_argument("--config", type=str, help="Config file generated by examples/generate_tune_parameter_list.py")
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters",
                        type=int,
                        default=200,
                        help="Number of benchmark iterations. "
                        "If --profile is set, this number is ignored")
    
    parser.add_argument("-o", "--output", type=str, 
                        default="benchmark_fp8_machete_mm_rocm_kbatch_manual.csv",
                        help="Path to write the results to")

    args = parser.parse_args()
    print(args)
    with open(args.config, "r") as f:
        config_json = json.load(f)

    print(config_json)

    main(
        config_json["MNK_List"],
        args.output,
         seed=args.seed,
         do_profile=args.profile,
         num_warmup_iters=args.num_warmup_iters,
         num_iters=args.num_iters)
    