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

# M = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
# N = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
# K = [1024, 1280, 2048, 4096, 7168, 8192, 16384]



def compute_kpad(K, NXdl=16):
    KPack = 16
    NLane = NXdl
    KLane = 64 / NLane

    if (K % (KLane * KPack) == 0):
        return 0
    
    return int(64 - (K % (KLane * KPack)))

def create_parameter_config(case_id=0):

    MNK_List = []

    if case_id == 0:
        MNK_List = [
            # [3840, 64,64],
            # [3840, 128,128],
            # [3840, 256,256],
            [3840, 2048, 2048],
            [3840, 4096, 4096],
            [3840, 8192, 8192],
            [3840, 16384, 16384],
            [56, 8192, 7392],
            [32, 1024, 8192],
            [32, 1280, 8192],
            [32, 8192, 1024],
            [32, 7168, 8192],
            [32, 8192, 3584],
            [5000, 1280, 8192],
            [5000, 8192, 1024],
            [5000, 7168, 8192],
            [5000, 8192, 3584],
        ]

    elif case_id == 1:
        # Add UC interested weight shape
        for b in range(args.batch_exponent_start, args.batch_exponent_end):

            B = 2**b
            mnk = [
                [B, 1280, 8192],
                [B, 8192, 1024],
                [B, 7168, 8192],
                # [B, 8192, 3584],
                [B, 6144, 4096],
                [B, 4096, 4096],
                [B, 28672, 4096],
                # [B, 4096, 14336], # opid 889 has problem
                [B, 2560, 8192],
                [B, 8192, 2048],
                [B, 14336, 8192],
                # [B, 8192, 7168],
                [B, 3072, 4096],
                [B, 4096, 2048],
                [B, 14336, 4096],
                # [B, 4096, 7168],
            ]

            MNK_List.extend(mnk)

    elif case_id == 2: # contains weird shape that model related
        
        # Add UC interested weight shape
        for b in range(args.batch_exponent_start, args.batch_exponent_end):

            B = 2**b
            mnk = [
                [B, 8192, 3584],
                [B, 4096, 14336], # opid 889 has problem
                [B, 8192, 7168],
                [B, 4096, 7168],
            ]

            MNK_List.extend(mnk)

    elif case_id == 3:
        # add model related weight shape
        for b in range(args.batch_exponent_start, args.batch_exponent_end):
            B = 2**b
            for model_name, shape_list in WEIGHT_SHAPES.items():

                for KN, tp_split_dim in shape_list:
                    for tp_size in [1, 2, 4, 8]:
                        if KN[tp_split_dim] % tp_size == 0:
                            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
                            K, N = KN
                            MNK_List.append(
                                [B, N, K]
                            )

    if args.pad:
        for i in range(len(MNK_List)):
            padK = compute_kpad(MNK_List[i][2], args.nxdl_value)
            MNK_List[i][2] = MNK_List[i][2] + padK

    return MNK_List


def main(args):

    MNK_List = create_parameter_config(case_id = args.case_id)

    with open(args.output, "w") as f:
        f.write(
            json.dumps(
                {
                    "MNK_List": MNK_List,
                    "padded": args.pad,
                    "NXDL": args.nxdl_value,
                },
                indent=4
            )
        )


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported dtype: {dt}")

    parser = argparse.ArgumentParser(
        description="Generate the M,N,K Shape into JSON file.")
    parser.add_argument("--case_id", type=int, default=1)
    parser.add_argument("-bes", "--batch_exponent_start", type=int, default=0)
    parser.add_argument("-bee", "--batch_exponent_end", type=int, default=15)
    parser.add_argument("-nxdl", "--nxdl_value", type=int, default=16)
    parser.add_argument("--pad", action='store_true')
    
    
    parser.add_argument("-o", "--output", type=str, 
                        default="MNK_ShapeForBenchmark.json",
                        help="Path to write the results to")

    args = parser.parse_args()
    print(args)

    main(args)
    