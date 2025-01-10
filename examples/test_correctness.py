import itertools

import pytest
import torch
import torch_ck # needs to be imported after torch
from utils import assert_verbose_allclose

MNK_List = [
    (3840, 64,64),
    (3840, 128,128),
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

NXDL_PREPACK_VALUE=[16, 32]
# NXDL_PREPACK_VALUE=[16]

DTYPES=[torch.float8_e4m3fnuz]
SEEDS = [0]


# Skip all tests if CUDA is not available
pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


# fail one test case 
# HIP_VISIBLE_DEVICES=6 pytest -c 1 examples/test_correctness.py::test_machete_prepack_B_tensor[16-mnk_shape5-dtype5-0]
@pytest.mark.parametrize("nxdl_value,mnk_shape,dtype,seed",
                         itertools.product(NXDL_PREPACK_VALUE, MNK_List, DTYPES,
                                           SEEDS))
@torch.inference_mode()
def test_machete_prepack_B_tensor(nxdl_value, mnk_shape, dtype, seed):

    print(nxdl_value, mnk_shape, dtype, seed)
    torch.manual_seed(seed)

    _, N, K = mnk_shape

    Bcpu = torch.rand((N, K), dtype=torch.float16).to("cpu")

    # Compute using the reference CPU implementation
    Bprepackcpu = torch.zeros((N, K), dtype=torch.float16).to("cpu")

    Bprepackcpu_output = Bprepackcpu.to(dtype)
    torch_ck.machete_prepack_B_cpu(Bcpu.to(dtype), Bprepackcpu_output, nxdl_value)


    print("Npad: ", 16 - (N % 16))
    print("Kpad: ", 16 - (K % 16))

    print("Npad32: ", 32 - (N % 32))
    print("Kpad32: ", 32 - (K % 32))

    print("Npad64: ", 64 - (N % 64))
    print("Kpad64: ", 64 - (K % 64))
    Kpad64 = (K % 64)
    # Compute using GPU implementation
    B = Bcpu.cuda().to(dtype)
    Bprepackcuda = torch.zeros((N, K + Kpad64), dtype=torch.float16)
    Bprepackcuda_output = Bprepackcuda.to(dtype)
    torch_ck.machete_prepack_B(B, Bprepackcuda_output, nxdl_value)

    if Kpad64 > 0:
        Bprepackcuda_output = Bprepackcuda_output[:, :K].contiguous()

    assert not torch.allclose(Bcpu, Bprepackcpu_output.to(torch.float16))

    # print(
    #     torch.amax(
    #         torch.abs(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20] - Bprepackcpu_output.to(torch.float16)[:20,:20])
    #     )
    # )
    # print(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20])

    assert_verbose_allclose(Bprepackcuda_output.to(torch.float16).clone().cpu(), Bprepackcpu_output.to(torch.float16))
    

@pytest.mark.parametrize("nxdl_value,mnk_shape,dtype,seed",
                         itertools.product(NXDL_PREPACK_VALUE, MNK_List, DTYPES,
                                           SEEDS))
@torch.inference_mode()
def test_machete_prepack_B_tensor_Debug(nxdl_value, mnk_shape, dtype, seed):

    print(nxdl_value, mnk_shape, dtype, seed)
    torch.manual_seed(seed)

    _, N, K = mnk_shape

    Bcpu = torch.rand((N, K), dtype=torch.float16).to("cpu")

    # Compute using the reference CPU implementation
    Bprepackcpu = torch.zeros((N, K), dtype=torch.float16).to("cpu")
    BprepackcpuSrcIndex = torch.zeros((N, K), dtype=torch.int).to("cpu")
    BprepackcpuDstIndex = torch.zeros((N, K), dtype=torch.int).to("cpu")

    Bprepackcpu_output = Bprepackcpu.to(dtype)
    torch_ck.machete_prepack_B_cpuDebug(Bcpu.to(dtype), Bprepackcpu_output, BprepackcpuSrcIndex, BprepackcpuDstIndex, nxdl_value)


    # print("Npad: ", 16 - (N % 16))
    # print("Kpad: ", 16 - (K % 16))

    # print("Npad32: ", 32 - (N % 32))
    # print("Kpad32: ", 32 - (K % 32))

    # print("Npad64: ", 64 - (N % 64))
    # print("Kpad64: ", 64 - (K % 64))
    # Kpad64 = (K % 64)
    # Compute using GPU implementation
    B = Bcpu.cuda().to(dtype)
    Bprepackcuda = torch.zeros((N, K), dtype=torch.float16)
    BprepackcudaSrcIndex = torch.zeros((N, K), dtype=torch.int)
    BprepackcudaDstIndex = torch.zeros((N, K), dtype=torch.int)
    Bprepackcuda_output = Bprepackcuda.to(dtype)
    torch_ck.machete_prepack_BDebug(B, Bprepackcuda_output, BprepackcudaSrcIndex, BprepackcudaDstIndex, nxdl_value)

    # if Kpad64 > 0:
    #     Bprepackcuda_output = Bprepackcuda_output[:, :K].contiguous()

    assert not torch.allclose(Bcpu, Bprepackcpu_output.to(torch.float16))

    # print(
    #     torch.amax(
    #         torch.abs(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20] - Bprepackcpu_output.to(torch.float16)[:20,:20])
    #     )
    # )
    # print(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20])

    print(BprepackcudaSrcIndex.clone().cpu()[:20,:20])
    print(BprepackcudaDstIndex.clone().cpu()[:20,:20])
    assert_verbose_allclose(BprepackcudaSrcIndex.clone().cpu(), BprepackcpuSrcIndex)
    assert_verbose_allclose(BprepackcudaDstIndex.clone().cpu(), BprepackcpuDstIndex)
    # assert_verbose_allclose(Bprepackcuda_output.to(torch.float16).clone().cpu(), Bprepackcpu_output.to(torch.float16))
    

