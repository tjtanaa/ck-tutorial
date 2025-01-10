import itertools

import pytest
import torch
import torch_ck # needs to be imported after torch
from utils import assert_verbose_allclose
import torch.nn.functional as F


def check_if_run_test_case(K, NXdl):
    
    KPack = 16
    NLane = NXdl
    KLane = 64 / NLane

    K0 = K / (KLane * KPack)

    return (K % (KLane * KPack) == 0)

def compute_kpad(K, NXdl):
    # KPack = 16
    # NLane = NXdl
    # KLane = 64 / NLane

    # if (K % (KLane * KPack) == 0):
    #     return 0
    
    # return int(64 - (K % (KLane * KPack)))
    
    if (K % 256 == 0):
        return 0
    return int(256 - (K % 256))


def machete_mm_out(
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

MNK_List = [
    (56, 7392, 8192),
    # (3840, 64,64),
    (3840, 128,128),
    (3840, 4096, 4096),
    (3840, 8192, 8192),
    (3840, 16384, 16384),
    # (56, 8192, 7392),
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
# NXDL_PREPACK_VALUE=[32]
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

    if not check_if_run_test_case(K, nxdl_value):
            pytest.skip("Skipping this test due K not divisible by (KLane * KPack)")

    Bcpu = torch.rand((N, K), dtype=torch.float16).to("cpu")

    # Compute using the reference CPU implementation
    Bprepackcpu = torch.zeros((N, K), dtype=torch.float16).to("cpu")

    Bprepackcpu_output = Bprepackcpu.to(dtype)
    torch_ck.machete_prepack_B_cpu(Bcpu.to(dtype), Bprepackcpu_output, nxdl_value)


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
    Bprepackcuda_output = Bprepackcuda.to(dtype)
    torch_ck.machete_prepack_B(B, Bprepackcuda_output, nxdl_value)

    assert not torch.allclose(Bcpu, Bprepackcpu_output.to(torch.float16))

    # print(
    #     torch.amax(
    #         torch.abs(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20] - Bprepackcpu_output.to(torch.float16)[:20,:20])
    #     )
    # )
    # print(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20])

    assert_verbose_allclose(Bprepackcuda_output.to(torch.float16).clone().cpu(), Bprepackcpu_output.to(torch.float16))
    # torch.equal(Bprepackcuda_output.clone().cpu(), Bprepackcpu_output)

    # Flag = torch.zeros((N, K), dtype=torch.int)
    # torch_ck.compare_fp8_cpuDebug(Bprepackcuda_output.clone().cpu(), Bprepackcpu_output, Flag, 1e-3, 1e-5)

    # assert torch.sum(Flag) < 1

    # assert False
    

# @pytest.mark.parametrize("nxdl_value,mnk_shape,dtype,seed",
#                          itertools.product(NXDL_PREPACK_VALUE, MNK_List, DTYPES,
#                                            SEEDS))
# @torch.inference_mode()
# def test_machete_prepack_B_tensor_Debug(nxdl_value, mnk_shape, dtype, seed):

#     print(nxdl_value, mnk_shape, dtype, seed)
#     torch.manual_seed(seed)

#     _, N, K = mnk_shape
#     if not check_if_run_test_case(K, nxdl_value):
#             pytest.skip("Skipping this test due K not divisible by (KLane * KPack)")

#     # CPU Data
#     Bcpu = torch.rand((N, K), dtype=torch.float16).to("cpu")

#     # GPU Data
#     B = Bcpu.clone().cuda().to(dtype)

#     # Compute using the reference CPU implementation
#     Bprepackcpu = torch.zeros((N, K), dtype=torch.float16).to("cpu")
#     BprepackcpuSrcIndex = torch.zeros((N, K), dtype=torch.int).to("cpu")
#     BprepackcpuDstIndex = torch.zeros((N, K), dtype=torch.int).to("cpu")

#     Bprepackcpu_output = Bprepackcpu.to(dtype)
#     torch_ck.machete_prepack_B_cpuDebug(Bcpu.to(dtype), Bprepackcpu_output, BprepackcpuSrcIndex, BprepackcpuDstIndex, nxdl_value)


#     # print("Npad: ", 16 - (N % 16))
#     # print("Kpad: ", 16 - (K % 16))

#     # print("Npad32: ", 32 - (N % 32))
#     # print("Kpad32: ", 32 - (K % 32))

#     # print("Npad64: ", 64 - (N % 64))
#     # print("Kpad64: ", 64 - (K % 64))
#     # Kpad64 = (K % 64)
#     # Compute using GPU implementation
#     Bprepackcuda = torch.zeros((N, K), dtype=torch.float16)
#     BprepackcudaSrcIndex = torch.zeros((N, K), dtype=torch.int)
#     BprepackcudaDstIndex = torch.zeros((N, K), dtype=torch.int)
#     Bprepackcuda_output = Bprepackcuda.to(dtype)
#     torch_ck.machete_prepack_BDebug(B, Bprepackcuda_output, BprepackcudaSrcIndex, BprepackcudaDstIndex, nxdl_value)

#     # if Kpad64 > 0:
#     #     Bprepackcuda_output = Bprepackcuda_output[:, :K].contiguous()

#     assert not torch.allclose(Bcpu, Bprepackcpu_output.to(torch.float16))

#     # print(
#     #     torch.amax(
#     #         torch.abs(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20] - Bprepackcpu_output.to(torch.float16)[:20,:20])
#     #     )
#     # )
#     # print(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20])

#     # print(BprepackcudaSrcIndex.clone().cpu()[-10:,-45:])
#     # print(BprepackcudaDstIndex.clone().cpu()[-10:,-45:])
#     # print(BprepackcpuDstIndex[-10:,-45:])
#     # assert_verbose_allclose(BprepackcudaSrcIndex.clone().cpu(), BprepackcpuSrcIndex)
#     # assert_verbose_allclose(BprepackcudaDstIndex.clone().cpu(), BprepackcpuDstIndex)
#     # torch.equal(Bprepackcuda_output.clone().cpu(), Bprepackcpu_output)
#     # print(Bprepackcuda_output.clone().cpu()[:10,:10])
#     # print(Bprepackcpu_output[:10,:10])
    
#     # print(Bprepackcuda_output.clone().cpu()[-10:,-45:])
#     # print(Bprepackcpu_output[-10:,-45:])
#     assert_verbose_allclose(
#         Bprepackcuda_output.to(torch.float16).clone().cpu(), 
#         Bprepackcpu_output.to(torch.float16), 
#         max_print=200
#     )

#     # assert False
    

@pytest.mark.parametrize("nxdl_value,mnk_shape,dtype,seed",
                         itertools.product(NXDL_PREPACK_VALUE, MNK_List, DTYPES,
                                           SEEDS))
@torch.inference_mode()
def test_machete_mm(nxdl_value, mnk_shape, dtype, seed):

    print(nxdl_value, mnk_shape, dtype, seed)
    torch.manual_seed(seed)

    M, N, K = mnk_shape

    if not check_if_run_test_case(K, nxdl_value):
            pytest.skip("Skipping this test due K not divisible by (KLane * KPack)")

    Bcpu = torch.rand((N, K), dtype=torch.float16).to("cpu")

    # # Compute using the reference CPU implementation
    # Bprepackcpu = torch.zeros((N, K), dtype=torch.float16).to("cpu")

    # Bprepackcpu_output = Bprepackcpu.to(dtype)
    # torch_ck.machete_prepack_B_cpu(Bcpu.to(dtype), Bprepackcpu_output, nxdl_value)

    # Kpad64 = compute_kpad(K, nxdl_value)
    Kpad64 = 0
    # print("Kpad64: ", Kpad64)
    # Compute using GPU implementation
    B = Bcpu.cuda().to(dtype)
    Bprepackcuda = torch.zeros((N, K + Kpad64), dtype=torch.float16)
    Bprepackcuda_output = Bprepackcuda.to(dtype)
    torch_ck.machete_prepack_B(B, Bprepackcuda_output, nxdl_value)

    A = torch.rand((M, K), dtype=torch.float16).cuda().to(torch.float8_e4m3fnuz)

    macheteA = A
    # if Kpad64 > 0:
    #     # Pad zeros along the K dimension
    #     # The padding format is (padding_left, padding_right) for the last dimension
    #     # macheteA = F.pad(A, (0, Kpad64), "constant", 0)
    #     macheteA = torch.zeros((M, K + Kpad64), dtype=torch.float16).cuda().to(torch.float8_e4m3fnuz)
    #     macheteA[:M, :K] = A
    
    # print("macheteA.shape: ", macheteA.shape, "\tBprepackcuda_output.shape: ", Bprepackcuda_output.shape)
    # print(macheteA[M:, K:])


    a_scale = torch.rand((M, 1), dtype=torch.float32).cuda()
    b_scale = torch.rand((N, 1), dtype=torch.float32).cuda()
    machete_output = torch.zeros((M, N), dtype=torch.float16).cuda()
    machete_mm_out(
            macheteA,
            Bprepackcuda_output,
            a_scale,
            b_scale,
            machete_output,
            opid=31,
            kbatch=1
            # opid=10,
            # kbatch=1
            )

    scale_a_dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b_dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    print("A.shape" ,A.shape, A.dtype)
    print("B.shape", B.shape, B.dtype)
    scaled_mm_output = torch._scaled_mm(
        A.to(torch.float8_e4m3fnuz), B.t(), scale_a_dummy, scale_b_dummy, out_dtype=torch.float16
    )
    scaled_mm_output = scaled_mm_output * a_scale * b_scale.t()

    assert_verbose_allclose(
         machete_output, 
         scaled_mm_output,
        #  rtol=1.0,
        #  atol=0.1,
        #  rtol=0.05,
         rtol=1e-3,
         atol=5e-8,
        #  atol=1e-9,
         max_print=200)

    # assert not torch.allclose(Bcpu, Bprepackcpu_output.to(torch.float16))

    # print(
    #     torch.amax(
    #         torch.abs(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20] - Bprepackcpu_output.to(torch.float16)[:20,:20])
    #     )
    # )
    # print(Bprepackcuda_output.to(torch.float16).clone().cpu()[:20,:20])

    # assert_verbose_allclose(Bprepackcuda_output.to(torch.float16).clone().cpu(), Bprepackcpu_output.to(torch.float16))
