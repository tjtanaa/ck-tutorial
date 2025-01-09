import torch
import torch_ck

M_N_K_shape_list = [
    # (3840, 64,64),
    # (3840, 128,128),
    # (3840, 4096, 4096),
    # (3840, 8192, 8192),
    # (3840, 16384, 16384),
    (56, 8192, 7392)
]

for M, N, K in M_N_K_shape_list:

    Bcpu = torch.rand((N, K), dtype=torch.float16)
    Bprepackcpu = torch.zeros((N, K), dtype=torch.float16)

    Bprepackcpufp8 = Bprepackcpu.to(torch.float8_e4m3fnuz)
    torch_ck.machete_prepack_B_cpu(Bcpu.to(torch.float8_e4m3fnuz), Bprepackcpufp8)

    B = Bcpu.cuda().to(torch.float8_e4m3fnuz)
    Bprepack= Bprepackcpu.cuda().to(torch.float8_e4m3fnuz)
    Bprepack_before = Bprepack.to(torch.float16)
    torch_ck.machete_prepack_B(B, Bprepack)

    B_fp16 = B.to(torch.float16)
    Bprepack_fp16 = Bprepack.to(torch.float16)

    assert not torch.allclose(Bprepack_before, Bprepack_fp16)

    assert torch.allclose(Bprepack_fp16.clone().cpu(), Bprepackcpufp8.to(torch.float16))


    A = torch.rand((M, K), dtype=torch.float16).cuda().to(torch.float8_e4m3fnuz)
    a_scale = torch.rand((M, 1), dtype=torch.float32).cuda()
    b_scale = torch.rand((N, 1), dtype=torch.float32).cuda()
    output = torch.zeros((M, N), dtype=torch.float16).cuda()

    mm_output = torch_ck.machete_mm(
        A,
        B,
        a_scale,
        b_scale,
        output
    )
    scale_a_dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b_dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    print("A.shape" ,A.shape, A.dtype)
    print("B.shape", B.shape, B.dtype)
    scaled_mm_output = torch._scaled_mm(
        A, B.t(), scale_a_dummy, scale_b_dummy, out_dtype=torch.float32
    )
    scaled_mm_output = scaled_mm_output * a_scale * b_scale.t()

    print(mm_output[:5,:5])
    print(scaled_mm_output[:5,:5])

