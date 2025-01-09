import torch
import torch_ck

M_N_K_shape_list = [
    # (3840, 64,64),
    # (3840, 128,128),
    # (3840, 4096, 4096),
    # (3840, 8192, 8192),
    # (3840, 16384, 16384),
    # (56, 8192, 7392),
    (32, 8192, 1024)
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

    # mm_output = torch_ck.machete_mm(
    #     A,
    #     B,
    #     a_scale,
    #     b_scale,
    # )
    mm_output = torch_ck.machete_mm_out(
        A,
        B,
        a_scale,
        b_scale,
        output,
    )
    print(mm_output.shape)

    assert mm_output.shape[0] == M
    assert mm_output.shape[1] == N

    print(mm_output[:10,:10])

