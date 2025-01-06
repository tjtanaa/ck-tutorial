import torch
import torch_ck

N_K_shape_list = [
    (64,64),
    (128,128),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
    (8192, 7392),
]

for N, K in N_K_shape_list:
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
    # print(torch.max(B.to(torch.float16)))
    # print(torch.max(Bprepack.to(torch.float16)))

    assert not torch.allclose(Bprepack_before, Bprepack_fp16)

    assert torch.allclose(Bprepack_fp16.clone().cpu(), Bprepackcpufp8.to(torch.float16))
