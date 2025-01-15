# from enum import Enum, auto
# from typing import List, Tuple, Optional
# from pydantic import BaseModel, Field, model_validator
# import math

# class MfmaInstr(Enum):
#     mfma_f32_32x32x1xf32 = auto()
#     mfma_f32_16x16x1xf32 = auto()
#     mfma_f32_4x4x1xf32 = auto()
#     mfma_f32_32x32x2xf32 = auto()
#     mfma_f32_16x16x4xf32 = auto()
#     mfma_f32_32x32x4f16 = auto()
#     mfma_f32_16x16x4f16 = auto()
#     mfma_f32_4x4x4f16 = auto()
#     mfma_f32_32x32x8f16 = auto()
#     mfma_f32_16x16x16f16 = auto()
#     mfma_f32_32x32x8bf16_1k = auto()
#     mfma_f32_16x16x16bf16_1k = auto()
#     mfma_f32_32x32x4bf16 = auto()
#     mfma_f32_16x16x8bf16 = auto()
#     mfma_i32_32x32x8i8 = auto()
#     mfma_i32_16x16x16i8 = auto()
#     mfma_i32_32x32x16i8 = auto()
#     mfma_i32_16x16x32i8 = auto()
#     mfma_f64_16x16x4f64 = auto()
#     mfma_f32_32x32x16f8f8 = auto()
#     mfma_f32_16x16x32f8f8 = auto()
#     mfma_f32_32x32x16bf8bf8 = auto()
#     mfma_f32_16x16x32bf8bf8 = auto()
#     mfma_f32_32x32x16f8bf8 = auto()
#     mfma_f32_16x16x32f8bf8 = auto()
#     mfma_f32_32x32x16bf8f8 = auto()
#     mfma_f32_16x16x32bf8f8 = auto()

# class MfmaType(BaseModel):
#     group_size: int
#     num_groups_per_blk: int
#     num_regs_per_blk: int
#     num_threads_per_blk: int
#     wave_size: int
#     num_input_blks: int
#     num_output_blks: int
#     m_per_blk: int
#     n_per_blk: int
#     k_per_blk: int
#     is_k_reduction: bool

#     def run(self, a, b, reg_c):
#         pass  # Placeholder for the actual implementation

# class MfmaSelector(BaseModel):
#     base_type: type
#     MPerXdlops: int
#     NPerXdlops: int
#     additional_type: Optional[type] = None

#     @staticmethod
#     def GetMfma(base_type, MPerXdlops, NPerXdlops, additional_type=None):
#         if base_type == float and MPerXdlops == 16 and NPerXdlops == 16:
#             return MfmaInstr.mfma_f32_16x16x4xf32
#         # Add more conditions based on the C++ template specializations
#         return MfmaInstr.mfma_f32_32x32x1xf32  # Default case

#     @property
#     def selected_mfma(self):
#         return self.GetMfma(self.base_type, self.MPerXdlops, self.NPerXdlops, self.additional_type)

#     def GetKPerXdlops(self):
#         selected_mfma = self.selected_mfma
#         mfma_type = self.get_mfma_type(selected_mfma)
#         return (mfma_type.is_k_reduction and mfma_type.num_input_blks or 1) * mfma_type.k_per_blk

#     def GetK1PerXdlops(self):
#         selected_mfma = self.selected_mfma
#         mfma_type = self.get_mfma_type(selected_mfma)
#         return mfma_type.k_per_blk

#     def get_mfma_type(self, mfma_instr: MfmaInstr) -> MfmaType:
#         # Map MfmaInstr to MfmaType
#         if mfma_instr == MfmaInstr.mfma_f32_32x32x1xf32:
#             return MfmaType(
#                 group_size=4,
#                 num_groups_per_blk=4,
#                 num_regs_per_blk=16,
#                 num_threads_per_blk=32,
#                 wave_size=64,
#                 num_input_blks=2,
#                 num_output_blks=2,
#                 m_per_blk=32,
#                 n_per_blk=32,
#                 k_per_blk=1,
#                 is_k_reduction=False
#             )
#         # Add more mappings based on the C++ template specializations
#         return MfmaType(
#             group_size=4,
#             num_groups_per_blk=4,
#             num_regs_per_blk=16,
#             num_threads_per_blk=32,
#             wave_size=64,
#             num_input_blks=2,
#             num_output_blks=2,
#             m_per_blk=32,
#             n_per_blk=32,
#             k_per_blk=1,
#             is_k_reduction=False
#         )  # Default case

# class XdlopsGemm(BaseModel):
#     base_type: type
#     MPerXdlops: int
#     NPerXdlops: int
#     KPack: int
#     additional_type: Optional[type] = None
#     TransposeC: bool = False

#     mfma: MfmaSelector = Field(default_factory=lambda: MfmaSelector(base_type=float, MPerXdlops=16, NPerXdlops=16))
#     mfma_instr: MfmaType = Field(default_factory=lambda: MfmaType(
#         group_size=4,
#         num_groups_per_blk=4,
#         num_regs_per_blk=16,
#         num_threads_per_blk=32,
#         wave_size=64,
#         num_input_blks=2,
#         num_output_blks=2,
#         m_per_blk=32,
#         n_per_blk=32,
#         k_per_blk=1,
#         is_k_reduction=False
#     ))

#     KPerXdlops: int = Field(default_factory=lambda: 1)
#     K1PerXdlops: int = Field(default_factory=lambda: 1)
#     K0PerXdlops: int = Field(default_factory=lambda: 1)

#     def __init__(self, **data):
#         super().__init__(**data)
#         self.mfma = MfmaSelector(base_type=self.base_type, MPerXdlops=self.MPerXdlops, NPerXdlops=self.NPerXdlops, additional_type=self.additional_type)
#         self.mfma_instr = self.mfma.get_mfma_type(self.mfma.selected_mfma)
#         self.KPerXdlops = self.mfma.GetKPerXdlops()
#         self.K1PerXdlops = self.mfma.GetK1PerXdlops()
#         self.K0PerXdlops = self.KPerXdlops // self.K1PerXdlops

#     def GetNumBlks(self):
#         return self.mfma_instr.num_output_blks

#     def GetNumXdlops(self):
#         return self.MPerXdlops * self.NPerXdlops // (self.mfma_instr.m_per_blk * self.mfma_instr.n_per_blk * self.mfma_instr.num_output_blks)

#     def GetRegSizePerXdlops(self):
#         return self.MPerXdlops * self.NPerXdlops // self.mfma_instr.wave_size

#     def GetWaveSize(self):
#         return self.mfma_instr.wave_size

#     def Run(self, p_a_wave, p_b_wave, p_c_thread):
#         for k in range(self.KPack // self.mfma_instr.k_per_blk):
#             if not self.TransposeC:
#                 self.mfma_instr.run(p_a_wave[k], p_b_wave[k], p_c_thread)
#             else:
#                 self.mfma_instr.run(p_b_wave[k], p_a_wave[k], p_c_thread)

#     def GetLaneId(self):
#         return 0  # Placeholder for actual implementation

#     def GetBlkIdx(self):
#         laneId = self.GetLaneId()
#         blk_idx = (laneId // self.mfma_instr.num_threads_per_blk, laneId % self.mfma_instr.num_threads_per_blk)
#         return blk_idx

#     def CalculateAThreadOriginDataIndex(self):
#         laneId = self.GetLaneId()
#         blk_idx = self.GetBlkIdx()
#         blk_id, blk_td = blk_idx
#         if self.mfma_instr.is_k_reduction:
#             return (blk_id, blk_td)
#         else:
#             return (0, laneId)

#     def CalculateBThreadOriginDataIndex(self):
#         laneId = self.GetLaneId()
#         blk_idx = self.GetBlkIdx()
#         blk_id, blk_td = blk_idx
#         if self.mfma_instr.is_k_reduction:
#             return (blk_id, blk_td)
#         else:
#             return (0, laneId)

#     def GetBeginOfThreadBlk(self, xdlops_i, blk_i):
#         blk_idx = self.GetBlkIdx()
#         blk_id, blk_td = blk_idx
#         n_offset = blk_i * self.mfma_instr.n_per_blk + blk_td
#         m_offset = xdlops_i * self.mfma_instr.m_per_blk + blk_id * self.mfma_instr.group_size
#         return (n_offset, m_offset) if self.TransposeC else (m_offset, n_offset)

#     def GetBeginOfThreadBlk4D(self, xdlops_i, blk_i):
#         blk_idx = self.GetBlkIdx()
#         blk_id, blk_td = blk_idx
#         return (blk_td, 0, blk_id, 0) if self.TransposeC else (0, blk_id, 0, blk_td)

#     def GetCM0M1M2NThreadBlkLengths(self):
#         return (self.mfma_instr.num_groups_per_blk, 1, self.mfma_instr.group_size, 1)

# # Example usage
# if __name__ == "__main__":
#     xdlops_gemm = XdlopsGemm(base_type=float, MPerXdlops=16, NPerXdlops=16, KPack=8)
#     print(xdlops_gemm.GetNumBlks())
#     print(xdlops_gemm.GetNumXdlops())
#     print(xdlops_gemm.GetRegSizePerXdlops())
#     print(xdlops_gemm.GetWaveSize())


template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16f8f8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16f8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32f8f8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32f8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

__device__ static constexpr index_t GetNumXdlops()
{
    return MPerXdlops * NPerXdlops /
            (mfma_instr.m_per_blk * mfma_instr.n_per_blk * mfma_instr.num_output_blks);
}

16 * 16 * 1