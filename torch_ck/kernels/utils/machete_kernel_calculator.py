from enum import Enum, IntEnum
from typing import Tuple, List, Optional
from pydantic import BaseModel, PositiveInt, Field, model_validator
import math

from typing import List, Tuple
from pydantic import BaseModel
import math


def mfmaGetNumXdlops(MPerXdlops=16, NPerXdlops=16):
    # composable_kernel/include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp
    # template <>
    # struct mfma_type<MfmaInstr::mfma_f32_32x32x16bf8f8>
    # {
    #     static constexpr index_t group_size          = 4;
    #     static constexpr index_t num_groups_per_blk  = 4;
    #     static constexpr index_t num_regs_per_blk    = 16;
    #     static constexpr index_t num_threads_per_blk = 32;
    #     static constexpr index_t wave_size           = 64;
    #     static constexpr index_t num_input_blks      = 2;
    #     static constexpr index_t num_output_blks     = 1;
    #     static constexpr index_t m_per_blk           = 32;
    #     static constexpr index_t n_per_blk           = 32;
    #     static constexpr index_t k_per_blk           = 8;
    #     static constexpr bool is_k_reduction         = true;

    #     template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    #     __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    #     {
    #         intrin_mfma_f32_32x32x16bf8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    #     }
    # };

    # template <>
    # struct mfma_type<MfmaInstr::mfma_f32_16x16x32bf8f8>
    # {
    #     static constexpr index_t group_size          = 4;
    #     static constexpr index_t num_groups_per_blk  = 1;
    #     static constexpr index_t num_regs_per_blk    = 4;
    #     static constexpr index_t num_threads_per_blk = 16;
    #     static constexpr index_t wave_size           = 64;
    #     static constexpr index_t num_input_blks      = 4;
    #     static constexpr index_t num_output_blks     = 1;
    #     static constexpr index_t m_per_blk           = 16;
    #     static constexpr index_t n_per_blk           = 16;
    #     static constexpr index_t k_per_blk           = 8;
    #     static constexpr bool is_k_reduction         = true;

    #     template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    #     __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    #     {
    #         intrin_mfma_f32_16x16x32bf8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    #     }
    # };


    # return MPerXdlops * NPerXdlops / (mfma_instr.m_per_blk * mfma_instr.n_per_blk * mfma_instr.num_output_blks);

    if NPerXdlops == 16: 
        num_input_blks = 4
        return MPerXdlops * NPerXdlops / (16 * 16 * 1)
    elif NPerXdlops == 32:
        num_input_blks = 2
        #3 GetKPerXdlops = selected_mfma.num_input_blks * k_per_blk
        #  GetK1PerXdlops = k_per_blk
        return MPerXdlops * NPerXdlops / (32 * 32 * 1)
    else:
        raise ValueError(f"Not support NPerXdlops {NPerXdlops}")
    

class Layout(str, Enum):
      Row='Row'
      Col='Col'

class GemmSpecialization(str, Enum):
    # Gemm
    Default = 'Default'
    MPadding = 'MPadding'
    NPadding = 'NPadding'
    KPadding = 'KPadding'
    MNPadding = 'MNPadding'
    MKPadding = 'MKPadding'
    NKPadding = 'NKPadding'
    MNKPadding = 'MNKPadding'
    # Gemm + Gemm
    OPadding = 'OPadding'
    MOPadding = 'MOPadding'
    NOPadding = 'NOPadding'
    KOPadding = 'KOPadding'
    MNOPadding = 'MNOPadding'
    MKOPadding = 'MKOPadding'
    NKOPadding = 'NKOPadding'
    MNKOPadding = 'MNKOPadding'

class CKDataType(str, Enum):
    fp8 = 'ck::f8_t'
    bf8 = 'ck::bf8_t'
    bhalf_t = 'ck::bhalf_t'
    half_t = 'ck::half_t'
    float = 'float'

class BlockGemmPipelineScheduler(str, Enum):
    Intrawave = 'Intrawave'
    Interwave = 'Interwave'

class BlockGemmPipelineVersion(str, Enum):
    v1 = 'v1'
    v2 = 'v2'
    v3 = 'v3'
    v4 = 'v4'
    v5 = 'v5'

class ElementOpDummy(str, Enum):
    ElementOp = "ElementOp"

class DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle_config(BaseModel):
    ALayout: Layout = Field(description='ALayout')
    BLayout: Layout = Field(description='BLayout')
    DsLayout: Tuple[Layout, Layout] = Field(description='DsLayout')
    CLayout: Layout = Field(description='ELayout')
    ADataType: CKDataType = Field(description='AData Type')
    BDataType: CKDataType = Field(description='BData Type')
    DsDataType: Tuple[CKDataType, CKDataType] = Field(description='DsData Type')
    CDataType: CKDataType = Field(description='EData Type')
    GemmAccDataType: CKDataType = Field(description='AccData Type')
    CShuffleDataType: CKDataType = Field(description='Cshuffle Type')
    AElementwiseOperation: ElementOpDummy = Field(description='A Elementwise Operation')
    BElementwiseOperation: ElementOpDummy = Field(description='B Elementwise Operation')
    CElementwiseOperation: ElementOpDummy = Field(description='C Elementwise Operation')
    GemmSpec: GemmSpecialization = Field(description='GEMM Specialization')
    BlockSize: int = Field(description='Block Size')
    MPerBlock: int = Field(description='MPer Block')
    NPerBlock: int = Field(description='NPer Block')
    KPerBlock: int = Field(description='KPer Block')
    AK1: int = Field(description='AK1')
    BK1: int = Field(description='BK1')
    MPerXDL: int = Field(description='MPer XDL')
    NPerXDL: int = Field(description='NPer XDL')
    MXdlPerWave: int = Field(description='MXdl Per Wave')
    NXdlPerWave: int = Field(description='NXdl Per Wave')
    ABlockTransferThreadClusterLengths_AK0_M_AK1: List[int]
    ABlockTransferThreadClusterArrangeOrder: List[int]
    ABlockTransferSrcAccessOrder: List[int]
    ABlockTransferSrcVectorDim: int
    ABlockTransferSrcScalarPerVector: int
    ABlockTransferDstScalarPerVector_AK1: int
    ABlockLdsExtraM : bool
    BBlockTransferThreadClusterLengths_BK0_N_BK1: List[int]
    BBlockTransferThreadClusterArrangeOrder: List[int]
    BBlockTransferSrcAccessOrder: List[int]
    BBlockTransferSrcVectorDim: int
    BBlockTransferSrcScalarPerVector: int
    BBlockTransferDstScalarPerVector_BK1: int
    BBlockLdsExtraN: bool
    CShuffleMXdlPerWavePerShuffle: int
    CShuffleNXdlPerWavePerShuffle: int
    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock: List[int]
    CDEShuffleBlockTransferScalarPerVectors: List[int]
    BlkGemmPipeSched : BlockGemmPipelineScheduler  = Field(BlockGemmPipelineScheduler.Intrawave, description='') 
    BlkGemmPipelineVer : BlockGemmPipelineVersion = Field(BlockGemmPipelineVersion.v1, description='') 
    ComputeTypeA: Optional[CKDataType] = None
    ComputeTypeB: Optional[CKDataType] = None
    LDSTypeA: Optional[CKDataType] = None
    LDSTypeB: Optional[CKDataType] = None

    @model_validator(mode='before')
    def set_default_compute_types(cls, values):
        if 'CDataType' in values:
            c_data_type = values['CDataType']
            if 'ComputeTypeA' not in values:
                values['ComputeTypeA'] = c_data_type
            if 'ComputeTypeB' not in values:
                values['ComputeTypeB'] = values['ComputeTypeA']
            if 'LDSTypeA' not in values:
                values['LDSTypeA'] = values['ComputeTypeA']
            if 'LDSTypeB' not in values:
                values['LDSTypeB'] = values['ComputeTypeB']
        return values
    
class DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle:
    
    def __init__(
        self,
        config: DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle_config
        ):
        pass


class GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle_config(DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle_config):
    # ALayout: Layout = Field(description='ALayout')
    # BLayout: Layout = Field(description='BLayout')
    # DsLayout: Tuple[Layout, Layout] = Field(description='DsLayout')
    # CLayout: Layout = Field(description='ELayout')
    # ADataType: CKDataType = Field(description='AData Type')
    # BDataType: CKDataType = Field(description='BData Type')
    # DsDataType: Tuple[CKDataType, CKDataType] = Field(description='DsData Type')
    # CDataType: CKDataType = Field(description='EData Type')
    # GemmAccDataType: CKDataType = Field(description='AccData Type')
    # CShuffleDataType: CKDataType = Field(description='Cshuffle Type')
    # AElementwiseOperation: ElementOpDummy = Field(description='A Elementwise Operation')
    # BElementwiseOperation: ElementOpDummy = Field(description='B Elementwise Operation')
    # CElementwiseOperation: ElementOpDummy = Field(description='C Elementwise Operation')
    # GemmSpec: GemmSpecialization = Field(description='GEMM Specialization')
    # BlockSize: int = Field(description='Block Size')
    # MPerBlock: int = Field(description='MPer Block')
    # NPerBlock: int = Field(description='NPer Block')
    # KPerBlock: int = Field(description='KPer Block')
    # AK1: int = Field(description='AK1')
    # BK1: int = Field(description='BK1')
    # MPerXDL: int = Field(description='MPer XDL')
    # NPerXDL: int = Field(description='NPer XDL')
    # MXdlPerWave: int = Field(description='MXdl Per Wave')
    # NXdlPerWave: int = Field(description='NXdl Per Wave')
    # ABlockTransferThreadClusterLengths_AK0_M_AK1: List[int]
    # ABlockTransferThreadClusterArrangeOrder: List[int]
    # ABlockTransferSrcAccessOrder: List[int]
    # ABlockTransferSrcVectorDim: int
    # ABlockTransferSrcScalarPerVector: int
    # ABlockTransferDstScalarPerVector_AK1: int
    AThreadTransferSrcResetCoordinateAfterRun: bool
    # ABlockLdsExtraM : bool
    # BBlockTransferThreadClusterLengths_BK0_N_BK1: List[int]
    # BBlockTransferThreadClusterArrangeOrder: List[int]
    # BBlockTransferSrcAccessOrder: List[int]
    # BBlockTransferSrcVectorDim: int
    # BBlockTransferSrcScalarPerVector: int
    # BBlockTransferDstScalarPerVector_BK1: int
    BThreadTransferSrcResetCoordinateAfterRun: bool
    # BBlockLdsExtraN: bool
    # CShuffleMXdlPerWavePerShuffle: int
    # CShuffleNXdlPerWavePerShuffle: int
    # CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock: List[int]
    # CDEShuffleBlockTransferScalarPerVectors: List[int]
    # BlkGemmPipeSched : BlockGemmPipelineScheduler  = Field(BlockGemmPipelineScheduler.Intrawave, description='') 
    # BlkGemmPipelineVer : BlockGemmPipelineVersion = Field(BlockGemmPipelineVersion.v1, description='') 
    # ComputeTypeA: Optional[CKDataType] = None
    # ComputeTypeB: Optional[CKDataType] = None
    # LDSTypeA: Optional[CKDataType] = None
    # LDSTypeB: Optional[CKDataType] = None

    @model_validator(mode='before')
    def set_default_compute_types(cls, values):
        if 'CDataType' in values:
            c_data_type = values['CDataType']
            if 'ComputeTypeA' not in values:
                values['ComputeTypeA'] = c_data_type
            if 'ComputeTypeB' not in values:
                values['ComputeTypeB'] = values['ComputeTypeA']
            if 'LDSTypeA' not in values:
                values['LDSTypeA'] = values['ComputeTypeA']
            if 'LDSTypeB' not in values:
                values['LDSTypeB'] = values['ComputeTypeB']
        return values


# Assuming Block2CTileMapDefault is a class with a static method CalculateGridSize
class Block2CTileMapDefault:
    @staticmethod
    def CalculateGridSize(M, N, MPerBlock, NPerBlock):
        # Placeholder implementation
        return (M // MPerBlock, N // NPerBlock)

class GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle:
    
    def __init__(
        self,
        config: GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle_config
        ):
        
        self.config = config

        self.CShuffleBlockTransferScalarPerVector_NPerBlock =\
            config.CDEShuffleBlockTransferScalarPerVectors[0]
        
        # K1 should be Number<...>
        self.AK0Number       = config.KPerBlock / config.AK1
        self.BK0Number       = config.KPerBlock / config.BK1
        self.AK1Number       = config.AK1
        self.BK1Number       = config.BK1
        self.BlockSizeNumber = config.BlockSize
        self.NumDTensor = len(config.DsDataType)

        # KPack = max(math.lcm(AK1Number, BK1Number), mfma_selector::selected_mfma.k_per_blk)
        k_per_blk = 8 
        KPack = max(math.lcm(config.AK1, config.BK1), k_per_blk)

        # KLane = mfma_selector::GetKPerXdlops() / mfma_selector::GetK1PerXdlops()
        # there are 64 lanes per workgroup
        # when we call the builtin_gcn we always perform 64/MPerXDL
        # E.g. mfma_f32_32x32x16bf8f8
        # num_input_blks = 64 / 32 = 2
        # E.g. mfma_f32_16x16x32bf8f8
        # num_input_blks = 64 / 16 = 4
        num_input_blks = 64 // config.NPerXDL
        KLane = num_input_blks 

        KRepeat = config.KPerBlock / KLane / KPack
        NLane   = config.NPerXDL
        NWave   = config.NPerBlock / config.NPerXDL / config.NXdlPerWave

        
        assert ((config.MXdlPerWave % config.CShuffleMXdlPerWavePerShuffle) == 0 and
                (config.NXdlPerWave % config.CShuffleNXdlPerWavePerShuffle) == 0), "wrong!"

        MWave = config.MPerBlock / (config.MXdlPerWave * config.MPerXDL)

        print(f"""
        config.AK1: {config.AK1}
        config.BK1: {config.BK1}
        KRepeat: {KRepeat}
        config.KPerBlock: {config.KPerBlock}
        KLane: {KLane}
        KPack: {KPack}

        NWave = config.NPerBlock / config.NPerXDL / config.NXdlPerWave
        NWave: {NWave}
        config.NPerBlock: {config.NPerBlock}
        config.NPerXDL: {config.NPerXDL}
        config.NXdlPerWave: {config.NXdlPerWave}
        
        MWave = config.MPerBlock / (config.MXdlPerWave * config.MPerXDL)
        MWave: {MWave}
        config.MPerBlock: {config.MPerBlock}
        config.MXdlPerWave: {config.MXdlPerWave}
        config.MPerXDL: {config.MPerXDL}
        (config.MXdlPerWave * config.MPerXDL): {(config.MXdlPerWave * config.MPerXDL)}
        
        """)

        # mi300 warpSize = 64
        warpSize = 64
        assert NWave * warpSize == config.BlockSize, f"NWave * warpSize == config.BlockSize [:] {NWave} * {warpSize} == {config.BlockSize}"

        self.NLane = NLane
        self.KPack = KPack
        self.KLane = KLane
        self.NWave = NWave
        self.MWave = MWave

    def CheckValidity(self, karg: "Problem") -> bool:
        # Static assertions (not directly translatable to Python, but we can use assertions)
        assert (self.config.MPerBlock % (self.config.MPerXDL * self.config.MXdlPerWave) == 0), "Invalid tuning param!"
        assert (self.config.NPerBlock % (self.config.NXdlPerWave * self.config.NPerXDL) == 0), "Invalid tuning param!"

        # Check MPerBlock condition
        if not (self.config.GemmSpec == GemmSpecialization.MPadding or
                self.config.GemmSpec == GemmSpecialization.MNPadding or
                self.config.GemmSpec == GemmSpecialization.MKPadding or
                self.config.GemmSpec == GemmSpecialization.MNKPadding) and \
                self.config.ALayout == Layout.Row:
            if karg.M % self.config.MPerBlock != 0:
                print(f"Arg M value is not a multiple of MPerBlock! M: {karg.M}")
                return False

        # Check NPerBlock condition
        if not (self.config.GemmSpec == GemmSpecialization.NPadding or
                self.config.GemmSpec == GemmSpecialization.MNPadding or
                self.config.GemmSpec == GemmSpecialization.NKPadding or
                self.config.GemmSpec == GemmSpecialization.MNKPadding) and \
                self.config.BLayout == Layout.Row:
            if karg.N % self.config.NPerBlock != 0:
                print(f"Arg N value is not a multiple of NPerBlock! N: {karg.N}")
                return False

        # Check KPerBlock condition
        if not (self.config.GemmSpec == GemmSpecialization.KPadding or
                self.config.GemmSpec == GemmSpecialization.MKPadding or
                self.config.GemmSpec == GemmSpecialization.NKPadding or
                self.config.GemmSpec == GemmSpecialization.MNKPadding):
            K_t = karg.KBatch * self.config.KPerBlock
            if karg.K % K_t != 0:
                print(f"Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: {karg.K}")
                return False
        else:
            KReadVec = math.lcm(self.config.AK1, self.config.BK1)
            K_t = karg.KBatch * KReadVec
            KReadPadSplited = (karg.K + K_t - 1) // K_t * KReadVec
            if (KReadPadSplited * (karg.KBatch - 1)) >= karg.K:
                return False

        # Check ABlockTransferSrcScalarPerVector condition
        if self.config.ALayout == Layout.Row:
            if karg.K % self.config.ABlockTransferSrcScalarPerVector != 0:
                print(f"Arg K ({karg.K}) value is not a multiple of ABlockTransferSrcScalarPerVector ({self.config.ABlockTransferSrcScalarPerVector})!")
                return False
        else:
            if karg.M % self.config.ABlockTransferSrcScalarPerVector != 0:
                print(f"Arg M ({karg.M}) value is not a multiple of ABlockTransferSrcScalarPerVector ({self.config.ABlockTransferSrcScalarPerVector})!")
                return False

        # Check BBlockTransferSrcScalarPerVector condition
        if self.config.BLayout == Layout.Row:
            if karg.N % self.config.BBlockTransferSrcScalarPerVector != 0:
                print(f"Arg N ({karg.N}) value is not a multiple of BBlockTransferSrcScalarPerVector ({self.config.BBlockTransferSrcScalarPerVector})!")
                return False
        else:
            if karg.K % self.config.BBlockTransferSrcScalarPerVector != 0:
                print(f"Arg K ({karg.K}) value is not a multiple of BBlockTransferSrcScalarPerVector ({self.config.BBlockTransferSrcScalarPerVector})!")
                return False

        # Check CShuffleBlockTransferScalarPerVector_NPerBlock condition
        if self.config.CLayout == Layout.Row:
            if karg.N % self.config.CDEShuffleBlockTransferScalarPerVectors[0] != 0:
                print(f"Arg N ({karg.N}) value is not a multiple of CShuffleBlockTransferScalarPerVector_NPerBlock ({self.config.CDEShuffleBlockTransferScalarPerVectors[0]})!")
                return False
        else:
            if karg.M % self.config.CDEShuffleBlockTransferScalarPerVectors[0] != 0:
                print(f"Arg M ({karg.M}) value is not a multiple of CShuffleBlockTransferScalarPerVector_NPerBlock ({self.config.CDEShuffleBlockTransferScalarPerVectors[0]})!")
                return False

        # Check gridwise gemm pipeline
        num_k_loop = karg.AK0 / (self.config.KPerBlock / self.config.AK1)
        # if num_k_loop <= self.config.BlkGemmPipeSched.PrefetchStages:
        # hardcode self.config.BlkGemmPipeSched.PrefetchStages to 2 first
        if num_k_loop <= 2:
            return False

        return True

        
    def integer_least_multiple(self, a, b):
        return (a + b - 1) // b * b

    def integer_divide_ceil(self, a, b):
        return (a + b - 1) // b

    def CalculateGridSize(self, M, N, KBatch):
        grid_size = Block2CTileMapDefault.CalculateGridSize(M, N, self.config.MPerBlock, self.config.NPerBlock)
        return (grid_size, 1, KBatch)

    def CalculateMPadded(self, M):
        return self.integer_least_multiple(M, self.config.MPerBlock)

    def CalculateNPadded(self, N):
        return self.integer_least_multiple(N, self.config.NPerBlock)

    def CalculateBN0Shuffled(self, N):
        return self.integer_divide_ceil(N, self.NLane)

    def CalculateBK0Shuffled(self, K):
        return self.integer_divide_ceil(K, self.KLane * self.KPack)

    def CalculateKPadded(self, K):
        return self.integer_divide_ceil(K, self.config.KPerBlock) * self.config.KPerBlock

    def CalculateAK0Padded(self, K, K_Batch=1):
        K_t = K_Batch * self.config.KPerBlock
        return (K + K_t - 1) // K_t * (self.config.KPerBlock // self.config.AK1)

    def CalculateBK0Padded(self, K, K_Batch=1):
        K_t = K_Batch * self.config.KPerBlock
        return (K + K_t - 1) // K_t * (self.config.KPerBlock // self.config.BK1)

    def CalculateKPadded(self, K, K_Batch=1):
        K_t = K_Batch * self.config.KPerBlock
        return (K + K_t - 1) // K_t * self.config.KPerBlock

    def CalculateKRead(self, K, K_Batch=1):
        KReadVec = math.lcm(self.config.AK1, self.config.BK1)
        K_t = K_Batch * KReadVec
        return (K + K_t - 1) // K_t * KReadVec

    def CalculateMBlock(self, M):
        return self.integer_divide_ceil(M, self.config.MPerBlock)

    def CalculateNBlock(self, N):
        return self.integer_divide_ceil(N, self.config.NPerBlock)

class Problem:
    kernel: GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle
    M: int
    N: int
    K: int
    StrideA: int
    StrideB: int
    StrideDs: List[int]
    StrideC: int
    KBatch: int
    MPadded: int = 0
    NPadded: int = 0
    KRead: int = 0
    KPadded: int = 0
    AK0: int = 0
    BK0: int = 0
    MBlock: int = 0
    NBlock: int = 0
    BN0Shuffled: int = 0
    BK0Shuffled: int = 0

    def __init__(self, 
                 kernel : GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle,
                 M: int, N: int, K: int, 
                 StrideA: int, StrideB: int, StrideDs: List[int], StrideC: int, 
                 KBatch: int):
        # super().__init__(
        #     kernel=kernel,
        #     M=M, N=N, K=K, 
        #     StrideA=StrideA, 
        #     StrideB=StrideB, 
        #     StrideDs=StrideDs, 
        #     StrideC=StrideC, 
        #     KBatch=KBatch)
        self.kernel=kernel
        self.M=M
        self.N=N
        self.K=K
        self.StrideA=StrideA
        self.StrideB=StrideB
        self.StrideDs=StrideDs
        self.StrideC=StrideC
        self.KBatch=KBatch
        self.NLane = self.kernel.config.NPerXDL
        self.MPadded = self.calculate_mpadded(self.M)
        self.NPadded = self.calculate_npadded(self.N)
        self.KRead = self.calculate_kread(self.K, self.KBatch)
        self.KPadded = self.calculate_kpadded(self.K, self.KBatch)
        self.AK0 = self.calculate_ak0padded(self.K, self.KBatch)
        self.BK0 = self.calculate_bk0padded(self.K, self.KBatch)
        self.MBlock = self.calculate_mblock(self.M)
        self.NBlock = self.calculate_nblock(self.N)
        self.BN0Shuffled = self.calculate_bn0shuffled(self.N)
        self.BK0Shuffled = self.calculate_bk0shuffled(self.K)


    def calculate_mpadded(self, M: int) -> int:
        return math.ceil(M / self.kernel.config.MPerBlock) * self.kernel.config.MPerBlock

    def calculate_npadded(self, N: int) -> int:
        return math.ceil(N / self.kernel.config.NPerBlock) * self.kernel.config.NPerBlock

    def calculate_kread(self, K: int, KBatch: int) -> int:
        KReadVec = math.lcm(self.kernel.config.AK1, self.kernel.config.BK1)
        K_t = KBatch * KReadVec
        return math.ceil(K / K_t) * KReadVec

    def calculate_kpadded(self, K: int, KBatch: int) -> int:
        K_t = KBatch * self.kernel.config.KPerBlock
        return math.ceil(K / K_t) * self.kernel.config.KPerBlock

    def calculate_ak0padded(self, K: int, KBatch: int) -> int:
        K_t = KBatch * self.kernel.config.KPerBlock
        return math.ceil(K / K_t) * (self.kernel.config.KPerBlock / self.kernel.config.AK1)

    def calculate_bk0padded(self, K: int, KBatch: int) -> int:
        K_t = KBatch * self.kernel.config.KPerBlock
        return math.ceil(K / K_t) * (self.kernel.config.KPerBlock / self.kernel.config.BK1)

    def calculate_mblock(self, M: int) -> int:
        return math.ceil(M / self.kernel.config.MPerBlock)

    def calculate_nblock(self, N: int) -> int:
        return math.ceil(N / self.kernel.config.NPerBlock)

    def calculate_bn0shuffled(self, N: int) -> int:
        return math.ceil(N / self.NLane)

    def calculate_bk0shuffled(self, K: int) -> int:
        return math.ceil(K / (self.kernel.KLane * self.kernel.KPack))

    def print(self):
        print(f"problem {{M: {self.M}, N: {self.N}, K: {self.K}, SA: {self.StrideA}, SB: {self.StrideB}, SC: {self.StrideC}, MP: {self.MPadded}, NP: {self.NPadded}, KRead: {self.KRead}, KP: {self.KPadded}, AK0: {self.AK0}, BK0: {self.BK0}, MBlock: {self.MBlock}, NBlock: {self.NBlock}}}")


def CheckThreadGroupTensorSliceTransfer_v7r3(
        # First condition class ThreadGroupTensorSliceTransfer_v7r3
        SliceLengths, 
        ThreadClusterLengths,
        # Second condition class ThreadwiseTensorSliceTransfer_v7r3
        SrcVectorDim: int, 
        SrcScalarPerVectors: List[int],
        DstVectorDim: int, 
        DstScalarPerVector: int
        ):
    # ThreadClusterLengths = Sequence<1,
    #                      CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
    #                      1,
    #                      CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
    # ThreadClusterArrangeOrder = CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,

    assert len(SliceLengths) == len(ThreadClusterLengths)

    thread_slice_lengths = []
    for i in range(len(SliceLengths)):

        assert SliceLengths[i] % ThreadClusterLengths[i] == 0, f"SliceLengths[{i}] not divisible by ThreadClusterLengths[{i}]"
        thread_slice_lengths.append(
            SliceLengths[i] // ThreadClusterLengths[i]
        )
    
    for i in range(len(SliceLengths)):
        # assert BlockSliceLengths == decltype(thread_slice_lengths * ThreadClusterLengths{}
        assert SliceLengths[i] == (thread_slice_lengths[i] * ThreadClusterLengths[i]), f"SliceLengths[{i}] not same as thread_slice_lengths[{i}] * ThreadClusterLengths[{i}]"


    def CheckThreadwiseTensorSliceTransfer_v7r3(
            SliceLengths: List[int], 
            SrcVectorDim: int, 
            SrcScalarPerVectors: int,
            DstVectorDim: int, 
            DstScalarPerVector: int
    ):
        SrcScalarPerVector = SrcScalarPerVectors[0]
        assert SliceLengths[SrcVectorDim] % SrcScalarPerVector == 0, f"wrong! cannot evenly divide. SliceLengths[SrcVectorDim] % SrcScalarPerVector == 0 [=] {SliceLengths} [{SrcVectorDim}] % {SrcScalarPerVector} = {SliceLengths[SrcVectorDim] % SrcScalarPerVector}"
        assert SliceLengths[DstVectorDim] % DstScalarPerVector == 0, f"wrong! cannot evenly divide. SliceLengths[DstVectorDim] % DstScalarPerVector == 0 [=] {SliceLengths} [{DstVectorDim}] % {DstScalarPerVector} = {SliceLengths[DstVectorDim] % DstScalarPerVector}"

    CheckThreadwiseTensorSliceTransfer_v7r3(
        thread_slice_lengths,
        SrcVectorDim, 
        SrcScalarPerVectors,
        DstVectorDim, 
        DstScalarPerVector
    )

    # # usage
    # CheckThreadGroupTensorSliceTransfer_v7r3(
    #     ThreadClusterLengths=[
    #         1,
    #         kernel.config.CShuffleMXdlPerWavePerShuffle * kernel.MWave * kernel.config.MPerXDL,
    #         1,
    #         kernel.config.CShuffleNXdlPerWavePerShuffle * kernel.NWave * kernel.config.NPerXDL
    #     ],
    #     ThreadClusterArrangeOrder=kernel.config.CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    # )

def CheckThreadGroupTensorSliceTransfer_v4r1(
    BlockSliceLengths,
    ThreadClusterLengths
):
    
    # BlockSliceLengths = Sequence<AK0Number, MPerBlock, AK1Number>,
    # ThreadClusterLengths = ABlockTransferThreadClusterLengths_AK0_M_AK1,
    
    assert len(BlockSliceLengths) == len(ThreadClusterLengths)

    thread_slice_lengths = []
    #  thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{}
    for i in range(len(ThreadClusterLengths)):

        assert ThreadClusterLengths[i] % ThreadClusterLengths[i] == 0, f"ThreadClusterLengths[{i}] not divisible by ThreadClusterLengths[{i}] [:] { ThreadClusterLengths[i] } % { ThreadClusterLengths[i] }"
        thread_slice_lengths.append(
            ThreadClusterLengths[i] // ThreadClusterLengths[i]
        )

    for i in range(len(BlockSliceLengths)):
        # assert BlockSliceLengths == decltype(thread_slice_lengths * ThreadClusterLengths{}
        assert BlockSliceLengths[i] == (thread_slice_lengths[i] * ThreadClusterLengths[i]), f"BlockSliceLengths[{i}] not same as thread_slice_lengths[{i}] * ThreadClusterLengths[{i}] [:] { BlockSliceLengths[i] } == { thread_slice_lengths[i] } * { ThreadClusterLengths[i] }"





if __name__ == "__main__":
    # Example usage
    # config = DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle_config(
    #     ALayout=Layout.Row,
    #     BLayout=Layout.Col,
    #     DsLayout=(Layout.Row, Layout.Col),
    #     CLayout=Layout.Row,
    #     ADataType=CKDataType.fp8,
    #     BDataType=CKDataType.fp8,
    #     DsDataType=(CKDataType.float, CKDataType.float) ,
    #     CDataType=CKDataType.half_t,
    #     GemmAccDataType=CKDataType.float,
    #     CShuffleDataType=CKDataType.float,
    #     AElementwiseOperation = ElementOpDummy.ElementOp,
    #     BElementwiseOperation = ElementOpDummy.ElementOp,
    #     CElementwiseOperation = ElementOpDummy.ElementOp,
    #     GemmSpec=GemmSpecialization.MNKPadding,
    #     BlockSize=256,
    #     MPerBlock=256,
    #     NPerBlock=128,
    #     KPerBlock=128,
    #     AK1=16,
    #     BK1=16,
    #     MPerXDL=32,
    #     NPerXDL=32,
    #     MXdlPerWave=8,
    #     NXdlPerWave=1,
    #     ABlockTransferThreadClusterLengths_AK0_M_AK1=[8, 32, 1],
    #     ABlockTransferThreadClusterArrangeOrder=[1, 0, 2],
    #     ABlockTransferSrcAccessOrder=[1, 0, 2],
    #     ABlockTransferSrcVectorDim=2,
    #     ABlockTransferSrcScalarPerVector=16,
    #     ABlockTransferDstScalarPerVector_AK1=16,
    #     ABlockLdsExtraM=False,
    #     BBlockTransferThreadClusterLengths_BK0_N_BK1=[8, 32, 1],
    #     BBlockTransferThreadClusterArrangeOrder=[1, 0, 2],
    #     BBlockTransferSrcAccessOrder=[1, 0, 2],
    #     BBlockTransferSrcVectorDim=2,
    #     BBlockTransferSrcScalarPerVector=16,
    #     BBlockTransferDstScalarPerVector_BK1=16,
    #     BBlockLdsExtraN=False,
    #     CShuffleMXdlPerWavePerShuffle=1,
    #     CShuffleNXdlPerWavePerShuffle=1,
    #     CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock=[1, 32, 1, 8],
    #     CDEShuffleBlockTransferScalarPerVectors=[8, 8, 1],
    #     BlkGemmPipeSched=BlockGemmPipelineScheduler.Intrawave,
    #     BlkGemmPipelineVer=BlockGemmPipelineVersion.v1,
    #     ComputeTypeA=CKDataType.fp8
    # )

        
    # # Constants (these should be defined based on your specific requirements)
    # MPerBlock = 128
    # NPerBlock = 128
    # KPerBlock = 32
    # AK1Value = 8
    # BK1Value = 8
    # NLane = 32
    # KLane = 4
    # KPack = 8



    ## Working Kernel Config
    # kernel_config = GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle_config(
    #     ALayout=Layout.Row,
    #     BLayout=Layout.Col,
    #     DsLayout=(Layout.Row, Layout.Col),
    #     CLayout=Layout.Row,
    #     ADataType=CKDataType.fp8,
    #     BDataType=CKDataType.fp8,
    #     DsDataType=(CKDataType.float, CKDataType.float) ,
    #     CDataType=CKDataType.half_t,
    #     GemmAccDataType=CKDataType.float,
    #     CShuffleDataType=CKDataType.float,
    #     AElementwiseOperation = ElementOpDummy.ElementOp,
    #     BElementwiseOperation = ElementOpDummy.ElementOp,
    #     CElementwiseOperation = ElementOpDummy.ElementOp,
    #     GemmSpec=GemmSpecialization.MNKPadding,
    #     BlockSize=256,
    #     MPerBlock=256,
    #     NPerBlock=128,
    #     KPerBlock=128,
    #     AK1=16,
    #     BK1=16,
    #     MPerXDL=32,
    #     NPerXDL=32,
    #     MXdlPerWave=8,
    #     NXdlPerWave=1,
    #     ABlockTransferThreadClusterLengths_AK0_M_AK1=[8, 32, 1],
    #     ABlockTransferThreadClusterArrangeOrder=[1, 0, 2],
    #     ABlockTransferSrcAccessOrder=[1, 0, 2],
    #     ABlockTransferSrcVectorDim=2,
    #     ABlockTransferSrcScalarPerVector=16,
    #     ABlockTransferDstScalarPerVector_AK1=16,
    #     AThreadTransferSrcResetCoordinateAfterRun=False,
    #     ABlockLdsExtraM=False,
    #     BBlockTransferThreadClusterLengths_BK0_N_BK1=[8, 32, 1],
    #     BBlockTransferThreadClusterArrangeOrder=[1, 0, 2],
    #     BBlockTransferSrcAccessOrder=[1, 0, 2],
    #     BBlockTransferSrcVectorDim=2,
    #     BBlockTransferSrcScalarPerVector=16,
    #     BBlockTransferDstScalarPerVector_BK1=16,
    #     BThreadTransferSrcResetCoordinateAfterRun=False,
    #     BBlockLdsExtraN=False,
    #     CShuffleMXdlPerWavePerShuffle=1,
    #     CShuffleNXdlPerWavePerShuffle=1,
    #     CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock=[1, 32, 1, 8],
    #     CDEShuffleBlockTransferScalarPerVectors=[8, 8, 1],
    #     BlkGemmPipeSched=BlockGemmPipelineScheduler.Intrawave,
    #     BlkGemmPipelineVer=BlockGemmPipelineVersion.v1,
    #     ComputeTypeA=CKDataType.fp8
    # )

    
    # kernel_config = GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle_config(
    #     ALayout=Layout.Row,
    #     BLayout=Layout.Col,
    #     DsLayout=(Layout.Row, Layout.Col),
    #     CLayout=Layout.Row,
    #     ADataType=CKDataType.fp8,
    #     BDataType=CKDataType.fp8,
    #     DsDataType=(CKDataType.float, CKDataType.float) ,
    #     CDataType=CKDataType.half_t,
    #     GemmAccDataType=CKDataType.float,
    #     CShuffleDataType=CKDataType.float,
    #     AElementwiseOperation = ElementOpDummy.ElementOp,
    #     BElementwiseOperation = ElementOpDummy.ElementOp,
    #     CElementwiseOperation = ElementOpDummy.ElementOp,
    #     GemmSpec=GemmSpecialization.MNKPadding,
    #     BlockSize=128,
    #     MPerBlock=16,
    #     NPerBlock=128,
    #     KPerBlock=128,
    #     AK1=16,
    #     BK1=16,
    #     MPerXDL=16,
    #     NPerXDL=16,
    #     MXdlPerWave=1,
    #     NXdlPerWave=4,
    #     ABlockTransferThreadClusterLengths_AK0_M_AK1=[8, 32, 1],
    #     ABlockTransferThreadClusterArrangeOrder=[1, 0, 2],
    #     ABlockTransferSrcAccessOrder=[1, 0, 2],
    #     ABlockTransferSrcVectorDim=2,
    #     ABlockTransferSrcScalarPerVector=16,
    #     ABlockTransferDstScalarPerVector_AK1=16,
    #     AThreadTransferSrcResetCoordinateAfterRun=False,
    #     ABlockLdsExtraM=False,
    #     BBlockTransferThreadClusterLengths_BK0_N_BK1=[16, 16, 1],
    #     BBlockTransferThreadClusterArrangeOrder=[1, 0, 2],
    #     BBlockTransferSrcAccessOrder=[1, 0, 2],
    #     BBlockTransferSrcVectorDim=2,
    #     BBlockTransferSrcScalarPerVector=16,
    #     BBlockTransferDstScalarPerVector_BK1=16,
    #     BThreadTransferSrcResetCoordinateAfterRun=False,
    #     BBlockLdsExtraN=False,
    #     CShuffleMXdlPerWavePerShuffle=1,
    #     CShuffleNXdlPerWavePerShuffle=1,
    #     CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock=[1, 16, 1, 16],
    #     CDEShuffleBlockTransferScalarPerVectors=[4, 4, 1],
    #     BlkGemmPipeSched=BlockGemmPipelineScheduler.Intrawave,
    #     BlkGemmPipelineVer=BlockGemmPipelineVersion.v1,
    #     ComputeTypeA=CKDataType.fp8
    # )


    kernel_config = GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle_config(
        ALayout=Layout.Row,
        BLayout=Layout.Col,
        DsLayout=(Layout.Row, Layout.Col),
        CLayout=Layout.Row,
        ADataType=CKDataType.fp8,
        BDataType=CKDataType.fp8,
        DsDataType=(CKDataType.float, CKDataType.float) ,
        CDataType=CKDataType.half_t,
        GemmAccDataType=CKDataType.float,
        CShuffleDataType=CKDataType.float,
        AElementwiseOperation = ElementOpDummy.ElementOp,
        BElementwiseOperation = ElementOpDummy.ElementOp,
        CElementwiseOperation = ElementOpDummy.ElementOp,
        GemmSpec=GemmSpecialization.MNKPadding,
        BlockSize=128,
        MPerBlock=16,
        NPerBlock=128,
        KPerBlock=128,
        AK1=16,
        BK1=16,
        MPerXDL=16,
        NPerXDL=16,
        MXdlPerWave=1,
        NXdlPerWave=4,
        ABlockTransferThreadClusterLengths_AK0_M_AK1=[8, 16, 16],
        ABlockTransferThreadClusterArrangeOrder=[1, 0, 2],
        ABlockTransferSrcAccessOrder=[1, 0, 2],
        ABlockTransferSrcVectorDim=2,
        ABlockTransferSrcScalarPerVector=16,
        ABlockTransferDstScalarPerVector_AK1=16,
        AThreadTransferSrcResetCoordinateAfterRun=False,
        ABlockLdsExtraM=False,
        BBlockTransferThreadClusterLengths_BK0_N_BK1=[8, 16, 16],
        BBlockTransferThreadClusterArrangeOrder=[1, 0, 2],
        BBlockTransferSrcAccessOrder=[1, 0, 2],
        BBlockTransferSrcVectorDim=2,
        BBlockTransferSrcScalarPerVector=16,
        BBlockTransferDstScalarPerVector_BK1=16,
        BThreadTransferSrcResetCoordinateAfterRun=False,
        BBlockLdsExtraN=False,
        CShuffleMXdlPerWavePerShuffle=1,
        CShuffleNXdlPerWavePerShuffle=1,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock=[1, 16, 1, 4],
        CDEShuffleBlockTransferScalarPerVectors=[8, 8, 1],
        BlkGemmPipeSched=BlockGemmPipelineScheduler.Intrawave,
        BlkGemmPipelineVer=BlockGemmPipelineVersion.v1,
        ComputeTypeA=CKDataType.fp8
    )


    kernel = GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle(kernel_config)


    CheckThreadGroupTensorSliceTransfer_v7r3(
        SliceLengths=[
            1,
            kernel.config.CShuffleMXdlPerWavePerShuffle * kernel.MWave * kernel.config.MPerXDL,
            1,
            kernel.config.CShuffleNXdlPerWavePerShuffle * kernel.NWave * kernel.config.NPerXDL
        ],
        ThreadClusterLengths=kernel.config.CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        SrcVectorDim=3, 
        SrcScalarPerVectors=kernel.config.CDEShuffleBlockTransferScalarPerVectors,
        DstVectorDim=3, 
        DstScalarPerVector=kernel.CShuffleBlockTransferScalarPerVector_NPerBlock
    )

    print(f"""
    CheckThreadGroupTensorSliceTransfer_v4r1()
            self.AK0Number = config.KPerBlock / config.AK1
        BlockSliceLengths=[
            kernel.AK0Number={kernel.AK0Number}
            kernel.config.MPerBlock={kernel.config.MPerBlock}
            kernel.AK1Number={kernel.AK1Number}
          ]

          ThreadClusterLengths=kernel.config.ABlockTransferThreadClusterLengths_AK0_M_AK1
        ThreadClusterLengths={kernel.config.ABlockTransferThreadClusterLengths_AK0_M_AK1}
    
    """)

    CheckThreadGroupTensorSliceTransfer_v4r1(
        BlockSliceLengths=[
            kernel.AK0Number, kernel.config.MPerBlock, kernel.AK1Number
        ],
        ThreadClusterLengths=kernel.config.ABlockTransferThreadClusterLengths_AK0_M_AK1
    )

    # # Example usage
    # problem = Problem(
    #     kernel=kernel,
    #     M=1, N=128, K=128, 
    #     StrideA=256, StrideB=256, StrideDs=[256, 256], 
    #     StrideC=256, 
    #     KBatch=1)
    # problem.print()

    # kernel.CheckValidity(
    #     problem
    # )