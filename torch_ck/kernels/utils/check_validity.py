from typing import List, Tuple
from enum import Enum
import math

# Enum for GemmSpecialization
class GemmSpecialization(Enum):
    MPadding = "MPadding"
    MNPadding = "MNPadding"
    MKPadding = "MKPadding"
    MNKPadding = "MNKPadding"
    NPadding = "NPadding"
    NKPadding = "NKPadding"
    KPadding = "KPadding"

# Enum for tensor layouts
class TensorLayout(Enum):
    RowMajor = "RowMajor"
    ColumnMajor = "ColumnMajor"

# Mock BlockwiseGemmPipe class for pipeline validity check
class BlockwiseGemmPipe:
    @staticmethod
    def PrefetchStages():
        return 4  # Example value

# Function to replicate CheckValidity
def CheckValidity(
    GemmSpec: GemmSpecialization,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    CLayout: TensorLayout,
    MPerBlock: int,
    NPerBlock: int,
    KPerBlock: int,
    MPerXdl: int,
    NPerXdl: int,
    MXdlPerWave: int,
    NXdlPerWave: int,
    ABlockTransferSrcScalarPerVector: int,
    BBlockTransferSrcScalarPerVector: int,
    CShuffleBlockTransferScalarPerVector_NPerBlock: int,
    karg_M: int,
    karg_N: int,
    karg_K: int,
    karg_KBatch: int,
    AK1Value: int,
    BK1Value: int,
    BlkGemmPipelineVer: str = "v4",
) -> bool:
    # Condition 1: Tuning Parameter Validation
    if (MPerBlock % (MPerXdl * MXdlPerWave) != 0) or (NPerBlock % (NXdlPerWave * NPerXdl) != 0):
        print("Invalid tuning parameters!")
        return False

    # Condition 2: Matrix M Dimension Validation
    if (
        GemmSpec
        not in [
            GemmSpecialization.MPadding,
            GemmSpecialization.MNPadding,
            GemmSpecialization.MKPadding,
            GemmSpecialization.MNKPadding,
        ]
        and ALayout != TensorLayout.RowMajor
    ):
        if karg_M % MPerBlock != 0:
            print(f"Arg M value ({karg_M}) is not a multiple of MPerBlock ({MPerBlock})!")
            return False

    # Condition 3: Matrix N Dimension Validation
    if (
        GemmSpec
        not in [
            GemmSpecialization.NPadding,
            GemmSpecialization.MNPadding,
            GemmSpecialization.NKPadding,
            GemmSpecialization.MNKPadding,
        ]
        and BLayout == TensorLayout.RowMajor
    ):
        if karg_N % NPerBlock != 0:
            print(f"Arg N value ({karg_N}) is not a multiple of NPerBlock ({NPerBlock})!")
            return False

    # Condition 4: Matrix K Dimension Validation
    if GemmSpec not in [
        GemmSpecialization.KPadding,
        GemmSpecialization.MKPadding,
        GemmSpecialization.NKPadding,
        GemmSpecialization.MNKPadding,
    ]:
        K_t = karg_KBatch * KPerBlock
        if karg_K % K_t != 0:
            print(
                f"Arg K value ({karg_K}) is not a multiple of K_Batch * KPerBlock ({K_t})!"
            )
            return False
    else:
        KReadVec = math.lcm(AK1Value, BK1Value)
        K_t = karg_KBatch * KReadVec
        KReadPadSplited = (karg_K + K_t - 1) // K_t * KReadVec
        if (KReadPadSplited * (karg_KBatch - 1)) >= karg_K:
            print("Invalid K dimension with padding!")
            return False

    # Condition 5: Matrix A Layout Validation
    if ALayout == TensorLayout.RowMajor:
        if karg_K % ABlockTransferSrcScalarPerVector != 0:
            print(
                f"Arg K ({karg_K}) is not a multiple of ABlockTransferSrcScalarPerVector ({ABlockTransferSrcScalarPerVector})!"
            )
            return False
    else:
        if karg_M % ABlockTransferSrcScalarPerVector != 0:
            print(
                f"Arg M ({karg_M}) is not a multiple of ABlockTransferSrcScalarPerVector ({ABlockTransferSrcScalarPerVector})!"
            )
            return False

    # Condition 6: Matrix B Layout Validation
    if BLayout == TensorLayout.RowMajor:
        if karg_N % BBlockTransferSrcScalarPerVector != 0:
            print(
                f"Arg N ({karg_N}) is not a multiple of BBlockTransferSrcScalarPerVector ({BBlockTransferSrcScalarPerVector})!"
            )
            return False
    else:
        if karg_K % BBlockTransferSrcScalarPerVector != 0:
            print(
                f"Arg K ({karg_K}) is not a multiple of BBlockTransferSrcScalarPerVector ({BBlockTransferSrcScalarPerVector})!"
            )
            return False

    # Condition 7: Matrix C Layout Validation
    if CLayout == TensorLayout.RowMajor:
        if karg_N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0:
            print(
                f"Arg N ({karg_N}) is not a multiple of CShuffleBlockTransferScalarPerVector_NPerBlock ({CShuffleBlockTransferScalarPerVector_NPerBlock})!"
            )
            return False
    else:
        if karg_M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0:
            print(
                f"Arg M ({karg_M}) is not a multiple of CShuffleBlockTransferScalarPerVector_NPerBlock ({CShuffleBlockTransferScalarPerVector_NPerBlock})!"
            )
            return False

    # Condition 8: Pipeline Validity Check (Optional)
    if BlkGemmPipelineVer != "v1":
        num_k_loop = karg_K // (KPerBlock // AK1Value)
        if num_k_loop <= BlockwiseGemmPipe.PrefetchStages():
            print("Pipeline validity check failed!")
            return False

    # If all conditions pass
    return True


# Example usage
if __name__ == "__main__":
    # Example parameters
    GemmSpec = GemmSpecialization.MNKPadding
    ALayout = TensorLayout.RowMajor
    BLayout = TensorLayout.ColumnMajor
    CLayout = TensorLayout.RowMajor
    MPerBlock = 128
    NPerBlock = 128
    KPerBlock = 32
    MPerXdl = 16
    NPerXdl = 16
    MXdlPerWave = 4
    NXdlPerWave = 4
    ABlockTransferSrcScalarPerVector = 4
    BBlockTransferSrcScalarPerVector = 4
    CShuffleBlockTransferScalarPerVector_NPerBlock = 4
    karg_M = 256
    karg_N = 256
    karg_K = 64
    karg_KBatch = 2
    AK1Value = 4
    BK1Value = 4

    # Call the function
    is_valid = CheckValidity(
        GemmSpec,
        ALayout,
        BLayout,
        CLayout,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerXdl,
        NPerXdl,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferSrcScalarPerVector,
        BBlockTransferSrcScalarPerVector,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        karg_M,
        karg_N,
        karg_K,
        karg_KBatch,
        AK1Value,
        BK1Value,
    )

    print(f"CheckValidity result: {is_valid}")