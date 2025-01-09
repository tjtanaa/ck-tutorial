import math
import numpy as np

def calculator(
    CDEShuffleBlockTransferScalarPerVectors,
    KPerBlock,
    AK1Value,
    BK1Value,
    BlockSize,
    DsDataType,
    selected_mfma_k_per_blk,
    mfma_selector_GetKPerXdlops,
    mfma_selector_GetK1PerXdlops,
    MPerXdl,
    MPerBlock,
    MXdlPerWave,
    NPerXdl,
    NPerBlock,
    NXdlPerWave,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    warpSize = 64
):

    assert selected_mfma_k_per_blk in [16, 32]

    CShuffleBlockTransferScalarPerVector_NPerBlock = CDEShuffleBlockTransferScalarPerVectors[0]

    # K1 should be Number<...>
    AK0Number = KPerBlock / AK1Value
    BK0Number = KPerBlock / BK1Value
    AK1Number = AK1Value
    BK1Number = BK1Value
    BlockSizeNumber = BlockSize

    # static constexpr index_t NumDTensor = DsDataType::Size();

    KPack = math.max(math.lcm(AK1Number, BK1Number), selected_mfma_k_per_blk)
    # KLane = mfma_selector::GetKPerXdlops() / mfma_selector::GetK1PerXdlops()
    KLane = mfma_selector_GetKPerXdlops / mfma_selector_GetK1PerXdlops

    KRepeat = KPerBlock / KLane / KPack
    NLane = NPerXdl
    NWave = NPerBlock / NPerXdl / NXdlPerWave

    assert NWave * warpSize == BlockSize


    assert (MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 and
            NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0
            )

    MWave = MPerBlock / MPerXdl / MXdlPerWave

    BlockSliceLengths = [
           1,
            CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
            1,
            CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl
    ]

    for i in range(4):
        assert(
            BlockSliceLengths[i] % 
            CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        ) == 0

    # auto cde_block_copy_lds_and_global = ThreadGroupTensorSliceTransfer_v7r3<
    #     ThisThreadBlock,
    #     decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType{})),
    #     Tuple<EDataType>,
    #     decltype(c_ds_desc_refs),
    #     decltype(tie(e_grid_desc_mblock_mperblock_nblock_nperblock)),
    #     CElementwiseOperation,
    #     Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // FIXME: make Sequence
    #                                                                 // support arbitray type
    #     Sequence<1,
    #              CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
    #              1,
    #              CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
    #     CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    #     Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
    #     Sequence<0, 1, 2, 3>, // typename SrcDimAccessOrder,
    #     Sequence<0, 1, 2, 3>, // typename DstDimAccessOrder,
    #     3,                    // index_t SrcVectorDim,
    #     3,                    // index_t DstVectorDim,
    #     CDEShuffleBlockTransferScalarPerVectors,
    #     CShuffleBlockTransferScalarPerVector_NPerBlock,
    #     sequence_merge_t<
    #         Sequence<true>,
    #         uniform_sequence_gen_t<NumDTensor,
    #                                false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
    #     Sequence<false>>                    // ThreadTransferDstResetCoordinateAfterRunFlags
    #     {c_ds_desc_refs,
    #      idx_c_ds_block_begin,
    #      tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
    #      make_tuple(make_multi_index(block_m_id, 0, block_n_id, 0)),
    #      c_element_op};

if __name__ == "__main__":
    calculator()

