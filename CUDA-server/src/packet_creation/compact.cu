#include "kernel.h"
#include "compact.cuh"
#include <cub/block/block_scan.cuh>

__global__ void scanBlocks(int num_slots, int num_phones, int num_arrays,
    uint8_t *const *map, uint32_t *scan, uint32_t *block_sums) {
    int boid_id = threadIdx.x + blockDim.x * blockIdx.x;
    int phone_id = blockIdx.y;
    int array_id = blockIdx.z;

    int array_offset = array_id * num_phones;
    typedef cub::BlockScan<uint, 256> BlockScan;
    
    __shared__ typename BlockScan::TempStorage temp_storage;
    uint thread_data = boid_id < num_slots ? map[array_offset + phone_id][boid_id] : 0;
    uint block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

    if (boid_id < num_slots) {
        scan[(array_offset + phone_id) * num_slots + boid_id] = thread_data;
    }

    if (threadIdx.x == 0) {
        block_sums[(array_offset + phone_id) * gridDim.x + blockIdx.x] = block_aggregate;
    }
}

__global__ void scanBlockTotals(int num_phones, int num_arrays, int num_blocks_x,
    uint32_t *block_sums, uint32_t *block_offsets, uint32_t *const *size_out) {
    int phone_id = blockIdx.y;
    int array_id = blockIdx.z;
    int idx = threadIdx.x;

    int array_offset = array_id * num_phones;

    using BlockScan = cub::BlockScan<uint, 256>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    uint thread_data = idx < num_blocks_x ? block_sums[(array_offset + phone_id) * num_blocks_x + idx] : 0;
    uint block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

    if (idx < num_blocks_x) {
        block_offsets[(array_offset + phone_id) * num_blocks_x + idx] = thread_data;
    }

    if (idx == 0) {
        size_out[array_id][phone_id] = block_aggregate;
    }
}

__global__ void map(int num_slots, int num_phones, int num_arrays,
    uint32_t *const *data, uint8_t *const *map) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = blockIdx.y;
    int idz = blockIdx.z;

    if (idx < num_slots && idy < num_phones && idz < num_arrays) {
        // do calc on data here
        map[idz * num_phones + idy][idx] = 1;
    }
}

__global__ void scatter(int num_slots, int num_phones, int num_arrays,
    uint8_t *const *map, const uint32_t *scan,
    const uint32_t *block_offsets, uint32_t *const *data, uint32_t *const *o_data) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = blockIdx.y;
    int idz = blockIdx.z;

    if (idx < num_slots && idy < num_phones && idz < num_arrays && map[idz * num_phones + idy][idx]) {
        int array = idz * num_phones + idy;
        uint32_t block_offset = block_offsets[array * gridDim.x + blockIdx.x];
        o_data[array][block_offset + scan[array * num_slots + idx]] =
            data == nullptr ? static_cast<uint32_t>(idx) : data[array][idx];
    }
}

void batch_stream_compaction(int num_slots, int num_phones, int num_arrays, uint32_t *const *odata,
    uint32_t *const *idata, uint8_t *const *map, uint32_t *scan,
    uint32_t *block_sums, uint32_t *block_offsets,
    uint32_t *const *size_out, cudaStream_t stream) {
  if (num_slots <= 0 || num_phones <= 0 || num_arrays <= 0) return;

  dim3 threadsPerBlock(256, 1, 1);
  dim3 numBlocks(
    (num_slots + threadsPerBlock.x - 1) / threadsPerBlock.x,
    num_phones,
    num_arrays
  );
  dim3 numBlocksTotals(1, num_phones, num_arrays);

  scanBlocks<<<numBlocks, threadsPerBlock, 0, stream>>>(
    num_slots, num_phones, num_arrays, map, scan, block_sums);
  scanBlockTotals<<<numBlocksTotals, threadsPerBlock, 0, stream>>>(
    num_phones, num_arrays, numBlocks.x, block_sums, block_offsets, size_out);
  scatter<<<numBlocks, threadsPerBlock, 0, stream>>>(
    num_slots, num_phones, num_arrays, map, scan, block_offsets, idata, odata);
}
