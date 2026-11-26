#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// n_x = slots/ranks per phone, n_y = phones, n_z = array types (2).
// map, idata, odata: device pointer tables indexed by idz * n_y + idy.
// map's first n_y pointers are existing free-slot flag buffers; the next n_y
// are new-boid flags. odata similarly holds free-slot buffers then new ranks.
// idata may be nullptr to scatter implicit indices instead of input values.
// size_out: device table of two pointers, dev_phoneFreeCounts/dev_phoneNewCounts.
// scan: n_z * n_y * n_x uint32_t elements.
// block_sums/block_offsets: n_z * n_y * ceil(n_x / 256) uint32_t elements.
// Supports up to 65,536 slots per phone with the current two-level scan.
void batch_stream_compaction(int n_x, int n_y, int n_z, uint32_t *const *odata,
    uint32_t *const *idata, uint8_t *const *map, uint32_t *scan,
    uint32_t *block_sums, uint32_t *block_offsets,
    uint32_t *const *size_out, cudaStream_t stream);

