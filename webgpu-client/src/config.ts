export const DEFAULT_BOIDS = 20;
export const MAX_BOIDS = 100_000;
export const MAX_SORT_SIZE = 131_072;
export const GRID_SIDE = 22;
export const GRID_CELLS = GRID_SIDE ** 3;
export const WORKGROUP_SIZE = 256;
export const SCAN_BLOCK_SIZE = WORKGROUP_SIZE * 2;
export const MAX_SCAN_BLOCK_COUNT = MAX_SORT_SIZE / SCAN_BLOCK_SIZE;

export const RADIX_PASSES = Math.ceil(Math.log2(GRID_CELLS));
