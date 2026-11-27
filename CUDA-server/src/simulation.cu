#define GLM_FORCE_CUDA

#include <cuda.h>
#include <math_constants.h>
#include "kernel.h"
#include "utilityCore.hpp"
#include "packet_creation/glint.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <cub/cub.cuh>

#include <glm/glm.hpp>

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
glm::vec3 *dev_pos;
glm::vec3 *dev_pos2;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;
uint32_t *dev_boidIds;
uint32_t *dev_boidIds2;

glm::vec3 *dev_pos_snapshot;
uint32_t *dev_boidIds_snapshot;
int *dev_gridStarts_snapshot;
int *dev_gridEnds_snapshot;


int *dev_particleArrayIndices;
int *dev_particleGridIndices;
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;
thrust::device_ptr<glm::vec3> dev_thrust_vel1;
thrust::device_ptr<glm::vec3> dev_thrust_vel2;
thrust::device_ptr<glm::vec3> dev_thrust_pos;
thrust::device_ptr<glm::vec3> dev_thrust_pos2;
thrust::device_ptr<uint32_t> dev_thrust_boidIds;
thrust::device_ptr<uint32_t> dev_thrust_boidIds2;

int *dev_gridCellStartIndices;
int *dev_gridCellEndIndices;

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

cudaStream_t simulation_stream;
cudaEvent_t simulationSnapshotReadyEvent;

GlintOctree glintOctree;

#if SIMULATION_TYPE == GLINT_SIMULATION
Boids::GlintLeafSnapshot *dev_glintLeaves_snapshot = nullptr;
uint32_t *dev_glintBoidIndices_snapshot = nullptr;
int *dev_glintLeafOffsets_snapshot = nullptr;
void *glintSnapshotScanTemp = nullptr;
size_t glintSnapshotScanTempBytes = 0;
uint32_t glintSnapshotLeafCount = 0;

struct GetGlintLeafSize {
  const int *active_leaves;
  const GlintOctree::GlintNode *nodes;
  __host__ __device__ int operator()(int leafIdx) const {
    return nodes[active_leaves[leafIdx]].size;
  }
};

__global__ void kernSnapshotGlintLeaves(int leafCount, const int *active_leaves,
    const GlintOctree::GlintNode *nodes, const GlintOctree::PartitionBuff *partitions,
    const int *offsets, Boids::GlintLeafSnapshot *leaves, uint32_t *boidIndices) {
  const int leafIdx = blockIdx.x;
  if (leafIdx >= leafCount) return;
  const int nodeIdx = active_leaves[leafIdx];
  const auto &node = nodes[nodeIdx];
  const int start = offsets[leafIdx];
  if (threadIdx.x == 0) {
    leaves[leafIdx] = {node.min_coords, node.max_coords,
      static_cast<uint32_t>(start), static_cast<uint32_t>(node.size), nodeIdx};
  }
  for (int i = threadIdx.x; i < node.size; i += blockDim.x) {
    boidIndices[start + i] = partitions[node.part_idx].boid_idx[i];
  }
}
#endif

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

__global__ void kernGenerateFlockPositions(int time, int N, glm::vec3 *pos) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  glm::vec3 flockCenter;
  switch (index % 3) {
    case 0: flockCenter = glm::vec3(-75.0f, -30.0f, 0.0f); break;
    case 1: flockCenter = glm::vec3(75.0f, -30.0f, 0.0f); break;
    default: flockCenter = glm::vec3(0.0f, 75.0f, 0.0f); break;
  }

  pos[index] = flockCenter + generateRandomVec3(time, index) * spawnFlockRadius;
}

__global__ void kernInitializeBoidIds(int N, uint32_t *boidIds) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    boidIds[index] = static_cast<uint32_t>(index);
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  cudaStreamCreate(&simulation_stream);
  cudaEventCreateWithFlags(&simulationSnapshotReadyEvent, cudaEventDisableTiming);

  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_boidIds, N * sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc dev_boidIds failed!");
  cudaMalloc((void**)&dev_boidIds2, N * sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc dev_boidIds2 failed!");
  cudaMalloc((void**)&dev_pos_snapshot, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_snapshot failed!");
  cudaMalloc((void**)&dev_boidIds_snapshot, N * sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc dev_boidIds_snapshot failed!");
  kernInitializeBoidIds<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(N, dev_boidIds);
  checkCUDAErrorWithLine("kernInitializeBoidIds failed!");

  // Initialize velocity to random values
  cudaMemsetAsync(dev_vel1, 0, N * sizeof(glm::vec3), simulation_stream);
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(1, numObjects,
    dev_vel1, maxSpeed);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  cudaMemsetAsync(dev_vel2, 0, N * sizeof(glm::vec3), simulation_stream);
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  kernGenerateFlockPositions<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(1, numObjects, dev_pos);
  checkCUDAErrorWithLine("kernGenerateFlockPositions failed!");

  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum = glm::vec3(0.0f);
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  cudaMalloc((void**)&dev_particleArrayIndices, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_gridStarts_snapshot, gridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_gridEnds_snapshot, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("allocate network grid snapshot failed!");

  // needed for use with thrust
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
  dev_thrust_vel1 = thrust::device_pointer_cast(dev_vel1);
  dev_thrust_vel2 = thrust::device_pointer_cast(dev_vel2);
  dev_thrust_pos = thrust::device_pointer_cast(dev_pos);
  dev_thrust_pos2 = thrust::device_pointer_cast(dev_pos2);
  dev_thrust_boidIds = thrust::device_pointer_cast(dev_boidIds);
  dev_thrust_boidIds2 = thrust::device_pointer_cast(dev_boidIds2);

#if SIMULATION_TYPE == GLINT_SIMULATION
  cudaStreamSynchronize(simulation_stream);
  glintOctree.init(N, dev_pos, simulation_stream);
  cudaMallocAsync((void **)&dev_glintLeaves_snapshot,
    NODE_STK_MAX_SIZE * sizeof(GlintLeafSnapshot), simulation_stream);
  cudaMallocAsync((void **)&dev_glintBoidIndices_snapshot, N * sizeof(uint32_t), simulation_stream);
  cudaMallocAsync((void **)&dev_glintLeafOffsets_snapshot, NODE_STK_MAX_SIZE * sizeof(int), simulation_stream);
  const auto leafData = glintOctree.get_leaf_data();
  auto leafSizes = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
    GetGlintLeafSize{leafData.active_leaves, leafData.nodes});
  cub::DeviceScan::ExclusiveSum(nullptr, glintSnapshotScanTempBytes, leafSizes,
    dev_glintLeafOffsets_snapshot, NODE_STK_MAX_SIZE, simulation_stream);
  cudaMallocAsync(&glintSnapshotScanTemp, glintSnapshotScanTempBytes, simulation_stream);
  checkCUDAErrorWithLine("allocate network GLINT snapshot failed!");
#endif

  cudaStreamSynchronize(simulation_stream);
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = 1.0f;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaStreamSynchronize(simulation_stream);
}


/******************
* stepSimulation *
******************/

__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= N) {
      return;
    }

    glm::vec3 boidPos = pos[i];
    int x = imin(imax((int)floor((boidPos.x - gridMin.x) * inverseCellWidth), 0), gridResolution - 1);
    int y = imin(imax((int)floor((boidPos.y - gridMin.y) * inverseCellWidth), 0), gridResolution - 1);
    int z = imin(imax((int)floor((boidPos.z - gridMin.z) * inverseCellWidth), 0), gridResolution - 1);

    indices[i] = i;
    gridIndices[i] = gridIndex3Dto1D(x, y, z, gridResolution);
}

__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N || index < 0) {
    return;
  }

  if (index == 0) {
    gridCellStartIndices[particleGridIndices[index]] = index;
  }
  if (index == N - 1) {
    gridCellEndIndices[particleGridIndices[index]] = index;
  }
  if (index > 0 && particleGridIndices[index] != particleGridIndices[index - 1]) {
    gridCellStartIndices[particleGridIndices[index]] = index;
    gridCellEndIndices[particleGridIndices[index - 1]] = index - 1;
  }
}

__device__ void calculateNeighborVelChangeCoherent(int cellX, int cellY, int cellZ, int dx, int dy, int dz,
    int gridResolution, int *gridCellStartIndices, int *gridCellEndIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 boidPos, int boidIndex,
    glm::vec3 &perceived_center, int &num_neighbors_r1, glm::vec3 &c, glm::vec3 &perceived_velocity,
    int &num_neighbors_r3) {
  int neighborCellX = cellX + dx;
  int neighborCellY = cellY + dy;
  int neighborCellZ = cellZ + dz;

  if (neighborCellX < 0 || neighborCellX >= gridResolution ||
    neighborCellY < 0 || neighborCellY >= gridResolution ||
    neighborCellZ < 0 || neighborCellZ >= gridResolution) {
      return;
    }

  int neighborCell = gridIndex3Dto1D(neighborCellX, neighborCellY, neighborCellZ, gridResolution);

  int startIndex = gridCellStartIndices[neighborCell];
  int endIndex = gridCellEndIndices[neighborCell];

  if (startIndex < 0 || endIndex < 0) {
    return;
  }

  for (int i = startIndex; i <= endIndex; i++) {
    int neighborBoid = i;
    if (neighborBoid == boidIndex) {
      continue;
    }

    const glm::vec3 neighborPos = pos[neighborBoid];
    const glm::vec3 neighborOffset = neighborPos - boidPos;
    const float distanceSquared = glm::dot(neighborOffset, neighborOffset);

    // rule 1
    if (distanceSquared < rule1Distance * rule1Distance) {
      perceived_center += neighborPos;
      num_neighbors_r1++;
    }

    // rule 2
    if (distanceSquared < rule2Distance * rule2Distance) {
      c -= neighborOffset;
    }

    // rule 3
    if (distanceSquared < rule3Distance * rule3Distance) {
      perceived_velocity += vel1[neighborBoid];
      num_neighbors_r3++;
    }
  }
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // rule 1
  glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
  int num_neighbors_r1 = 0;

  // rule 2
  glm::vec3 c = glm::vec3(0.0f, 0.0f, 0.0f);

  // rule 3
  glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
  int num_neighbors_r3 = 0;

  int boidIndex = threadIdx.x + (blockIdx.x * blockDim.x);
  if (boidIndex >= N) {
    return;
  }

  glm::vec3 boidPos = pos[boidIndex];
  int cellX = floor((boidPos.x - gridMin.x) * inverseCellWidth);
  int cellY = floor((boidPos.y - gridMin.y) * inverseCellWidth);
  int cellZ = floor((boidPos.z - gridMin.z) * inverseCellWidth);

  // calculate if lower half or upper half all 3 dims to get dx, dy, dz
  float localX = (boidPos.x - gridMin.x) * inverseCellWidth - cellX;
  float localY = (boidPos.y - gridMin.y) * inverseCellWidth - cellY;
  float localZ = (boidPos.z - gridMin.z) * inverseCellWidth - cellZ;
  int dx = localX < 0.5f ? -1 : 1;
  int dy = localY < 0.5f ? -1 : 1;
  int dz = localZ < 0.5f ? -1 : 1;

  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, 0, 0, 0, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, dx, 0, 0, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, 0, dy, 0, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, 0, 0, dz, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, dx, dy, 0, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, dx, 0, dz, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, 0, dy, dz, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);
  calculateNeighborVelChangeCoherent(cellX, cellY, cellZ, dx, dy, dz, gridResolution, gridCellStartIndices, gridCellEndIndices, pos, vel1, boidPos, boidIndex, perceived_center, num_neighbors_r1, c, perceived_velocity, num_neighbors_r3);


  glm::vec3 rule1_vel_change = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_vel_change = glm::vec3(0.0f, 0.0f, 0.0f);

  if (num_neighbors_r1 != 0) {
    perceived_center /= num_neighbors_r1;
    rule1_vel_change = (perceived_center - boidPos) * rule1Scale;
  }

  if (num_neighbors_r3 != 0) {
    perceived_velocity /= num_neighbors_r3;
    rule3_vel_change = perceived_velocity * rule3Scale;
  }

  glm::vec3 rule2_vel_change = c * rule2Scale;
  glm::vec3 newVel = vel1[boidIndex] + rule1_vel_change + rule2_vel_change + rule3_vel_change;

  if (glm::length(newVel) > maxSpeed) {
    newVel = glm::normalize(newVel) * maxSpeed;
  }
  vel2[boidIndex] = newVel;
}

const uint32_t* Boids::getDeviceBoidIds() {
  return dev_boidIds;
}

void Boids::stepSimulationCoherentGrid(float dt, const CameraCullData *cameras,
    bool *cameraChanged, uint32_t phoneCount) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    int gridResolution = gridSideCount;

    dim3 fullBlocksPerGridGrid((gridCellCount + blockSize - 1) / blockSize);

    kernResetIntBuffer<<<fullBlocksPerGridGrid, blockSize, 0, simulation_stream>>>(gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernResetIntBuffer<<<fullBlocksPerGridGrid, blockSize, 0, simulation_stream>>>(gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernComputeIndices<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, gridResolution,
      gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    thrust::sort_by_key(thrust::cuda::par_nosync.on(simulation_stream),
      dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
      dev_thrust_particleArrayIndices);

    kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // Reorder particle data into sorted grid-cell order.
    thrust::gather(thrust::cuda::par_nosync.on(simulation_stream),
      dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_pos, dev_thrust_pos2);
    thrust::gather(thrust::cuda::par_nosync.on(simulation_stream),
      dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_vel1, dev_thrust_vel2);
    thrust::gather(thrust::cuda::par_nosync.on(simulation_stream),
      dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects,
      dev_thrust_boidIds, dev_thrust_boidIds2);

    glm::vec3 *tempPos = dev_pos;
    dev_pos = dev_pos2;
    dev_pos2 = tempPos;

    thrust::device_ptr<glm::vec3> tempThrustPos = dev_thrust_pos;
    dev_thrust_pos = dev_thrust_pos2;
    dev_thrust_pos2 = tempThrustPos;

    uint32_t *tempBoidIds = dev_boidIds;
    dev_boidIds = dev_boidIds2;
    dev_boidIds2 = tempBoidIds;

    thrust::device_ptr<uint32_t> tempThrustBoidIds = dev_thrust_boidIds;
    dev_thrust_boidIds = dev_thrust_boidIds2;
    dev_thrust_boidIds2 = tempThrustBoidIds;

    glm::vec3 *tempVel = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = tempVel;

    thrust::device_ptr<glm::vec3> tempThrustVel = dev_thrust_vel1;
    dev_thrust_vel1 = dev_thrust_vel2;
    dev_thrust_vel2 = tempThrustVel;

    bool requestPacket = false;
    {
      std::lock_guard<std::mutex> lock(ready_request_network_packet_lock);
      if (phoneCount != 0 && ready_request_network_packet) {
        ready_request_network_packet = false;
        requestPacket = true;
      }
    }
    if (requestPacket) {
      requestNetworkPacket(cameras, cameraChanged, phoneCount);
    }

    kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, gridResolution, gridMinimum, gridInverseCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

    tempVel = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = tempVel;

    tempThrustVel = dev_thrust_vel1;
    dev_thrust_vel1 = dev_thrust_vel2;
    dev_thrust_vel2 = tempThrustVel;

    kernUpdatePos<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernUpdatePos failed!");
  
    cudaStreamSynchronize(simulation_stream);
}

void Boids::stepSimulationGlint(float dt, CameraCullData *cameras, bool *cameraChanged, uint32_t phoneCount) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    int gridResolution = gridSideCount;

#if DEBUG_GLINT_PIPELINE_TIMINGS || DEBUG_GLINT_TREE_UPDATES
    glintOctree.begin_debug_step();
#endif
    glintOctree.update_partition(dev_pos, simulation_stream);

    bool requestPacket = false;
    {
      std::lock_guard<std::mutex> lock(ready_request_network_packet_lock);
      if (phoneCount != 0 && ready_request_network_packet) {
        ready_request_network_packet = false;
        requestPacket = true;
      }
    }
    if (requestPacket) {
      requestNetworkPacket(cameras, cameraChanged, phoneCount);
    }

    glintOctree.update_filter(simulation_stream);
    glintOctree.update_velocity(dev_vel1, dev_vel2, simulation_stream);
    glintOctree.update_tree(simulation_stream);

    std::swap(dev_vel1, dev_vel2);
#if DEBUG_GLINT_PIPELINE_TIMINGS
    glintOctree.begin_debug_stage(GlintOctree::GLINT_TIMING_MOVE, simulation_stream);
#endif
    kernUpdatePos<<<fullBlocksPerGrid, blockSize, 0, simulation_stream>>>(numObjects, dt, dev_pos, dev_vel1);
#if DEBUG_GLINT_PIPELINE_TIMINGS
    glintOctree.end_debug_stage(GlintOctree::GLINT_TIMING_MOVE, simulation_stream);
#endif
    checkCUDAErrorWithLine("kernUpdatePos failed!");
  
    cudaStreamSynchronize(simulation_stream);
#if DEBUG_GLINT_PIPELINE_TIMINGS || DEBUG_GLINT_TREE_UPDATES
    glintOctree.finish_debug_step();
#endif
}

#if DEBUG_GLINT_PIPELINE_TIMINGS || DEBUG_GLINT_TREE_UPDATES
void Boids::printGlintDebugStats() {
#if SIMULATION_TYPE == GLINT_SIMULATION
    glintOctree.print_debug_stats();
#endif
}
#endif

Boids::GridSnapshot Boids::getNetworkGridSnapshot() {
  return {dev_gridStarts_snapshot, dev_gridEnds_snapshot,
    static_cast<uint32_t>(gridCellCount), static_cast<uint32_t>(gridSideCount),
    gridCellWidth, gridMinimum};
}

#if SIMULATION_TYPE == GLINT_SIMULATION
Boids::GlintSnapshot Boids::getNetworkGlintSnapshot() {
  return {dev_glintLeaves_snapshot, dev_glintBoidIndices_snapshot, glintSnapshotLeafCount};
}
#else
Boids::GlintSnapshot Boids::getNetworkGlintSnapshot() {
  return {};
}
#endif

void Boids::prepareSimulationSnapshotsForNetwork() {
#if DEBUG_GLINT_PIPELINE_TIMINGS && SIMULATION_TYPE == GLINT_SIMULATION
    glintOctree.begin_debug_stage(GlintOctree::GLINT_TIMING_SNAPSHOT_COPY, simulation_stream);
#endif
    cudaMemcpyAsync(dev_pos_snapshot, dev_pos, numObjects * sizeof(glm::vec3),
      cudaMemcpyDeviceToDevice, simulation_stream);

#if DEBUG_GLINT_PIPELINE_TIMINGS && SIMULATION_TYPE == GLINT_SIMULATION
    glintOctree.end_debug_stage(GlintOctree::GLINT_TIMING_SNAPSHOT_COPY, simulation_stream);
#endif
  #if SIMULATION_TYPE == GRID_SIMULATION
    cudaMemcpyAsync(dev_boidIds_snapshot, dev_boidIds, numObjects * sizeof(uint32_t),
      cudaMemcpyDeviceToDevice, simulation_stream);
    cudaMemcpyAsync(dev_gridStarts_snapshot, dev_gridCellStartIndices,
      gridCellCount * sizeof(int), cudaMemcpyDeviceToDevice, simulation_stream);
    cudaMemcpyAsync(dev_gridEnds_snapshot, dev_gridCellEndIndices,
      gridCellCount * sizeof(int), cudaMemcpyDeviceToDevice, simulation_stream);
  #elif SIMULATION_TYPE == GLINT_SIMULATION
    const auto leafData = glintOctree.get_leaf_data();
    glintSnapshotLeafCount = leafData.count;
    auto leafSizes = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
      GetGlintLeafSize{leafData.active_leaves, leafData.nodes});
#if DEBUG_GLINT_PIPELINE_TIMINGS
    glintOctree.begin_debug_stage(GlintOctree::GLINT_TIMING_SNAPSHOT_SCAN, simulation_stream);
#endif
    cub::DeviceScan::ExclusiveSum(glintSnapshotScanTemp, glintSnapshotScanTempBytes,
      leafSizes, dev_glintLeafOffsets_snapshot, glintSnapshotLeafCount, simulation_stream);
#if DEBUG_GLINT_PIPELINE_TIMINGS
    glintOctree.end_debug_stage(GlintOctree::GLINT_TIMING_SNAPSHOT_SCAN, simulation_stream);
    glintOctree.begin_debug_stage(GlintOctree::GLINT_TIMING_SNAPSHOT_LEAVES, simulation_stream);
#endif
    kernSnapshotGlintLeaves<<<glintSnapshotLeafCount, blockSize, 0, simulation_stream>>>(
      glintSnapshotLeafCount, leafData.active_leaves, leafData.nodes, leafData.partitions,
      dev_glintLeafOffsets_snapshot, dev_glintLeaves_snapshot, dev_glintBoidIndices_snapshot);
#if DEBUG_GLINT_PIPELINE_TIMINGS
    glintOctree.end_debug_stage(GlintOctree::GLINT_TIMING_SNAPSHOT_LEAVES, simulation_stream);
#endif
    checkCUDAErrorWithLine("snapshot GLINT leaves failed!");
  #endif
      cudaEventRecord(simulationSnapshotReadyEvent, simulation_stream);
}

void releaseNetworkBuffers();

void Boids::endSimulation() {
  cudaStreamSynchronize(simulation_stream);
#if SIMULATION_TYPE == GLINT_SIMULATION
  glintOctree.deletion(simulation_stream);
  cudaFreeAsync(dev_glintLeaves_snapshot, simulation_stream);
  cudaFreeAsync(dev_glintBoidIndices_snapshot, simulation_stream);
  cudaFreeAsync(dev_glintLeafOffsets_snapshot, simulation_stream);
  cudaFreeAsync(glintSnapshotScanTemp, simulation_stream);
  dev_glintLeaves_snapshot = nullptr;
  dev_glintBoidIndices_snapshot = nullptr;
  dev_glintLeafOffsets_snapshot = nullptr;
  glintSnapshotScanTemp = nullptr;
  glintSnapshotLeafCount = 0;
#endif
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_boidIds);
  cudaFree(dev_boidIds2);
  cudaFree(dev_pos_snapshot);
  cudaFree(dev_boidIds_snapshot);
  cudaFree(dev_gridStarts_snapshot);
  cudaFree(dev_gridEnds_snapshot);
  cudaFree(dev_pos);
  cudaFree(dev_pos2);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  releaseNetworkBuffers();
  cudaStreamDestroy(simulation_stream);
  cudaEventDestroy(simulationSnapshotReadyEvent);
}
