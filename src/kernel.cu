#define GLM_FORCE_CUDA

#include <cuda.h>
#include <math_constants.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
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


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

#define rule1Distance 15.0f
#define rule2Distance 9.0f
#define rule3Distance 15.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 3.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 300.0f
#define spawnFlockRadius 90.0f

#define cullDistanceTolerance 36.0f
#define cullFrustumTolerance 0.12f

#define MAX_BOIDS_PER_PHONE 20000

#define DIST_VIEW 0
#define DIST_SPHERE 1
#define MIXED 2

#ifndef CULL_TYPE
#define CULL_TYPE DIST_VIEW
#endif

// MIXED keeps every nearby boid while a camera turns, then uses a slightly
// expanded frustum for the remainder of the camera's distance range.
#define mixedCullSphereFraction 0.5f

// Each signed 16-bit position-delta component represents 1/1024 simulation units.
// The encodable displacement range per packet is approximately [-32, 32] units.
#define networkPositionDeltaScale (1.0f / 1024.0f)
#define networkPositionDepthMaxDistance (Boids::cullDistance + cullDistanceTolerance)

// All three uint16_t components set to this value denote a free/deleted slot.
#define networkUpdateFreeComponent 0xFFFFu

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

Boids::CameraCullData *dev_networkCameras;
float **dev_phoneDistanceBuffers;
uint32_t **dev_phoneSortedBoidIndices;
int32_t *dev_phoneVisibleCounts;
uint32_t networkPhoneCount = 0;
std::vector<float *> hostPhoneDistanceBuffers;
std::vector<uint32_t *> hostPhoneSortedBoidIndices;

int **dev_boidSlotInPacketByPhone;
glm::vec3 **dev_clientPositionsByPhone;
Boids::UpdateBoidData **dev_boidUpdatePackets;
Boids::NewBoidData **dev_boidNewPackets;
uint32_t **dev_packetEpochsByPhone;
uint32_t **dev_boidIdBySlotInPacket;
uint8_t **dev_freeSlotFlagsByPhone;
uint8_t **dev_newBoidFlagsByPhone;
uint32_t **dev_freeSlotsByPhone;
uint32_t **dev_newRanksByPhone;
uint32_t *dev_phoneFreeCounts;
uint32_t *dev_phoneNewCounts;
uint32_t *dev_phoneUpdatePayloadLengths;
uint32_t *dev_phoneNewPayloadLengths;
void *dev_packetCompactionTempStorage;
size_t packetCompactionTempStorageBytes = 0;
uint32_t networkEpoch = 0;
std::vector<int *> hostBoidSlotInPacketByPhone;
std::vector<glm::vec3 *> hostClientPositionsByPhone;
std::vector<Boids::UpdateBoidData *> hostBoidUpdatePackets;
std::vector<Boids::NewBoidData *> hostBoidNewPackets;
std::vector<uint32_t *> hostPacketEpochsByPhone;
std::vector<uint32_t *> hostBoidIdBySlotInPacket;
std::vector<uint8_t *> hostFreeSlotFlagsByPhone;
std::vector<uint8_t *> hostNewBoidFlagsByPhone;
std::vector<uint32_t *> hostFreeSlotsByPhone;
std::vector<uint32_t *> hostNewRanksByPhone;

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
  kernInitializeBoidIds<<<fullBlocksPerGrid, blockSize>>>(N, dev_boidIds);
  checkCUDAErrorWithLine("kernInitializeBoidIds failed!");

  // Initialize velocity to random values
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_vel1, maxSpeed);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  kernGenerateFlockPositions<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos);
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

  // needed for use with thrust
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
  dev_thrust_vel1 = thrust::device_pointer_cast(dev_vel1);
  dev_thrust_vel2 = thrust::device_pointer_cast(dev_vel2);
  dev_thrust_pos = thrust::device_pointer_cast(dev_pos);
  dev_thrust_pos2 = thrust::device_pointer_cast(dev_pos2);
  dev_thrust_boidIds = thrust::device_pointer_cast(dev_boidIds);
  dev_thrust_boidIds2 = thrust::device_pointer_cast(dev_boidIds2);

  cudaDeviceSynchronize();
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

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
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

    // rule 1
    if (glm::distance(boidPos, pos[neighborBoid]) < rule1Distance) {
      perceived_center += pos[neighborBoid];
      num_neighbors_r1++;
    }

    // rule 2
    if (glm::distance(boidPos, pos[neighborBoid]) < rule2Distance) {
      c -= (pos[neighborBoid] - boidPos);
    }

    // rule 3
    if (glm::distance(boidPos, pos[neighborBoid]) < rule3Distance) {
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

__device__ bool isInsideCameraFrustum(const glm::vec3 &position,
  const Boids::CameraCullData &camera, float tolerance) {
  const glm::vec4 clipPosition = camera.projMat * glm::vec4(position, 1.0f);
  if (clipPosition.w <= 0.0f) {
    return false;
  }

  const float clipLimit = clipPosition.w * (1.0f + tolerance);
  return clipPosition.x >= -clipLimit && clipPosition.x <= clipLimit &&
    clipPosition.y >= -clipLimit && clipPosition.y <= clipLimit &&
    clipPosition.z >= -clipLimit && clipPosition.z <= clipLimit;
}

__device__ bool shouldBoidSurviveNaive(uint32_t boidIndex,
  const glm::vec3 *positions, const Boids::CameraCullData &camera,
  float *distanceToCamera) {
  const glm::vec3 offset = positions[boidIndex] - camera.position;
  const float distance = glm::length(offset);
  *distanceToCamera = distance;

#if CULL_TYPE == DIST_VIEW
  return distance <= Boids::cullDistance + cullDistanceTolerance &&
    isInsideCameraFrustum(positions[boidIndex], camera, 0.0f);
#elif CULL_TYPE == DIST_SPHERE
  return distance <= Boids::cullDistance + cullDistanceTolerance;
#elif CULL_TYPE == MIXED
  if (distance <= Boids::cullDistance * mixedCullSphereFraction) {
    return true;
  }
  return distance <= Boids::cullDistance + cullDistanceTolerance &&
    isInsideCameraFrustum(positions[boidIndex], camera, cullFrustumTolerance);
#else
#error "CULL_TYPE must be DIST_VIEW, DIST_SPHERE, or MIXED"
#endif
}

__global__ void kernCullBoidsByCameras(uint32_t boidCount,
  const glm::vec3 *positions, const Boids::CameraCullData *cameras,
  uint32_t phoneCount, float **distancesByPhone, uint32_t **sortedBoidIndicesByPhone) {
  const uint32_t boidIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (boidIndex >= boidCount) {
    return;
  }

  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    float distanceToCamera;
    const bool survives = shouldBoidSurviveNaive(boidIndex, positions,
      cameras[phoneIndex], &distanceToCamera);
    distancesByPhone[phoneIndex][boidIndex] = survives ? distanceToCamera : CUDART_INF_F;
    sortedBoidIndicesByPhone[phoneIndex][boidIndex] = boidIndex;
  }
}

void Boids::initializeNetworkPhones(const CameraCullData *cameras, uint32_t phoneCount) {
  for (float *distanceBuffer : hostPhoneDistanceBuffers) {
    cudaFree(distanceBuffer);
  }
  for (uint32_t *sortedIndices : hostPhoneSortedBoidIndices) {
    cudaFree(sortedIndices);
  }
  for (int *slotMap : hostBoidSlotInPacketByPhone) {
    cudaFree(slotMap);
  }
  for (glm::vec3 *clientPositions : hostClientPositionsByPhone) {
    cudaFree(clientPositions);
  }
  for (Boids::UpdateBoidData *updatePacket : hostBoidUpdatePackets) {
    cudaFree(updatePacket);
  }
  for (Boids::NewBoidData *newPacket : hostBoidNewPackets) {
    cudaFree(newPacket);
  }
  for (uint32_t *packetEpochs : hostPacketEpochsByPhone) {
    cudaFree(packetEpochs);
  }
  for (uint32_t *boidIdsBySlot : hostBoidIdBySlotInPacket) {
    cudaFree(boidIdsBySlot);
  }
  for (uint8_t *freeFlags : hostFreeSlotFlagsByPhone) {
    cudaFree(freeFlags);
  }
  for (uint8_t *newFlags : hostNewBoidFlagsByPhone) {
    cudaFree(newFlags);
  }
  for (uint32_t *freeSlots : hostFreeSlotsByPhone) {
    cudaFree(freeSlots);
  }
  for (uint32_t *newRanks : hostNewRanksByPhone) {
    cudaFree(newRanks);
  }
  for (uint32_t *boidIdsBySlot : hostBoidIdBySlotInPacket) {
    cudaFree(boidIdsBySlot);
  }
  for (uint8_t *freeFlags : hostFreeSlotFlagsByPhone) {
    cudaFree(freeFlags);
  }
  for (uint8_t *newFlags : hostNewBoidFlagsByPhone) {
    cudaFree(newFlags);
  }
  for (uint32_t *freeSlots : hostFreeSlotsByPhone) {
    cudaFree(freeSlots);
  }
  for (uint32_t *newRanks : hostNewRanksByPhone) {
    cudaFree(newRanks);
  }
  hostPhoneDistanceBuffers.clear();
  hostPhoneSortedBoidIndices.clear();
  hostBoidSlotInPacketByPhone.clear();
  hostClientPositionsByPhone.clear();
  hostBoidUpdatePackets.clear();
  hostBoidNewPackets.clear();
  hostPacketEpochsByPhone.clear();
  hostBoidIdBySlotInPacket.clear();
  hostFreeSlotFlagsByPhone.clear();
  hostNewBoidFlagsByPhone.clear();
  hostFreeSlotsByPhone.clear();
  hostNewRanksByPhone.clear();
  if (dev_networkCameras != nullptr) cudaFree(dev_networkCameras);
  if (dev_phoneDistanceBuffers != nullptr) cudaFree(dev_phoneDistanceBuffers);
  if (dev_phoneSortedBoidIndices != nullptr) cudaFree(dev_phoneSortedBoidIndices);
  if (dev_phoneVisibleCounts != nullptr) cudaFree(dev_phoneVisibleCounts);
  if (dev_boidSlotInPacketByPhone != nullptr) cudaFree(dev_boidSlotInPacketByPhone);
  if (dev_clientPositionsByPhone != nullptr) cudaFree(dev_clientPositionsByPhone);
  if (dev_boidUpdatePackets != nullptr) cudaFree(dev_boidUpdatePackets);
  if (dev_boidNewPackets != nullptr) cudaFree(dev_boidNewPackets);
  if (dev_packetEpochsByPhone != nullptr) cudaFree(dev_packetEpochsByPhone);
  if (dev_boidIdBySlotInPacket != nullptr) cudaFree(dev_boidIdBySlotInPacket);
  if (dev_freeSlotFlagsByPhone != nullptr) cudaFree(dev_freeSlotFlagsByPhone);
  if (dev_newBoidFlagsByPhone != nullptr) cudaFree(dev_newBoidFlagsByPhone);
  if (dev_freeSlotsByPhone != nullptr) cudaFree(dev_freeSlotsByPhone);
  if (dev_newRanksByPhone != nullptr) cudaFree(dev_newRanksByPhone);
  if (dev_phoneFreeCounts != nullptr) cudaFree(dev_phoneFreeCounts);
  if (dev_phoneNewCounts != nullptr) cudaFree(dev_phoneNewCounts);
  if (dev_phoneUpdatePayloadLengths != nullptr) cudaFree(dev_phoneUpdatePayloadLengths);
  if (dev_phoneNewPayloadLengths != nullptr) cudaFree(dev_phoneNewPayloadLengths);
  if (dev_packetCompactionTempStorage != nullptr) cudaFree(dev_packetCompactionTempStorage);

  dev_networkCameras = nullptr;
  dev_phoneDistanceBuffers = nullptr;
  dev_phoneSortedBoidIndices = nullptr;
  dev_phoneVisibleCounts = nullptr;
  dev_boidSlotInPacketByPhone = nullptr;
  dev_clientPositionsByPhone = nullptr;
  dev_boidUpdatePackets = nullptr;
  dev_boidNewPackets = nullptr;
  dev_packetEpochsByPhone = nullptr;
  dev_boidIdBySlotInPacket = nullptr;
  dev_freeSlotFlagsByPhone = nullptr;
  dev_newBoidFlagsByPhone = nullptr;
  dev_freeSlotsByPhone = nullptr;
  dev_newRanksByPhone = nullptr;
  dev_phoneFreeCounts = nullptr;
  dev_phoneNewCounts = nullptr;
  dev_phoneUpdatePayloadLengths = nullptr;
  dev_phoneNewPayloadLengths = nullptr;
  dev_packetCompactionTempStorage = nullptr;
  packetCompactionTempStorageBytes = 0;
  networkEpoch = 0;
  networkPhoneCount = phoneCount;
  if (phoneCount == 0) {
    return;
  }

  hostPhoneDistanceBuffers.resize(phoneCount);
  hostPhoneSortedBoidIndices.resize(phoneCount);
  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    cudaMalloc((void **)&hostPhoneDistanceBuffers[phoneIndex], numObjects * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc phone distance buffer failed!");
    cudaMalloc((void **)&hostPhoneSortedBoidIndices[phoneIndex], numObjects * sizeof(uint32_t));
    checkCUDAErrorWithLine("cudaMalloc phone sorted-index buffer failed!");
  }

  cudaMalloc((void **)&dev_networkCameras, phoneCount * sizeof(Boids::CameraCullData));
  checkCUDAErrorWithLine("cudaMalloc network cameras failed!");
  cudaMemcpy(dev_networkCameras, cameras, phoneCount * sizeof(Boids::CameraCullData),
    cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("cudaMemcpy initial network cameras failed!");
  cudaMalloc((void **)&dev_phoneDistanceBuffers, phoneCount * sizeof(float *));
  checkCUDAErrorWithLine("cudaMalloc phone distance pointer array failed!");
  cudaMemcpy(dev_phoneDistanceBuffers, hostPhoneDistanceBuffers.data(),
    phoneCount * sizeof(float *), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("cudaMemcpy phone distance pointer array failed!");
  cudaMalloc((void **)&dev_phoneSortedBoidIndices, phoneCount * sizeof(uint32_t *));
  checkCUDAErrorWithLine("cudaMalloc phone sorted-index pointer array failed!");
  cudaMemcpy(dev_phoneSortedBoidIndices, hostPhoneSortedBoidIndices.data(),
    phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("cudaMemcpy phone sorted-index pointer array failed!");
  cudaMalloc((void **)&dev_phoneVisibleCounts, phoneCount * sizeof(int32_t));
  checkCUDAErrorWithLine("cudaMalloc phone visible count array failed!");

  hostBoidSlotInPacketByPhone.resize(phoneCount);
  hostClientPositionsByPhone.resize(phoneCount);
  hostBoidUpdatePackets.resize(phoneCount);
  hostBoidNewPackets.resize(phoneCount);
  hostPacketEpochsByPhone.resize(phoneCount);
  hostBoidIdBySlotInPacket.resize(phoneCount);
  hostFreeSlotFlagsByPhone.resize(phoneCount);
  hostNewBoidFlagsByPhone.resize(phoneCount);
  hostFreeSlotsByPhone.resize(phoneCount);
  hostNewRanksByPhone.resize(phoneCount);
  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    cudaMalloc((void **)&hostBoidSlotInPacketByPhone[phoneIndex], numObjects * sizeof(int));
    cudaMemset(hostBoidSlotInPacketByPhone[phoneIndex], 0xFF, numObjects * sizeof(int));
    cudaMalloc((void **)&hostClientPositionsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(glm::vec3));
    cudaMemset(hostClientPositionsByPhone[phoneIndex], 0, MAX_BOIDS_PER_PHONE * sizeof(glm::vec3));
    cudaMalloc((void **)&hostBoidUpdatePackets[phoneIndex],
      MAX_BOIDS_PER_PHONE * sizeof(Boids::UpdateBoidData));
    cudaMalloc((void **)&hostBoidNewPackets[phoneIndex],
      MAX_BOIDS_PER_PHONE * sizeof(Boids::NewBoidData));
    cudaMalloc((void **)&hostPacketEpochsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMemset(hostPacketEpochsByPhone[phoneIndex], 0, MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMalloc((void **)&hostBoidIdBySlotInPacket[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMemset(hostBoidIdBySlotInPacket[phoneIndex], 0xFF, MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMalloc((void **)&hostFreeSlotFlagsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint8_t));
    cudaMalloc((void **)&hostNewBoidFlagsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint8_t));
    cudaMalloc((void **)&hostFreeSlotsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMalloc((void **)&hostNewRanksByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    checkCUDAErrorWithLine("cudaMalloc network packet state failed!");
  }

  cudaMalloc((void **)&dev_boidSlotInPacketByPhone, phoneCount * sizeof(int *));
  cudaMalloc((void **)&dev_clientPositionsByPhone, phoneCount * sizeof(glm::vec3 *));
  cudaMalloc((void **)&dev_boidUpdatePackets, phoneCount * sizeof(Boids::UpdateBoidData *));
  cudaMalloc((void **)&dev_boidNewPackets, phoneCount * sizeof(Boids::NewBoidData *));
  cudaMalloc((void **)&dev_packetEpochsByPhone, phoneCount * sizeof(uint32_t *));
  cudaMalloc((void **)&dev_boidIdBySlotInPacket, phoneCount * sizeof(uint32_t *));
  cudaMalloc((void **)&dev_freeSlotFlagsByPhone, phoneCount * sizeof(uint8_t *));
  cudaMalloc((void **)&dev_newBoidFlagsByPhone, phoneCount * sizeof(uint8_t *));
  cudaMalloc((void **)&dev_freeSlotsByPhone, phoneCount * sizeof(uint32_t *));
  cudaMalloc((void **)&dev_newRanksByPhone, phoneCount * sizeof(uint32_t *));
  cudaMalloc((void **)&dev_phoneFreeCounts, phoneCount * sizeof(uint32_t));
  cudaMalloc((void **)&dev_phoneNewCounts, phoneCount * sizeof(uint32_t));
  cudaMalloc((void **)&dev_phoneUpdatePayloadLengths, phoneCount * sizeof(uint32_t));
  cudaMalloc((void **)&dev_phoneNewPayloadLengths, phoneCount * sizeof(uint32_t));
  cudaMemcpy(dev_boidSlotInPacketByPhone, hostBoidSlotInPacketByPhone.data(),
    phoneCount * sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_clientPositionsByPhone, hostClientPositionsByPhone.data(),
    phoneCount * sizeof(glm::vec3 *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_boidUpdatePackets, hostBoidUpdatePackets.data(),
    phoneCount * sizeof(Boids::UpdateBoidData *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_boidNewPackets, hostBoidNewPackets.data(),
    phoneCount * sizeof(Boids::NewBoidData *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_packetEpochsByPhone, hostPacketEpochsByPhone.data(),
    phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_boidIdBySlotInPacket, hostBoidIdBySlotInPacket.data(),
    phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_freeSlotFlagsByPhone, hostFreeSlotFlagsByPhone.data(),
    phoneCount * sizeof(uint8_t *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_newBoidFlagsByPhone, hostNewBoidFlagsByPhone.data(),
    phoneCount * sizeof(uint8_t *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_freeSlotsByPhone, hostFreeSlotsByPhone.data(),
    phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_newRanksByPhone, hostNewRanksByPhone.data(),
    phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("cudaMemcpy network packet state pointers failed!");

  thrust::counting_iterator<uint32_t> slotIndices(0);
  size_t freeSelectBytes = 0;
  size_t newSelectBytes = 0;
  cub::DeviceSelect::Flagged(nullptr, freeSelectBytes, slotIndices,
    hostFreeSlotFlagsByPhone[0], hostFreeSlotsByPhone[0], dev_phoneFreeCounts,
    MAX_BOIDS_PER_PHONE);
  cub::DeviceSelect::Flagged(nullptr, newSelectBytes, slotIndices,
    hostNewBoidFlagsByPhone[0], hostNewRanksByPhone[0], dev_phoneNewCounts,
    MAX_BOIDS_PER_PHONE);
  packetCompactionTempStorageBytes = imax(freeSelectBytes, newSelectBytes);
  cudaMalloc(&dev_packetCompactionTempStorage, packetCompactionTempStorageBytes);
  checkCUDAErrorWithLine("cudaMalloc packet compaction storage failed!");
}

__global__ void kernGetFirstInf(uint32_t boidCount, uint32_t phoneCount,
  float **distancesByPhone, int32_t *visibleCounts) {
  const uint32_t boidIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t phoneIndex = threadIdx.y + blockIdx.y * blockDim.y;
  if (boidIndex >= boidCount || phoneIndex >= phoneCount) {
    return;
  }

  const float *distances = distancesByPhone[phoneIndex];
  const bool currentIsInfinite = distances[boidIndex] == CUDART_INF_F;
  if (boidIndex == 0 && currentIsInfinite) {
    visibleCounts[phoneIndex] = -1;
  } else if (boidIndex == boidCount - 1 && !currentIsInfinite) {
    visibleCounts[phoneIndex] = imin(static_cast<int>(boidIndex), MAX_BOIDS_PER_PHONE - 1);
  } else if (!currentIsInfinite && distances[boidIndex + 1] == CUDART_INF_F) {
    visibleCounts[phoneIndex] = imin(static_cast<int>(boidIndex), MAX_BOIDS_PER_PHONE - 1);
  }
}

__device__ int16_t quantizeNetworkDelta(float delta) {
  const int quantized = static_cast<int>(roundf(delta / networkPositionDeltaScale));
  return static_cast<int16_t>(imax(-32768, imin(32767, quantized)));
}

__device__ void markUpdateSlotFree(Boids::UpdateBoidData &update) {
  update.pos_delta[0] = static_cast<int16_t>(networkUpdateFreeComponent);
  update.pos_delta[1] = static_cast<int16_t>(networkUpdateFreeComponent);
  update.pos_delta[2] = static_cast<int16_t>(networkUpdateFreeComponent);
}

__device__ bool usePolarNewBoidEncoding(const glm::vec3 &position,
  const Boids::CameraCullData &camera) {
#if CULL_TYPE == DIST_VIEW
  return false;
#elif CULL_TYPE == DIST_SPHERE
  return true;
#elif CULL_TYPE == MIXED
  return !isInsideCameraFrustum(position, camera, 0.0f);
#endif
}

__device__ uint8_t quantizeUnitFloat(float value) {
  return static_cast<uint8_t>(roundf(fminf(1.0f, fmaxf(0.0f, value)) * 255.0f));
}

__device__ uint16_t quantizeNetworkDepth(float distance) {
  return static_cast<uint16_t>(roundf(fminf(1.0f,
    fmaxf(0.0f, distance / networkPositionDepthMaxDistance)) * 65535.0f));
}

__device__ float dequantizeNetworkDepth(uint16_t depth) {
  return static_cast<float>(depth) * networkPositionDepthMaxDistance / 65535.0f;
}

__device__ glm::vec3 encodeNewBoidPosition(Boids::NewBoidData &boidNewData,
  uint16_t slot, const glm::vec3 &position, const Boids::CameraCullData &camera) {
  const glm::vec3 offset = position - camera.position;
  const float distance = glm::length(offset);
  const uint16_t depth = quantizeNetworkDepth(distance);
  bool usePolar = usePolarNewBoidEncoding(position, camera);

  if (!usePolar) {
    const glm::vec4 clipPosition = camera.projMat * glm::vec4(position, 1.0f);
    if (clipPosition.w > 0.0f) {
      const float ndcX = clipPosition.x / clipPosition.w;
      const float ndcY = clipPosition.y / clipPosition.w;
      const uint8_t encodedX = quantizeUnitFloat(ndcX * 0.5f + 0.5f);
      const uint8_t encodedY = quantizeUnitFloat(ndcY * 0.5f + 0.5f);
      boidNewData.new_idx_slot = slot;
      boidNewData.position.view = { encodedX, encodedY, depth };

      const glm::mat4 inverseProjView = glm::inverse(camera.projMat);
      const float decodedNdcX = static_cast<float>(encodedX) * 2.0f / 255.0f - 1.0f;
      const float decodedNdcY = static_cast<float>(encodedY) * 2.0f / 255.0f - 1.0f;
      glm::vec4 farPoint = inverseProjView * glm::vec4(decodedNdcX, decodedNdcY, 1.0f, 1.0f);
      farPoint /= farPoint.w;
      const glm::vec3 ray = glm::normalize(glm::vec3(farPoint) - camera.position);
      return camera.position + ray * dequantizeNetworkDepth(depth);
    }
    usePolar = true;
  }

  constexpr float pi = 3.14159265358979323846f;
  constexpr float twoPi = 6.28318530717958647692f;
  const glm::vec3 direction = distance > 1.0e-5f ? offset / distance : glm::vec3(0.0f, 0.0f, 1.0f);
  float azimuth = atan2f(direction.y, direction.x);
  if (azimuth < 0.0f) azimuth += twoPi;
  const float elevation = asinf(fminf(1.0f, fmaxf(-1.0f, direction.z)));
  const uint8_t encodedAzimuth = quantizeUnitFloat(azimuth / twoPi);
  const uint8_t encodedElevation = quantizeUnitFloat(elevation / pi + 0.5f);
  boidNewData.new_idx_slot = slot | Boids::NEW_BOID_POLAR_ENCODING_BIT;
  boidNewData.position.polar = { encodedAzimuth, encodedElevation, depth };

  const float decodedAzimuth = static_cast<float>(encodedAzimuth) * twoPi / 255.0f;
  const float decodedElevation = static_cast<float>(encodedElevation) * pi / 255.0f - pi * 0.5f;
  const float cosElevation = cosf(decodedElevation);
  const glm::vec3 decodedDirection(cosElevation * cosf(decodedAzimuth),
    cosElevation * sinf(decodedAzimuth), sinf(decodedElevation));
  return camera.position + decodedDirection * dequantizeNetworkDepth(depth);
}

__global__ void kernBuildNetworkPackets(uint32_t phoneCount, uint32_t currentEpoch,
  const glm::vec3 *boidPositions, const uint32_t *boidIds,
  uint32_t **sortedBoidIndicesByPhone, const int32_t *visibleLastIndices,
  int **boidSlotInPacketByPhone, glm::vec3 **clientPositionsByPhone,
  Boids::UpdateBoidData **boidUpdatePackets, Boids::NewBoidData **boidNewPackets,
  uint32_t **packetEpochsByPhone, uint8_t **newBoidFlagsByPhone) {
  const uint32_t linearIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t phoneIndex = linearIndex / MAX_BOIDS_PER_PHONE;
  const uint32_t selectedRank = linearIndex % MAX_BOIDS_PER_PHONE;
  if (phoneIndex >= phoneCount) {
    return;
  }
  const int32_t visibleLastIndex = visibleLastIndices[phoneIndex];
  if (visibleLastIndex < 0 || selectedRank > static_cast<uint32_t>(visibleLastIndex)) {
    newBoidFlagsByPhone[phoneIndex][selectedRank] = 0;
    return;
  }

  const uint32_t currentBoidIndex = sortedBoidIndicesByPhone[phoneIndex][selectedRank];
  const uint32_t stableBoidId = boidIds[currentBoidIndex];
  const int slotInPacket = boidSlotInPacketByPhone[phoneIndex][stableBoidId];
  if (slotInPacket > -1) {
    newBoidFlagsByPhone[phoneIndex][selectedRank] = 0;
    // Existing boid: encode current position relative to the client's estimate.
    glm::vec3 &clientPosition = clientPositionsByPhone[phoneIndex][slotInPacket];
    const glm::vec3 delta = boidPositions[currentBoidIndex] - clientPosition;
    int16_t deltaX = quantizeNetworkDelta(delta.x);
    int16_t deltaY = quantizeNetworkDelta(delta.y);
    int16_t deltaZ = quantizeNetworkDelta(delta.z);
    if (deltaX == -1 && deltaY == -1 && deltaZ == -1) {
      deltaZ = -2;
    }

    Boids::UpdateBoidData &boidUpdatePacketData = boidUpdatePackets[phoneIndex][slotInPacket];
    boidUpdatePacketData.pos_delta[0] = deltaX;
    boidUpdatePacketData.pos_delta[1] = deltaY;
    boidUpdatePacketData.pos_delta[2] = deltaZ;

    clientPosition += glm::vec3(deltaX, deltaY, deltaZ) * networkPositionDeltaScale;
    packetEpochsByPhone[phoneIndex][slotInPacket] = currentEpoch;
  } else {
    // Mark if new boid to create later
    newBoidFlagsByPhone[phoneIndex][selectedRank] = 1;
  }
}

__global__ void kernClearFreePacketSlots(uint32_t phoneCount, uint32_t currentEpoch,
  int **boidSlotInPacketByPhone, uint32_t **boidIdBySlotInPacket,
  uint32_t **packetEpochsByPhone, Boids::UpdateBoidData **boidUpdatePackets,
  uint8_t **freeSlotFlagsByPhone) {
  const uint32_t linearIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t phoneIndex = linearIndex / MAX_BOIDS_PER_PHONE;
  const uint32_t slot = linearIndex % MAX_BOIDS_PER_PHONE;
  if (phoneIndex >= phoneCount) {
    return;
  }

  uint32_t &epoch = packetEpochsByPhone[phoneIndex][slot];
  const bool isFree = epoch == UINT32_MAX || epoch < currentEpoch;
  if (isFree) {
    if (epoch < currentEpoch) {
      const uint32_t stableBoidId = boidIdBySlotInPacket[phoneIndex][slot];
      if (stableBoidId != UINT32_MAX) {
        boidSlotInPacketByPhone[phoneIndex][stableBoidId] = -1;
      }
      boidIdBySlotInPacket[phoneIndex][slot] = UINT32_MAX;
      epoch = UINT32_MAX;
    }
    markUpdateSlotFree(boidUpdatePackets[phoneIndex][slot]);
  }
  freeSlotFlagsByPhone[phoneIndex][slot] = isFree;
}

__global__ void kernAssignNewPacketSlots(uint32_t phoneCount, uint32_t currentEpoch,
  const glm::vec3 *boidPositions, const uint32_t *boidIds,
  const Boids::CameraCullData *cameras, uint32_t **sortedBoidIndicesByPhone,
  uint32_t **newRanksByPhone, const uint32_t *newCounts,
  uint32_t **freeSlotsByPhone, const uint32_t *freeCounts,
  int **boidSlotInPacketByPhone, uint32_t **boidIdBySlotInPacket,
  uint32_t **packetEpochsByPhone, glm::vec3 **clientPositionsByPhone,
  Boids::NewBoidData **boidNewPackets) {
  const uint32_t linearIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t phoneIndex = linearIndex / MAX_BOIDS_PER_PHONE;
  const uint32_t newIndex = linearIndex % MAX_BOIDS_PER_PHONE;
  if (phoneIndex >= phoneCount || newIndex >= newCounts[phoneIndex] ||
      newIndex >= freeCounts[phoneIndex]) {
    return;
  }

  const uint32_t selectedRank = newRanksByPhone[phoneIndex][newIndex];
  const uint32_t slot = freeSlotsByPhone[phoneIndex][newIndex];
  const uint32_t currentBoidIndex = sortedBoidIndicesByPhone[phoneIndex][selectedRank];
  const uint32_t stableBoidId = boidIds[currentBoidIndex];

  boidSlotInPacketByPhone[phoneIndex][stableBoidId] = static_cast<int>(slot);
  boidIdBySlotInPacket[phoneIndex][slot] = stableBoidId;
  packetEpochsByPhone[phoneIndex][slot] = currentEpoch;

  // New data is compact by newIndex; this return value is precisely the state
  // reconstructed from the quantized absolute record.
  Boids::NewBoidData &boidNewData = boidNewPackets[phoneIndex][newIndex];
  clientPositionsByPhone[phoneIndex][slot] = encodeNewBoidPosition(boidNewData,
    static_cast<uint16_t>(slot), boidPositions[currentBoidIndex], cameras[phoneIndex]);
}

__global__ void kernDeriveNetworkPayloadLengths(uint32_t phoneCount,
  uint32_t currentEpoch, uint32_t **packetEpochsByPhone,
  const uint32_t *newCounts, uint32_t *updatePayloadLengths,
  uint32_t *newPayloadLengths) {
  const uint32_t phoneIndex = blockIdx.x;
  if (phoneIndex >= phoneCount) {
    return;
  }

  uint32_t localHighWater = 0;
  for (uint32_t slot = threadIdx.x; slot < MAX_BOIDS_PER_PHONE; slot += blockDim.x) {
    if (packetEpochsByPhone[phoneIndex][slot] == currentEpoch) {
      localHighWater = slot + 1;
    }
  }

  __shared__ uint32_t blockHighWater[blockSize];
  blockHighWater[threadIdx.x] = localHighWater;
  __syncthreads();
  for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      blockHighWater[threadIdx.x] = imax(blockHighWater[threadIdx.x],
        blockHighWater[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    updatePayloadLengths[phoneIndex] = blockHighWater[0] * sizeof(Boids::UpdateBoidData);
    newPayloadLengths[phoneIndex] = newCounts[phoneIndex] * sizeof(Boids::NewBoidData);
  }
}

void Boids::createNetworkPackets() {
  if (networkPhoneCount == 0) {
    return;
  }

  dim3 blocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernCullBoidsByCameras<<<blocksPerGrid, blockSize>>>(numObjects, dev_pos,
    dev_networkCameras, networkPhoneCount, dev_phoneDistanceBuffers, dev_phoneSortedBoidIndices);
  checkCUDAErrorWithLine("kernCullBoidsByCameras failed!");

  for (uint32_t phoneIndex = 0; phoneIndex < networkPhoneCount; ++phoneIndex) {
    thrust::device_ptr<float> distanceBegin =
      thrust::device_pointer_cast(hostPhoneDistanceBuffers[phoneIndex]);
    thrust::device_ptr<uint32_t> boidIndexBegin =
      thrust::device_pointer_cast(hostPhoneSortedBoidIndices[phoneIndex]);
    thrust::sort_by_key(distanceBegin, distanceBegin + numObjects, boidIndexBegin);
  }

  dim3 cullCountBlockSize(32, 4);
  dim3 cullCountBlocksPerGrid((numObjects + cullCountBlockSize.x - 1) / cullCountBlockSize.x,
    (networkPhoneCount + cullCountBlockSize.y - 1) / cullCountBlockSize.y);
  kernGetFirstInf<<<cullCountBlocksPerGrid, cullCountBlockSize>>>(numObjects, networkPhoneCount,
    dev_phoneDistanceBuffers, dev_phoneVisibleCounts);
  checkCUDAErrorWithLine("kernGetFirstInf failed!");

  const uint32_t packetWorkItems = networkPhoneCount * MAX_BOIDS_PER_PHONE;
  dim3 packetBlocksPerGrid((packetWorkItems + blockSize - 1) / blockSize);
  const uint32_t currentEpoch = ++networkEpoch;
  kernBuildNetworkPackets<<<packetBlocksPerGrid, blockSize>>>(networkPhoneCount,
    currentEpoch, dev_pos, dev_boidIds, dev_phoneSortedBoidIndices, dev_phoneVisibleCounts,
    dev_boidSlotInPacketByPhone, dev_clientPositionsByPhone, dev_boidUpdatePackets,
    dev_boidNewPackets, dev_packetEpochsByPhone, dev_newBoidFlagsByPhone);
  checkCUDAErrorWithLine("kernBuildNetworkPackets failed!");

  kernClearFreePacketSlots<<<packetBlocksPerGrid, blockSize>>>(networkPhoneCount,
    currentEpoch, dev_boidSlotInPacketByPhone, dev_boidIdBySlotInPacket,
    dev_packetEpochsByPhone, dev_boidUpdatePackets, dev_freeSlotFlagsByPhone);
  checkCUDAErrorWithLine("kernClearFreePacketSlots failed!");

  thrust::counting_iterator<uint32_t> slotIndices(0);
  for (uint32_t phoneIndex = 0; phoneIndex < networkPhoneCount; ++phoneIndex) {
    size_t tempStorageBytes = packetCompactionTempStorageBytes;
    cub::DeviceSelect::Flagged(dev_packetCompactionTempStorage, tempStorageBytes,
      slotIndices, hostFreeSlotFlagsByPhone[phoneIndex], hostFreeSlotsByPhone[phoneIndex],
      dev_phoneFreeCounts + phoneIndex, MAX_BOIDS_PER_PHONE);
    cub::DeviceSelect::Flagged(dev_packetCompactionTempStorage, tempStorageBytes,
      slotIndices, hostNewBoidFlagsByPhone[phoneIndex], hostNewRanksByPhone[phoneIndex],
      dev_phoneNewCounts + phoneIndex, MAX_BOIDS_PER_PHONE);
  }

  std::vector<uint32_t> newBoidCounts(networkPhoneCount);
  std::vector<int32_t> visibleLastIndices(networkPhoneCount);
  cudaMemcpy(newBoidCounts.data(), dev_phoneNewCounts,
    networkPhoneCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("cudaMemcpy compacted new-boid counts failed!");
  cudaMemcpy(visibleLastIndices.data(), dev_phoneVisibleCounts,
    networkPhoneCount * sizeof(int32_t), cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("cudaMemcpy visible boid counts failed!");
  for (uint32_t phoneIndex = 0; phoneIndex < networkPhoneCount; ++phoneIndex) {
    const uint32_t selectedCount = visibleLastIndices[phoneIndex] < 0 ? 0u :
      static_cast<uint32_t>(visibleLastIndices[phoneIndex]) + 1u;
    const uint32_t updateCount = selectedCount - newBoidCounts[phoneIndex];
    printf("Phone %u compacted packet data: %u boid updates, %u new boids\n", phoneIndex,
      updateCount, newBoidCounts[phoneIndex]);
  }

  kernAssignNewPacketSlots<<<packetBlocksPerGrid, blockSize>>>(networkPhoneCount,
    currentEpoch, dev_pos, dev_boidIds, dev_networkCameras, dev_phoneSortedBoidIndices,
    dev_newRanksByPhone, dev_phoneNewCounts, dev_freeSlotsByPhone, dev_phoneFreeCounts,
    dev_boidSlotInPacketByPhone, dev_boidIdBySlotInPacket, dev_packetEpochsByPhone,
    dev_clientPositionsByPhone, dev_boidNewPackets);
  checkCUDAErrorWithLine("kernAssignNewPacketSlots failed!");

  kernDeriveNetworkPayloadLengths<<<networkPhoneCount, blockSize>>>(networkPhoneCount,
    currentEpoch, dev_packetEpochsByPhone, dev_phoneNewCounts,
    dev_phoneUpdatePayloadLengths, dev_phoneNewPayloadLengths);
  checkCUDAErrorWithLine("kernDeriveNetworkPayloadLengths failed!");

  cudaDeviceSynchronize();
}

void Boids::cullBoidsByCameras(const CameraCullData *cameras, bool *cameraChanged,
  uint32_t phoneCount) {
  if (phoneCount != networkPhoneCount) {
    initializeNetworkPhones(cameras, phoneCount);
  }
  if (phoneCount == 0) {
    return;
  }

  for (uint32_t phone_idx = 0; phone_idx < phoneCount; phone_idx++) {
    if (!cameraChanged[phone_idx]) {
      continue;
    }
    cudaMemcpy(dev_networkCameras + phone_idx, cameras + phone_idx,
      sizeof(CameraCullData), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy changed network camera failed!");
    cameraChanged[phone_idx] = false;
  }

  createNetworkPackets();
}

float* const* Boids::getDevicePhoneDistances() {
  return dev_phoneDistanceBuffers;
}

const int32_t* Boids::getDevicePhoneVisibleCounts() {
  return dev_phoneVisibleCounts;
}

const uint32_t* Boids::getDevicePhoneUpdatePayloadLengths() {
  return dev_phoneUpdatePayloadLengths;
}

const uint32_t* Boids::getDevicePhoneNewPayloadLengths() {
  return dev_phoneNewPayloadLengths;
}

Boids::UpdateBoidData* const* Boids::getDevicePhoneUpdatePayloads() {
  return dev_boidUpdatePackets;
}

Boids::NewBoidData* const* Boids::getDevicePhoneNewPayloads() {
  return dev_boidNewPackets;
}

void Boids::stepSimulationCoherentGrid(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    int gridResolution = gridSideCount;

    dim3 fullBlocksPerGridGrid((gridCellCount + blockSize - 1) / blockSize);

    kernResetIntBuffer<<<fullBlocksPerGridGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernResetIntBuffer<<<fullBlocksPerGridGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridResolution,
      gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // Reorder particle data into sorted grid-cell order.
    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_pos, dev_thrust_pos2);
    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_vel1, dev_thrust_vel2);
    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects,
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

    kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridResolution, gridMinimum, gridInverseCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

    tempVel = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = tempVel;

    tempThrustVel = dev_thrust_vel1;
    dev_thrust_vel1 = dev_thrust_vel2;
    dev_thrust_vel2 = tempThrustVel;

    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // kernComputeCameraCulling<<<fullBlocksPerGrid, blockSize>>>();
    // checkCUDAErrorWithLine("kernComputeCameraCulling failed!");

    cudaDeviceSynchronize();
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_boidIds);
  cudaFree(dev_boidIds2);
  cudaFree(dev_pos);
  cudaFree(dev_pos2);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  for (float *distanceBuffer : hostPhoneDistanceBuffers) {
    cudaFree(distanceBuffer);
  }
  for (int *slotMap : hostBoidSlotInPacketByPhone) {
    cudaFree(slotMap);
  }
  for (glm::vec3 *clientPositions : hostClientPositionsByPhone) {
    cudaFree(clientPositions);
  }
  for (Boids::UpdateBoidData *updatePacket : hostBoidUpdatePackets) {
    cudaFree(updatePacket);
  }
  for (Boids::NewBoidData *newPacket : hostBoidNewPackets) {
    cudaFree(newPacket);
  }
  for (uint32_t *packetEpochs : hostPacketEpochsByPhone) {
    cudaFree(packetEpochs);
  }
  cudaFree(dev_networkCameras);
  cudaFree(dev_phoneDistanceBuffers);
  cudaFree(dev_phoneSortedBoidIndices);
  cudaFree(dev_phoneVisibleCounts);
  cudaFree(dev_boidSlotInPacketByPhone);
  cudaFree(dev_clientPositionsByPhone);
  cudaFree(dev_boidUpdatePackets);
  cudaFree(dev_boidNewPackets);
  cudaFree(dev_packetEpochsByPhone);
  cudaFree(dev_boidIdBySlotInPacket);
  cudaFree(dev_freeSlotFlagsByPhone);
  cudaFree(dev_newBoidFlagsByPhone);
  cudaFree(dev_freeSlotsByPhone);
  cudaFree(dev_newRanksByPhone);
  cudaFree(dev_phoneFreeCounts);
  cudaFree(dev_phoneNewCounts);
  cudaFree(dev_phoneUpdatePayloadLengths);
  cudaFree(dev_phoneNewPayloadLengths);
  cudaFree(dev_packetCompactionTempStorage);
}
