#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/scan.h>

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

#define cullDistance 240.0f
#define cullDistanceTolerance 36.0f
#define cullFrustumTolerance 0.12f

#define spawnFlockRadius 90.0f

#define flockMaxRadius rule1Distance
#define flockMaxAxisDeviationDegrees 15.0f
#define flockYawBins 24
#define flockPitchBins 12
#define flockDirectionBinCount (flockYawBins * flockPitchBins)
#define flockDirectionKeyBits 9
#define flockDirectionKeyMask ((1 << flockDirectionKeyBits) - 1)

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
glm::vec3 *dev_pos;
glm::vec3 *dev_pos2;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

int *dev_particleArrayIndices;
int *dev_particleGridIndices;
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;
thrust::device_ptr<glm::vec3> dev_thrust_vel1;
thrust::device_ptr<glm::vec3> dev_thrust_vel2;
thrust::device_ptr<glm::vec3> dev_thrust_pos;
thrust::device_ptr<glm::vec3> dev_thrust_pos2;

int *dev_gridCellStartIndices;
int *dev_gridCellEndIndices;

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

float flockCellWidth;
float flockInverseCellWidth;
int flockGridSideCount;
glm::vec3 flockGridMinimum;

uint32_t *dev_flockKeys;
uint32_t *dev_flockIndices;
int *dev_flockHeadFlags;
int *dev_flockGroupIds;
Boids::Flock *dev_flocks;
Boids::CompressedBoid *dev_compressedBoids;
uint32_t *dev_flockCount;

Boids::CameraCullData *dev_networkCameras;
uint8_t **dev_phonePacketBuffers;
uint32_t *dev_phonePacketSizes;
uint32_t networkPhoneCount = 0;
size_t phonePacketCapacityBytes = 0;
std::vector<uint8_t *> hostPhonePacketBuffers;

thrust::device_ptr<uint32_t> dev_thrust_flockKeys;
thrust::device_ptr<uint32_t> dev_thrust_flockIndices;
thrust::device_ptr<int> dev_thrust_flockHeadFlags;
thrust::device_ptr<int> dev_thrust_flockGroupIds;

/******************
* initSimulation *
******************/

__device__ bool shouldBoidSurviveNaive(
  int i,
  const glm::vec3 *pos,
  const glm::vec3 &cameraPosition,
  const glm::mat4 &projMat) {
  const glm::vec3 offset = pos[i] - cameraPosition;
  if (glm::length(offset) > cullDistance + cullDistanceTolerance) {
    return false;
  }

  const glm::vec4 clipPosition = projMat * glm::vec4(pos[i], 1.0f);
  if (clipPosition.w <= 0.0f) {
    return false;
  }

  const float clipLimit = clipPosition.w * (1.0f + cullFrustumTolerance);
  return clipPosition.x >= -clipLimit && clipPosition.x <= clipLimit &&
         clipPosition.y >= -clipLimit && clipPosition.y <= clipLimit &&
         clipPosition.z >= -clipLimit && clipPosition.z <= clipLimit;
}

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

__device__ float wrapAngle(float angle) {
  constexpr float twoPi = 6.28318530717958647692f;
  return angle < 0.0f ? angle + twoPi : angle;
}

__device__ float signedAngleDelta(float angle, float reference) {
  constexpr float pi = 3.14159265358979323846f;
  constexpr float twoPi = 6.28318530717958647692f;
  float delta = angle - reference;
  if (delta > pi) delta -= twoPi;
  if (delta < -pi) delta += twoPi;
  return delta;
}

__device__ glm::vec3 boidForward(const glm::vec3 &velocity) {
  float speed = glm::length(velocity);
  return speed > 0.00001f ? velocity / speed : glm::vec3(0.0f, 1.0f, 0.0f);
}

__device__ uint32_t directionBin(const glm::vec3 &forward) {
  constexpr float pi = 3.14159265358979323846f;
  float yaw = wrapAngle(atan2f(forward.y, forward.x));
  float pitch = asinf(fminf(1.0f, fmaxf(-1.0f, forward.z)));
  int yawBin = min(flockYawBins - 1, (int)(yaw * flockYawBins / (2.0f * pi)));
  int pitchBin = min(flockPitchBins - 1, (int)((pitch + 0.5f * pi) * flockPitchBins / pi));
  return pitchBin * flockYawBins + yawBin;
}

__device__ glm::vec3 directionBinCenter(uint32_t bin) {
  constexpr float pi = 3.14159265358979323846f;
  int yawBin = bin % flockYawBins;
  int pitchBin = bin / flockYawBins;
  float yaw = (yawBin + 0.5f) * (2.0f * pi / flockYawBins);
  float pitch = (pitchBin + 0.5f) * (pi / flockPitchBins) - 0.5f * pi;
  float cosPitch = cosf(pitch);
  return glm::vec3(cosPitch * cosf(yaw), cosPitch * sinf(yaw), sinf(pitch));
}

__device__ glm::vec3 flockCellCenter(uint32_t cell, int gridSideCount,
  glm::vec3 gridMinimum, float cellWidth) {
  int z = cell / (gridSideCount * gridSideCount);
  int remaining = cell - z * gridSideCount * gridSideCount;
  int y = remaining / gridSideCount;
  int x = remaining - y * gridSideCount;
  return gridMinimum + glm::vec3(
      (x + 0.5f) * cellWidth,
      (y + 0.5f) * cellWidth,
      (z + 0.5f) * cellWidth);
}

__device__ uint8_t packSignedRotation(float radians) {
  constexpr float pi = 3.14159265358979323846f;
  float maxRadians = flockMaxAxisDeviationDegrees * pi / 180.0f;
  float clamped = fminf(maxRadians, fmaxf(-maxRadians, radians));
  uint8_t magnitude = (uint8_t)min(127, (int)roundf(fabsf(clamped) * 127.0f / maxRadians));
  return (clamped < 0.0f ? 0x80 : 0x00) | magnitude;
}

__global__ void kernComputeFlockKeys(int N, const glm::vec3 *pos, const glm::vec3 *vel,
  int gridSideCount, glm::vec3 gridMinimum, float inverseCellWidth,
  uint32_t *keys, uint32_t *indices) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }

  glm::vec3 boidPos = pos[index];
  int x = imin(imax((int)floorf((boidPos.x - gridMinimum.x) * inverseCellWidth), 0), gridSideCount - 1);
  int y = imin(imax((int)floorf((boidPos.y - gridMinimum.y) * inverseCellWidth), 0), gridSideCount - 1);
  int z = imin(imax((int)floorf((boidPos.z - gridMinimum.z) * inverseCellWidth), 0), gridSideCount - 1);
  uint32_t cell = x + y * gridSideCount + z * gridSideCount * gridSideCount;

  keys[index] = (cell << flockDirectionKeyBits) | directionBin(boidForward(vel[index]));
  indices[index] = index;
}

__global__ void kernMarkFlockHeads(int N, const uint32_t *keys, int *headFlags) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    headFlags[index] = index == 0 || keys[index] != keys[index - 1];
  }
}

__global__ void kernBuildFlocksAndCompress(int N, const glm::vec3 *pos, const glm::vec3 *vel,
  const uint32_t *keys, const uint32_t *indices, const int *headFlags, const int *groupIds,
  int gridSideCount, glm::vec3 gridMinimum, float cellWidth,
  Boids::Flock *flocks, Boids::CompressedBoid *compressedBoids, uint32_t *flockCount) {
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  if (start >= N || !headFlags[start]) {
    return;
  }

  int end = start + 1;
  while (end < N && keys[end] == keys[start]) {
    end++;
  }

  glm::vec3 sumPosition(0.0f);
  glm::vec3 sumForward(0.0f);
  for (int sortedIndex = start; sortedIndex < end; sortedIndex++) {
    uint32_t boidIndex = indices[sortedIndex];
    sumPosition += pos[boidIndex];
    sumForward += boidForward(vel[boidIndex]);
  }

  uint32_t count = end - start;
  glm::vec3 center = sumPosition / (float)count;
  glm::vec3 direction = glm::length(sumForward) > 0.00001f
      ? glm::normalize(sumForward)
      : directionBinCenter(keys[start] & flockDirectionKeyMask);

  constexpr float pi = 3.14159265358979323846f;
  float maxRadians = flockMaxAxisDeviationDegrees * pi / 180.0f;
  float directionYaw = atan2f(direction.y, direction.x);
  float directionPitch = asinf(fminf(1.0f, fmaxf(-1.0f, direction.z)));
  float radius = 0.0f;
  bool valid = true;

  for (int sortedIndex = start; sortedIndex < end; sortedIndex++) {
    uint32_t boidIndex = indices[sortedIndex];
    glm::vec3 offset = pos[boidIndex] - center;
    radius = fmaxf(radius, glm::length(offset));
    glm::vec3 forward = boidForward(vel[boidIndex]);
    float yaw = atan2f(forward.y, forward.x);
    float pitch = asinf(fminf(1.0f, fmaxf(-1.0f, forward.z)));
    valid = valid && fabsf(signedAngleDelta(yaw, directionYaw)) <= maxRadians &&
      fabsf(pitch - directionPitch) <= maxRadians;
  }

  if (!valid || radius > flockMaxRadius) {
    uint32_t cell = keys[start] >> flockDirectionKeyBits;
    center = flockCellCenter(cell, gridSideCount, gridMinimum, cellWidth);
    direction = directionBinCenter(keys[start] & flockDirectionKeyMask);
    radius = 0.0f;
    for (int sortedIndex = start; sortedIndex < end; ++sortedIndex) {
      radius = fmaxf(radius, glm::length(pos[indices[sortedIndex]] - center));
    }
  }

  uint32_t flockId = groupIds[start];
  flocks[flockId] = { center, radius, direction, (uint32_t)start, count };
  atomicMax(flockCount, flockId + 1);

  directionYaw = atan2f(direction.y, direction.x);
  directionPitch = asinf(fminf(1.0f, fmaxf(-1.0f, direction.z)));
  float inverseRadius = radius > 1.0e-5f ? 1.0f / radius : 0.0f;

  for (int sortedIndex = start; sortedIndex < end; ++sortedIndex) {
    uint32_t boidIndex = indices[sortedIndex];
    glm::vec3 offset = pos[boidIndex] - center;
    float distance = glm::length(offset);
    float azimuth = distance > 1.0e-5f ? wrapAngle(atan2f(offset.y, offset.x)) : 0.0f;
    float polar = distance > 1.0e-5f ? acosf(fminf(1.0f, fmaxf(-1.0f, offset.z / distance))) : 0.0f;
    glm::vec3 forward = boidForward(vel[boidIndex]);
    float yaw = atan2f(forward.y, forward.x);
    float pitch = asinf(fminf(1.0f, fmaxf(-1.0f, forward.z)));

    Boids::CompressedBoid compressed;
    compressed.radius = (uint8_t)min(255, (int)roundf(distance * inverseRadius * 255.0f));
    compressed.azimuth = (uint8_t)min(255, (int)roundf(azimuth * 255.0f / (2.0f * pi)));
    compressed.polar = (uint8_t)min(255, (int)roundf(polar * 255.0f / pi));
    compressed.rotationX = packSignedRotation(signedAngleDelta(yaw, directionYaw));
    compressed.rotationY = packSignedRotation(pitch - directionPitch);
    compressed.rotationZ = packSignedRotation(0.0f);
    compressedBoids[sortedIndex] = compressed;
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

  flockCellWidth = 2.0f * flockMaxRadius / 1.73205080757f;
  flockInverseCellWidth = 1.0f / flockCellWidth;
  flockGridSideCount = (int)ceilf((2.0f * scene_scale) / flockCellWidth);
  flockGridMinimum = glm::vec3(-scene_scale);

  cudaMalloc((void**)&dev_particleArrayIndices, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_flockKeys, numObjects * sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc dev_flockKeys failed!");

  cudaMalloc((void**)&dev_flockIndices, numObjects * sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc dev_flockIndices failed!");

  cudaMalloc((void**)&dev_flockHeadFlags, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_flockHeadFlags failed!");

  cudaMalloc((void**)&dev_flockGroupIds, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_flockGroupIds failed!");

  cudaMalloc((void**)&dev_flocks, numObjects * sizeof(Boids::Flock));
  checkCUDAErrorWithLine("cudaMalloc dev_flocks failed!");

  cudaMalloc((void**)&dev_compressedBoids, numObjects * sizeof(Boids::CompressedBoid));
  checkCUDAErrorWithLine("cudaMalloc dev_compressedBoids failed!");

  cudaMalloc((void**)&dev_flockCount, sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc dev_flockCount failed!");

  // needed for use with thrust
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
  dev_thrust_vel1 = thrust::device_pointer_cast(dev_vel1);
  dev_thrust_vel2 = thrust::device_pointer_cast(dev_vel2);
  dev_thrust_pos = thrust::device_pointer_cast(dev_pos);
  dev_thrust_pos2 = thrust::device_pointer_cast(dev_pos2);
  dev_thrust_flockKeys = thrust::device_pointer_cast(dev_flockKeys);
  dev_thrust_flockIndices = thrust::device_pointer_cast(dev_flockIndices);
  dev_thrust_flockHeadFlags = thrust::device_pointer_cast(dev_flockHeadFlags);
  dev_thrust_flockGroupIds = thrust::device_pointer_cast(dev_flockGroupIds);

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

void Boids::computeFlocks() {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernComputeFlockKeys<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1,
    flockGridSideCount, flockGridMinimum, flockInverseCellWidth, dev_flockKeys, dev_flockIndices);
  checkCUDAErrorWithLine("kernComputeFlockKeys failed!");

  thrust::sort_by_key(dev_thrust_flockKeys, dev_thrust_flockKeys + numObjects,
    dev_thrust_flockIndices);

  kernMarkFlockHeads<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_flockKeys,
    dev_flockHeadFlags);
  checkCUDAErrorWithLine("kernMarkFlockHeads failed!");

  thrust::exclusive_scan(dev_thrust_flockHeadFlags,
    dev_thrust_flockHeadFlags + numObjects, dev_thrust_flockGroupIds);

  cudaMemset(dev_flockCount, 0, sizeof(uint32_t));
  kernBuildFlocksAndCompress<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1,
    dev_flockKeys, dev_flockIndices, dev_flockHeadFlags, dev_flockGroupIds,
    flockGridSideCount, flockGridMinimum, flockCellWidth, dev_flocks, dev_compressedBoids,
    dev_flockCount);
  checkCUDAErrorWithLine("kernBuildFlocksAndCompress failed!");

  cudaDeviceSynchronize();

  uint32_t flockCount = 0;
  cudaMemcpy(&flockCount, dev_flockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  const size_t compressedBytes = flockCount * sizeof(Boids::Flock) +
    numObjects * sizeof(Boids::CompressedBoid);
  printf("Flock compression complete: %u flocks, %zu bytes\n", flockCount, compressedBytes);
}

const Boids::Flock* Boids::getDeviceFlocks() {
  return dev_flocks;
}

const Boids::CompressedBoid* Boids::getDeviceCompressedBoids() {
  return dev_compressedBoids;
}

const uint32_t* Boids::getDeviceFlockCount() {
  return dev_flockCount;
}

__host__ __device__ uint32_t alignPacketChunk(uint32_t bytes) {
  return (bytes + 3u) & ~3u;
}

__device__ bool shouldFlockSurviveNaive(const Boids::Flock &flock,
  const Boids::CameraCullData &camera) {
  if (glm::length(flock.position - camera.position) >
      camera.maxDistance + cullDistanceTolerance + flock.radius) {
    return false;
  }

  glm::vec4 clipPosition = camera.projMat * glm::vec4(flock.position, 1.0f);
  if (clipPosition.w <= 0.0f) {
    return false;
  }

  float clipLimit = clipPosition.w * (1.0f + cullFrustumTolerance);
  return clipPosition.x >= -clipLimit && clipPosition.x <= clipLimit &&
    clipPosition.y >= -clipLimit && clipPosition.y <= clipLimit &&
    clipPosition.z >= -clipLimit && clipPosition.z <= clipLimit;
}

__global__ void kernCullFlocksByCameras(uint32_t flockCount,
  const Boids::Flock *flocks, const Boids::CompressedBoid *compressedBoids,
  const Boids::CameraCullData *cameras, uint32_t phoneCount,
  uint8_t **phonePacketBuffers, uint32_t *phonePacketSizes) {
  uint32_t flockIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (flockIndex >= flockCount) {
    return;
  }

  const Boids::Flock flock = flocks[flockIndex];
  const uint32_t payloadBytes = sizeof(Boids::NetworkFlock) +
    flock.numBoids * sizeof(Boids::CompressedBoid);
  const uint32_t chunkBytes = alignPacketChunk(payloadBytes);

  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    if (!shouldFlockSurviveNaive(flock, cameras[phoneIndex])) {
      continue;
    }

    const uint32_t packetOffset = atomicAdd(&phonePacketSizes[phoneIndex], chunkBytes);
    uint8_t *packet = phonePacketBuffers[phoneIndex] + packetOffset;
    Boids::NetworkFlock *packetFlock = reinterpret_cast<Boids::NetworkFlock *>(packet);
    *packetFlock = { flock.position, flock.radius, flock.direction, flock.numBoids };

    uint8_t *packetBoids = packet + sizeof(Boids::NetworkFlock);
    const uint8_t *sourceBoids = reinterpret_cast<const uint8_t *>(
      compressedBoids + flock.startIndex);
    for (uint32_t byte = 0; byte < flock.numBoids * sizeof(Boids::CompressedBoid); ++byte) {
      packetBoids[byte] = sourceBoids[byte];
    }
  }
}

void Boids::initializeNetworkPhones(uint32_t phoneCount) {
  for (uint8_t *packetBuffer : hostPhonePacketBuffers) {
    cudaFree(packetBuffer);
  }
  hostPhonePacketBuffers.clear();
  if (dev_networkCameras != nullptr) cudaFree(dev_networkCameras);
  if (dev_phonePacketBuffers != nullptr) cudaFree(dev_phonePacketBuffers);
  if (dev_phonePacketSizes != nullptr) cudaFree(dev_phonePacketSizes);

  networkPhoneCount = phoneCount;
  phonePacketCapacityBytes = numObjects * alignPacketChunk(
    sizeof(Boids::NetworkFlock) + sizeof(Boids::CompressedBoid));

  hostPhonePacketBuffers.resize(phoneCount);
  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    cudaMalloc((void **)&hostPhonePacketBuffers[phoneIndex], phonePacketCapacityBytes);
    checkCUDAErrorWithLine("cudaMalloc phone packet buffer failed!");
  }

  cudaMalloc((void **)&dev_networkCameras, phoneCount * sizeof(Boids::CameraCullData));
  checkCUDAErrorWithLine("cudaMalloc network cameras failed!");
  cudaMalloc((void **)&dev_phonePacketBuffers, phoneCount * sizeof(uint8_t *));
  checkCUDAErrorWithLine("cudaMalloc phone packet pointer array failed!");
  cudaMalloc((void **)&dev_phonePacketSizes, phoneCount * sizeof(uint32_t));
  checkCUDAErrorWithLine("cudaMalloc phone packet size array failed!");
  cudaMemcpy(dev_phonePacketBuffers, hostPhonePacketBuffers.data(), phoneCount * sizeof(uint8_t *),
    cudaMemcpyHostToDevice);
}

void Boids::createNetworkPackets(const CameraCullData *cameras, uint32_t phoneCount) {
  if (phoneCount != networkPhoneCount) {
    initializeNetworkPhones(phoneCount);
  }

  computeFlocks();

  uint32_t flockCount = 0;
  cudaMemcpy(&flockCount, dev_flockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(dev_networkCameras, cameras, phoneCount * sizeof(CameraCullData), cudaMemcpyHostToDevice);
  cudaMemset(dev_phonePacketSizes, 0, phoneCount * sizeof(uint32_t));

  dim3 blocksPerGrid((flockCount + blockSize - 1) / blockSize);
  kernCullFlocksByCameras<<<blocksPerGrid, blockSize>>>(flockCount, dev_flocks,
    dev_compressedBoids, dev_networkCameras, phoneCount, dev_phonePacketBuffers,
    dev_phonePacketSizes);
  checkCUDAErrorWithLine("kernCullFlocksByCameras failed!");
  cudaDeviceSynchronize();

  std::vector<uint32_t> phonePacketSizes(phoneCount);
  cudaMemcpy(phonePacketSizes.data(), dev_phonePacketSizes, phoneCount * sizeof(uint32_t),
    cudaMemcpyDeviceToHost);
  size_t totalPacketBytes = 0;
  for (uint32_t packetBytes : phonePacketSizes) {
    totalPacketBytes += packetBytes;
  }
  printf("Network packet compression complete: %zu bytes across %u phones\n",
    totalPacketBytes, phoneCount);
}

uint8_t* const* Boids::getDevicePhonePackets() {
  return dev_phonePacketBuffers;
}

const uint32_t* Boids::getDevicePhonePacketSizes() {
  return dev_phonePacketSizes;
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

    glm::vec3 *tempPos = dev_pos;
    dev_pos = dev_pos2;
    dev_pos2 = tempPos;

    thrust::device_ptr<glm::vec3> tempThrustPos = dev_thrust_pos;
    dev_thrust_pos = dev_thrust_pos2;
    dev_thrust_pos2 = tempThrustPos;

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
  cudaFree(dev_pos);
  cudaFree(dev_pos2);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_flockKeys);
  cudaFree(dev_flockIndices);
  cudaFree(dev_flockHeadFlags);
  cudaFree(dev_flockGroupIds);
  cudaFree(dev_flocks);
  cudaFree(dev_compressedBoids);
  cudaFree(dev_flockCount);
  for (uint8_t *packetBuffer : hostPhonePacketBuffers) {
    cudaFree(packetBuffer);
  }
  cudaFree(dev_networkCameras);
  cudaFree(dev_phonePacketBuffers);
  cudaFree(dev_phonePacketSizes);

}
