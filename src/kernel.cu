#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>

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

#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

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

  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

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

  float c_scale = -1.0f / s_scale;

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

}
