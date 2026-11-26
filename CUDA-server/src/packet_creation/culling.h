#pragma once

#include "../kernel.h"

// Also used by MIXED-mode new-boid encoding in network.cu. Keep inline so
// callers in different CUDA translation units do not require device linking.
inline __device__ bool isInsideCameraFrustum(const glm::vec3 &position,
    const Boids::CameraCullData &camera, float tolerance) {
  const glm::vec4 clipPosition = camera.projMat * glm::vec4(position, 1.0f);
  if (clipPosition.w <= 0.0f) return false;
  const float clipLimit = clipPosition.w * (1.0f + tolerance);
  return clipPosition.x >= -clipLimit && clipPosition.x <= clipLimit &&
    clipPosition.y >= -clipLimit && clipPosition.y <= clipLimit &&
    clipPosition.z >= -clipLimit && clipPosition.z <= clipLimit;
}

// Uncalled baseline: blockIdx.x = cell, blockIdx.y = phone batch,
// threadIdx.x = phone within the batch. Requires tile-size shared storage.
__global__ void gpicNodeProcessingNaive(Boids::GridSnapshot grid, uint32_t phoneCount,
    const uint32_t *cellCameraMasks, const glm::vec3 *positions,
    const Boids::CameraCullData *cameras, int32_t *visibleCounts,
    uint32_t **selectedIndicesByPhone);

__global__ void gpicNodeProcessing(Boids::GridSnapshot grid, uint32_t phoneCount,
    const uint32_t *cellCameraMasks, const glm::vec3 *positions,
    const Boids::CameraCullData *cameras, int32_t *visibleCounts,
    uint32_t **selectedIndicesByPhone);

__global__ void gpicCullQueryRegisteringGrid(Boids::GridSnapshot grid, uint32_t phoneCount,
    const Boids::CameraCullData *cameras, uint32_t *cellCameraMasks);

__global__ void kernCullBoidsByCameras(uint32_t boidCount,
  const glm::vec3 *positions, const Boids::CameraCullData *cameras,
  uint32_t phoneCount, int32_t *visibleCounts, uint32_t **selectedIndicesByPhone);
