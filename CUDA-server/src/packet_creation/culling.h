#pragma once

#include "../kernel.h"

inline __device__ bool isInsideCameraFrustum(const glm::vec3 &position,
    const Boids::CameraCullData &camera, float tolerance) {
  const glm::vec4 clipPosition = camera.projMat * glm::vec4(position, 1.0f);
  if (clipPosition.w <= 0.0f) return false;
  const float clipLimit = clipPosition.w * (1.0f + tolerance);
  return clipPosition.x >= -clipLimit && clipPosition.x <= clipLimit &&
    clipPosition.y >= -clipLimit && clipPosition.y <= clipLimit &&
    clipPosition.z >= -clipLimit && clipPosition.z <= clipLimit;
}

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

__global__ void gpicCullQueryRegisteringGridGlint(Boids::GlintSnapshot glint,
    uint32_t phoneCount, const Boids::CameraCullData *cameras, uint32_t *cellCameraMasks);

__global__ void gpicNodeProcessingGlint(Boids::GlintSnapshot glint, uint32_t phoneCount,
    const uint32_t *cellCameraMasks, const glm::vec3 *positions,
    const Boids::CameraCullData *cameras, int32_t *visibleCounts,
    uint32_t **selectedIndicesByPhone);


__global__ void kernCullBoidsByCameras(uint32_t boidCount,
  const glm::vec3 *positions, const Boids::CameraCullData *cameras,
  uint32_t phoneCount, int32_t *visibleCounts, uint32_t **selectedIndicesByPhone);
