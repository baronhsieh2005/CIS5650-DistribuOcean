#define GLM_FORCE_CUDA

#include <cuda.h>
#include "culling.h"
#include <cmath>

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif

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
#endif
}

__global__ void kernCullBoidsByCameras(uint32_t boidCount,
  const glm::vec3 *positions, const Boids::CameraCullData *cameras,
  uint32_t phoneCount, int32_t *visibleCounts, uint32_t **selectedIndicesByPhone) {
  const uint32_t boidIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (boidIndex >= boidCount) {
    return;
  }

  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    float distanceToCamera;
    const bool survives = shouldBoidSurviveNaive(boidIndex, positions,
      cameras[phoneIndex], &distanceToCamera);
    if (survives) {
      const int32_t rank = atomicAdd(visibleCounts + phoneIndex, 1);
      if (rank < MAX_BOIDS_PER_PHONE) {
        selectedIndicesByPhone[phoneIndex][rank] = boidIndex;
      }
    }
  }
}

__device__ bool cellIntersectsFrustum(const glm::vec3 &center, float halfWidth,
    const Boids::CameraCullData &camera, float tolerance) {
  const glm::mat4 &m = camera.projMat;
  const glm::vec4 clip = m * glm::vec4(center, 1.0f);
  const glm::vec3 rowW(m[0][3], m[1][3], m[2][3]);
  const float extentW = halfWidth * (fabsf(rowW.x) + fabsf(rowW.y) + fabsf(rowW.z));
  const float epsilon = 0.0001f * (1.0f + fabsf(clip.w) + extentW);
  if (clip.w + extentW < -epsilon) return false;
  const float limit = 1.0f + tolerance;
  for (int axis = 0; axis < 3; ++axis) {
    const glm::vec3 rowAxis(m[0][axis], m[1][axis], m[2][axis]);
    const glm::vec3 plus = limit * rowW + rowAxis;
    const glm::vec3 minus = limit * rowW - rowAxis;
    const float plusExtent = halfWidth * (fabsf(plus.x) + fabsf(plus.y) + fabsf(plus.z));
    const float minusExtent = halfWidth * (fabsf(minus.x) + fabsf(minus.y) + fabsf(minus.z));
    if (limit * clip.w + clip[axis] + plusExtent < -epsilon ||
        limit * clip.w - clip[axis] + minusExtent < -epsilon) return false;
  }
  return true;
}

__device__ bool cellIsVisible(uint32_t cellIdx, const Boids::GridSnapshot &grid,
    const Boids::CameraCullData &camera) {
  const uint32_t x = cellIdx % grid.sideCount;
  const uint32_t y = (cellIdx / grid.sideCount) % grid.sideCount;
  const uint32_t z = cellIdx / (grid.sideCount * grid.sideCount);
  const float halfWidth = 0.5f * grid.cellWidth;
  const glm::vec3 center = grid.minimum + grid.cellWidth *
    (glm::vec3(x, y, z) + glm::vec3(0.5f));
  const glm::vec3 d = glm::max(glm::abs(center - camera.position) -
    glm::vec3(halfWidth), glm::vec3(0.0f));
  const float distanceSquared = glm::dot(d, d);
#if CULL_TYPE == MIXED
  const float innerRadius = Boids::cullDistance * mixedCullSphereFraction;
  if (distanceSquared <= innerRadius * innerRadius * (1.0f + 0.000001f)) return true;
#endif
  const float radius = Boids::cullDistance + cullDistanceTolerance;
  if (distanceSquared > radius * radius * (1.0f + 0.000001f)) return false;
#if CULL_TYPE == DIST_SPHERE
  return true;
#elif CULL_TYPE == DIST_VIEW
  return cellIntersectsFrustum(center, halfWidth, camera, 0.0f);
#elif CULL_TYPE == MIXED
  return cellIntersectsFrustum(center, halfWidth, camera, cullFrustumTolerance);
#endif
}

__global__ void gpicCullQueryRegisteringGrid(Boids::GridSnapshot grid, uint32_t phoneCount,
    const Boids::CameraCullData *cameras, uint32_t *cellCameraMasks) {
  const uint32_t cellIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t cameraIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (cellIdx >= grid.cellCount || cameraIdx >= phoneCount || grid.cellStarts[cellIdx] < 0) {
    return;
  }

  const uint32_t word = cameraIdx / 32;
  const uint32_t bit = cameraIdx % 32;
  const int intsPerCell = (phoneCount + 31) / 32;

  if (cellIsVisible(cellIdx, grid, cameras[cameraIdx])) {
    atomicOr(&cellCameraMasks[static_cast<size_t>(cellIdx) * intsPerCell + word], 1u << bit);
  }
}

__global__ void gpicNodeProcessing(Boids::GridSnapshot grid, uint32_t phoneCount,
    const uint32_t *cellCameraMasks, const glm::vec3 *positions,
    const Boids::CameraCullData *cameras, int32_t *visibleCounts,
    uint32_t **selectedIndicesByPhone) {
  const uint32_t cellIdx = blockIdx.x;
  if (cellIdx >= grid.cellCount) return;

  const int start = grid.cellStarts[cellIdx];
  if (start < 0) return;

  const uint32_t warpIdx = threadIdx.x / 32;
  const uint32_t laneIdx = threadIdx.x % 32;
  const uint32_t warpsPerBlock = blockDim.x / 32;
  const uint32_t intsPerCell = (phoneCount + 31) / 32;

  extern __shared__ glm::vec3 boidPositions[];
  uint32_t *registeredCameras = reinterpret_cast<uint32_t *>(boidPositions + GPIC_SMEM_TILE_SIZE);
  __shared__ uint32_t registeredCount;
  if (threadIdx.x == 0) {
    uint32_t count = 0;
    for (uint32_t word = 0; word < intsPerCell; ++word) {
      uint32_t mask = cellCameraMasks[static_cast<size_t>(cellIdx) * intsPerCell + word];
      while (mask != 0) {
        const uint32_t bit = __ffs(mask) - 1;
        const uint32_t phoneIdx = word * 32 + bit;
        if (phoneIdx < phoneCount) registeredCameras[count++] = phoneIdx;
        mask &= mask - 1;
      }
    }
    registeredCount = count;
  }
  __syncthreads();
  if (registeredCount == 0) return;

  const int end = grid.cellEnds[cellIdx];
  for (int tileStart = start; tileStart <= end; tileStart += GPIC_SMEM_TILE_SIZE) {
    const int tileCount = imin(GPIC_SMEM_TILE_SIZE, end - tileStart + 1);
    for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
      boidPositions[i] = positions[tileStart + i];
    }
    __syncthreads();

    for (uint32_t query = warpIdx; query < registeredCount; query += warpsPerBlock) {
      const uint32_t phoneIdx = registeredCameras[query];
      const Boids::CameraCullData camera = cameras[phoneIdx];
      for (int first = 0; first < tileCount; first += 32) {
        const int i = first + laneIdx;
        float distanceToCamera;
        const bool visible = i < tileCount &&
          shouldBoidSurviveNaive(i, boidPositions, camera, &distanceToCamera);
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, visible);
        if (mask == 0) continue;
        int32_t firstRank = 0;
        if (laneIdx == 0) {
          firstRank = atomicAdd(visibleCounts + phoneIdx, __popc(mask));
        }
        firstRank = __shfl_sync(0xFFFFFFFFu, firstRank, 0);
        if (visible) {
          const int32_t rank = firstRank + __popc(mask & ((1u << laneIdx) - 1u));
          if (rank < MAX_BOIDS_PER_PHONE) {
            selectedIndicesByPhone[phoneIdx][rank] = tileStart + i;
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void gpicNodeProcessingNaive(Boids::GridSnapshot grid, uint32_t phoneCount,
    const uint32_t *cellCameraMasks, const glm::vec3 *positions,
    const Boids::CameraCullData *cameras, int32_t *visibleCounts,
    uint32_t **selectedIndicesByPhone) {
  const uint32_t cellIdx = blockIdx.x;
  const uint32_t phoneIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (cellIdx >= grid.cellCount) return;
  const int start = grid.cellStarts[cellIdx];
  if (start < 0) return;

  const uint32_t intsPerCell = (phoneCount + 31) / 32;
  bool activeCamera = false;
  if (phoneIdx < phoneCount) {
    activeCamera = (cellCameraMasks[static_cast<size_t>(cellIdx) * intsPerCell +
      phoneIdx / 32] & (1u << (phoneIdx % 32))) != 0;
  }
  if (!__syncthreads_or(activeCamera)) return;

  const int end = grid.cellEnds[cellIdx];
  Boids::CameraCullData camera;
  if (activeCamera) camera = cameras[phoneIdx];
  extern __shared__ glm::vec3 boidPositions[];

  for (int tileStart = start; tileStart <= end; tileStart += GPIC_SMEM_TILE_SIZE) {
    const int tileCount = imin(GPIC_SMEM_TILE_SIZE, end - tileStart + 1);
    for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
      boidPositions[i] = positions[tileStart + i];
    }
    __syncthreads();
    if (activeCamera) {
      for (int i = 0; i < tileCount; ++i) {
        float distanceToCamera;
        if (shouldBoidSurviveNaive(i, boidPositions, camera, &distanceToCamera)) {
          const int32_t rank = atomicAdd(visibleCounts + phoneIdx, 1);
          if (rank < MAX_BOIDS_PER_PHONE) {
            selectedIndicesByPhone[phoneIdx][rank] = tileStart + i;
          }
        }
      }
    }
    __syncthreads();
  }
}


__device__ bool glintLeafIsVisible(const Boids::GlintLeafSnapshot &leaf,
    const Boids::CameraCullData &camera) {
  const glm::vec3 center = (leaf.minimum + leaf.maximum) * 0.5f;
  const glm::vec3 halfSize = (leaf.maximum - leaf.minimum) * 0.5f;
  const glm::vec3 d = glm::max(glm::abs(center - camera.position) - halfSize,
    glm::vec3(0.0f));
  const float distanceSquared = glm::dot(d, d);
#if CULL_TYPE == MIXED
  const float innerRadius = Boids::cullDistance * mixedCullSphereFraction;
  if (distanceSquared <= innerRadius * innerRadius * (1.0f + 0.000001f)) return true;
#endif
  const float radius = Boids::cullDistance + cullDistanceTolerance;
  if (distanceSquared > radius * radius * (1.0f + 0.000001f)) return false;
#if CULL_TYPE == DIST_SPHERE
  return true;
#else
  const float halfWidth = fmaxf(halfSize.x, fmaxf(halfSize.y, halfSize.z));
#if CULL_TYPE == DIST_VIEW
  return cellIntersectsFrustum(center, halfWidth, camera, 0.0f);
#elif CULL_TYPE == MIXED
  return cellIntersectsFrustum(center, halfWidth, camera, cullFrustumTolerance);
#endif
#endif
}

__global__ void gpicCullQueryRegisteringGridGlint(Boids::GlintSnapshot glint,
    uint32_t phoneCount, const Boids::CameraCullData *cameras, uint32_t *cellCameraMasks) {
  const uint32_t cellIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t cameraIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (cellIdx >= glint.leafCount || cameraIdx >= phoneCount ||
      glint.leaves[cellIdx].numBoids == 0) return;
  const uint32_t intsPerCell = (phoneCount + 31) / 32;
  if (glintLeafIsVisible(glint.leaves[cellIdx], cameras[cameraIdx])) {
    atomicOr(&cellCameraMasks[static_cast<size_t>(cellIdx) * intsPerCell + cameraIdx / 32],
      1u << (cameraIdx % 32));
  }
}

__global__ void gpicNodeProcessingGlint(Boids::GlintSnapshot glint, uint32_t phoneCount,
    const uint32_t *cellCameraMasks, const glm::vec3 *positions,
    const Boids::CameraCullData *cameras, int32_t *visibleCounts,
    uint32_t **selectedIndicesByPhone) {
  const uint32_t cellIdx = blockIdx.x;
  if (cellIdx >= glint.leafCount) return;

  const auto leaf = glint.leaves[cellIdx];
  if (leaf.numBoids == 0) return;
  const int start = leaf.startIndex;

  const uint32_t warpIdx = threadIdx.x / 32;
  const uint32_t laneIdx = threadIdx.x % 32;
  const uint32_t warpsPerBlock = blockDim.x / 32;
  const uint32_t intsPerCell = (phoneCount + 31) / 32;

  extern __shared__ glm::vec3 boidPositions[];
  uint32_t *registeredCameras = reinterpret_cast<uint32_t *>(boidPositions + GPIC_SMEM_TILE_SIZE);
  __shared__ uint32_t registeredCount;
  if (threadIdx.x == 0) {
    uint32_t count = 0;
    for (uint32_t word = 0; word < intsPerCell; ++word) {
      uint32_t mask = cellCameraMasks[static_cast<size_t>(cellIdx) * intsPerCell + word];
      while (mask != 0) {
        const uint32_t bit = __ffs(mask) - 1;
        const uint32_t phoneIdx = word * 32 + bit;
        if (phoneIdx < phoneCount) registeredCameras[count++] = phoneIdx;
        mask &= mask - 1;
      }
    }
    registeredCount = count;
  }
  __syncthreads();
  if (registeredCount == 0) return;

  const int end = start + leaf.numBoids - 1;
  for (int tileStart = start; tileStart <= end; tileStart += GPIC_SMEM_TILE_SIZE) {
    const int tileCount = imin(GPIC_SMEM_TILE_SIZE, end - tileStart + 1);
    for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
      boidPositions[i] = positions[glint.boidIndices[tileStart + i]];
    }
    __syncthreads();

    for (uint32_t query = warpIdx; query < registeredCount; query += warpsPerBlock) {
      const uint32_t phoneIdx = registeredCameras[query];
      const Boids::CameraCullData camera = cameras[phoneIdx];
      for (int first = 0; first < tileCount; first += 32) {
        const int i = first + laneIdx;
        float distanceToCamera;
        const bool visible = i < tileCount &&
          shouldBoidSurviveNaive(i, boidPositions, camera, &distanceToCamera);
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, visible);
        if (mask == 0) continue;
        int32_t firstRank = 0;
        if (laneIdx == 0) {
          firstRank = atomicAdd(visibleCounts + phoneIdx, __popc(mask));
        }
        firstRank = __shfl_sync(0xFFFFFFFFu, firstRank, 0);
        if (visible) {
          const int32_t rank = firstRank + __popc(mask & ((1u << laneIdx) - 1u));
          if (rank < MAX_BOIDS_PER_PHONE) {
            selectedIndicesByPhone[phoneIdx][rank] = glint.boidIndices[tileStart + i];
          }
        }
      }
    }
    __syncthreads();
  }
}
