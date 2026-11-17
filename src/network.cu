#define GLM_FORCE_CUDA

#include <cuda.h>
#include <math_constants.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

#include <glm/glm.hpp>

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define cullDistanceTolerance 36.0f
#define cullFrustumTolerance 0.12f
#define MAX_BOIDS_PER_PHONE 20000
#define DIST_VIEW 0
#define DIST_SPHERE 1
#define MIXED 2
#define CULL_TYPE DIST_VIEW
#define mixedCullSphereFraction 0.5f
#define networkPositionDeltaScale (1.0f / 1024.0f)
#define networkPositionDepthMaxDistance (Boids::cullDistance + cullDistanceTolerance)
#define networkUpdateFreeComponent 0xFFFFu

cudaStream_t network_stream = nullptr;

void checkCUDAError(const char *msg, int line);
extern int numObjects;
extern uint32_t *dev_boidIds_snapshot;
extern glm::vec3 *dev_pos_snapshot;
extern cudaEvent_t simulationSnapshotReadyEvent;

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

std::vector<std::unique_ptr<std::binary_semaphore>> phone_packet_ready_semaphore;
std::vector<std::thread> phone_packet_send_threads;
bool stop_phone_packet_send_loop = false;
Boids::UpdateBoidData **host_phone_update_packets = nullptr;
Boids::NewBoidData **host_phone_new_packets = nullptr;
uint32_t *host_phone_update_payload_lengths = nullptr;
uint32_t *host_phone_new_payload_lengths = nullptr;
cudaEvent_t phone_packet_sizes_ready_event = nullptr;
std::vector<cudaEvent_t> phone_packet_copy_ready_events;

std::vector<Boids::CameraCullData> queuedNetworkCameras;
std::unique_ptr<bool[]> queuedNetworkCameraChanged;
uint32_t queuedNetworkPhoneCount = 0;

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
  if (network_stream == nullptr) {
    cudaStreamCreate(&network_stream);
  }
  if (!phone_packet_send_threads.empty()) {
    stop_phone_packet_send_loop = true;
    for (const auto &semaphore : phone_packet_ready_semaphore) {
      semaphore->release();
    }
    for (std::thread &thread : phone_packet_send_threads) {
      thread.join();
    }
    phone_packet_send_threads.clear();
    phone_packet_ready_semaphore.clear();
  }
  for (cudaEvent_t event : phone_packet_copy_ready_events) {
    cudaEventDestroy(event);
  }
  phone_packet_copy_ready_events.clear();
  if (phone_packet_sizes_ready_event != nullptr) {
    cudaEventDestroy(phone_packet_sizes_ready_event);
    phone_packet_sizes_ready_event = nullptr;
  }
  if (host_phone_update_packets != nullptr) {
    for (uint32_t phoneIndex = 0; phoneIndex < networkPhoneCount; ++phoneIndex) {
      cudaFreeHost(host_phone_update_packets[phoneIndex]);
      cudaFreeHost(host_phone_new_packets[phoneIndex]);
    }
    cudaFreeHost(host_phone_update_packets);
    cudaFreeHost(host_phone_new_packets);
    cudaFreeHost(host_phone_update_payload_lengths);
    cudaFreeHost(host_phone_new_payload_lengths);
    host_phone_update_packets = nullptr;
    host_phone_new_packets = nullptr;
    host_phone_update_payload_lengths = nullptr;
    host_phone_new_payload_lengths = nullptr;
  }
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

  stop_phone_packet_send_loop = false;
  phone_packet_ready_semaphore.reserve(phoneCount);
  phone_packet_send_threads.reserve(phoneCount);
  cudaMallocHost((void **)&host_phone_update_packets,
    phoneCount * sizeof(Boids::UpdateBoidData *));
  cudaMallocHost((void **)&host_phone_new_packets,
    phoneCount * sizeof(Boids::NewBoidData *));
  cudaMallocHost((void **)&host_phone_update_payload_lengths, phoneCount * sizeof(uint32_t));
  cudaMallocHost((void **)&host_phone_new_payload_lengths, phoneCount * sizeof(uint32_t));
  cudaEventCreateWithFlags(&phone_packet_sizes_ready_event, cudaEventDisableTiming);
  phone_packet_copy_ready_events.resize(phoneCount);
  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    cudaMallocHost((void **)&host_phone_update_packets[phoneIndex],
      MAX_BOIDS_PER_PHONE * sizeof(Boids::UpdateBoidData));
    cudaMallocHost((void **)&host_phone_new_packets[phoneIndex],
      MAX_BOIDS_PER_PHONE * sizeof(Boids::NewBoidData));
    host_phone_update_payload_lengths[phoneIndex] = 0;
    host_phone_new_payload_lengths[phoneIndex] = 0;
    phone_packet_ready_semaphore.emplace_back(std::make_unique<std::binary_semaphore>(0));
    cudaEventCreateWithFlags(&phone_packet_copy_ready_events[phoneIndex], cudaEventDisableTiming);
    phone_packet_send_threads.emplace_back(Boids::phoneNetworkSendLoop, phoneIndex);
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
  cudaMemcpyAsync(dev_networkCameras, cameras, phoneCount * sizeof(Boids::CameraCullData),
    cudaMemcpyHostToDevice, network_stream);
  checkCUDAErrorWithLine("cudaMemcpy initial network cameras failed!");
  cudaMalloc((void **)&dev_phoneDistanceBuffers, phoneCount * sizeof(float *));
  checkCUDAErrorWithLine("cudaMalloc phone distance pointer array failed!");
  cudaMemcpyAsync(dev_phoneDistanceBuffers, hostPhoneDistanceBuffers.data(),
    phoneCount * sizeof(float *), cudaMemcpyHostToDevice, network_stream);
  checkCUDAErrorWithLine("cudaMemcpy phone distance pointer array failed!");
  cudaMalloc((void **)&dev_phoneSortedBoidIndices, phoneCount * sizeof(uint32_t *));
  checkCUDAErrorWithLine("cudaMalloc phone sorted-index pointer array failed!");
  cudaMemcpyAsync(dev_phoneSortedBoidIndices, hostPhoneSortedBoidIndices.data(),
    phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice, network_stream);
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
    cudaMemsetAsync(hostBoidSlotInPacketByPhone[phoneIndex], 0xFF, numObjects * sizeof(int), network_stream);
    cudaMalloc((void **)&hostClientPositionsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(glm::vec3));
    cudaMemsetAsync(hostClientPositionsByPhone[phoneIndex], 0, MAX_BOIDS_PER_PHONE * sizeof(glm::vec3), network_stream);
    cudaMalloc((void **)&hostBoidUpdatePackets[phoneIndex],
      MAX_BOIDS_PER_PHONE * sizeof(Boids::UpdateBoidData));
    cudaMalloc((void **)&hostBoidNewPackets[phoneIndex],
      MAX_BOIDS_PER_PHONE * sizeof(Boids::NewBoidData));
    cudaMalloc((void **)&hostPacketEpochsByPhone[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMemsetAsync(hostPacketEpochsByPhone[phoneIndex], 0, MAX_BOIDS_PER_PHONE * sizeof(uint32_t), network_stream);
    cudaMalloc((void **)&hostBoidIdBySlotInPacket[phoneIndex], MAX_BOIDS_PER_PHONE * sizeof(uint32_t));
    cudaMemsetAsync(hostBoidIdBySlotInPacket[phoneIndex], 0xFF, MAX_BOIDS_PER_PHONE * sizeof(uint32_t), network_stream);
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
  cudaMemcpyAsync(dev_boidSlotInPacketByPhone, hostBoidSlotInPacketByPhone.data(), phoneCount * sizeof(int *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_clientPositionsByPhone, hostClientPositionsByPhone.data(), phoneCount * sizeof(glm::vec3 *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_boidUpdatePackets, hostBoidUpdatePackets.data(), phoneCount * sizeof(Boids::UpdateBoidData *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_boidNewPackets, hostBoidNewPackets.data(), phoneCount * sizeof(Boids::NewBoidData *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_packetEpochsByPhone, hostPacketEpochsByPhone.data(), phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_boidIdBySlotInPacket, hostBoidIdBySlotInPacket.data(), phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_freeSlotFlagsByPhone, hostFreeSlotFlagsByPhone.data(), phoneCount * sizeof(uint8_t *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_newBoidFlagsByPhone, hostNewBoidFlagsByPhone.data(), phoneCount * sizeof(uint8_t *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_freeSlotsByPhone, hostFreeSlotsByPhone.data(), phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice, network_stream);
  cudaMemcpyAsync(dev_newRanksByPhone, hostNewRanksByPhone.data(), phoneCount * sizeof(uint32_t *), cudaMemcpyHostToDevice, network_stream);
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
  return static_cast<int16_t>(imax(-0x8000, imin(0x7FFF, quantized)));
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
  return static_cast<uint8_t>(roundf(fminf(1.0f, fmaxf(0.0f, value)) * 0xFF));
}

__device__ uint16_t quantizeNetworkDepth(float distance) {
  return static_cast<uint16_t>(roundf(fminf(1.0f,
    fmaxf(0.0f, distance / networkPositionDepthMaxDistance)) * 0xFFFF));
}

__device__ float dequantizeNetworkDepth(uint16_t depth) {
  return static_cast<float>(depth) * networkPositionDepthMaxDistance / 0xFFFF;
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

  cudaStreamWaitEvent(network_stream, simulationSnapshotReadyEvent, 0);

  dim3 blocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernCullBoidsByCameras<<<blocksPerGrid, blockSize, 0, network_stream>>>(numObjects, dev_pos_snapshot,
    dev_networkCameras, networkPhoneCount, dev_phoneDistanceBuffers, dev_phoneSortedBoidIndices);
  checkCUDAErrorWithLine("kernCullBoidsByCameras failed!");

  for (uint32_t phoneIndex = 0; phoneIndex < networkPhoneCount; ++phoneIndex) {
    thrust::device_ptr<float> distanceBegin =
      thrust::device_pointer_cast(hostPhoneDistanceBuffers[phoneIndex]);
    thrust::device_ptr<uint32_t> boidIndexBegin =
      thrust::device_pointer_cast(hostPhoneSortedBoidIndices[phoneIndex]);
    thrust::sort_by_key(thrust::cuda::par_nosync.on(network_stream),
      distanceBegin, distanceBegin + numObjects, boidIndexBegin);
  }

  dim3 cullCountBlockSize(32, 4);
  dim3 cullCountBlocksPerGrid((numObjects + cullCountBlockSize.x - 1) / cullCountBlockSize.x,
    (networkPhoneCount + cullCountBlockSize.y - 1) / cullCountBlockSize.y);
  kernGetFirstInf<<<cullCountBlocksPerGrid, cullCountBlockSize, 0, network_stream>>>(numObjects, networkPhoneCount,
    dev_phoneDistanceBuffers, dev_phoneVisibleCounts);
  checkCUDAErrorWithLine("kernGetFirstInf failed!");

  const uint32_t packetWorkItems = networkPhoneCount * MAX_BOIDS_PER_PHONE;
  dim3 packetBlocksPerGrid((packetWorkItems + blockSize - 1) / blockSize);
  const uint32_t currentEpoch = networkEpoch++;
  kernBuildNetworkPackets<<<packetBlocksPerGrid, blockSize, 0, network_stream>>>(networkPhoneCount,
    currentEpoch, dev_pos_snapshot, dev_boidIds_snapshot, dev_phoneSortedBoidIndices, dev_phoneVisibleCounts,
    dev_boidSlotInPacketByPhone, dev_clientPositionsByPhone, dev_boidUpdatePackets,
    dev_boidNewPackets, dev_packetEpochsByPhone, dev_newBoidFlagsByPhone);
  checkCUDAErrorWithLine("kernBuildNetworkPackets failed!");

  kernClearFreePacketSlots<<<packetBlocksPerGrid, blockSize, 0, network_stream>>>(networkPhoneCount,
    currentEpoch, dev_boidSlotInPacketByPhone, dev_boidIdBySlotInPacket,
    dev_packetEpochsByPhone, dev_boidUpdatePackets, dev_freeSlotFlagsByPhone);
  checkCUDAErrorWithLine("kernClearFreePacketSlots failed!");

  thrust::counting_iterator<uint32_t> slotIndices(0);
  for (uint32_t phone_idx = 0; phone_idx < networkPhoneCount; ++phone_idx) {
    size_t tempStorageBytes = packetCompactionTempStorageBytes;
    cub::DeviceSelect::Flagged(dev_packetCompactionTempStorage, tempStorageBytes,
      slotIndices, hostFreeSlotFlagsByPhone[phone_idx], hostFreeSlotsByPhone[phone_idx],
      dev_phoneFreeCounts + phone_idx, MAX_BOIDS_PER_PHONE, network_stream);
    cub::DeviceSelect::Flagged(dev_packetCompactionTempStorage, tempStorageBytes,
      slotIndices, hostNewBoidFlagsByPhone[phone_idx], hostNewRanksByPhone[phone_idx],
      dev_phoneNewCounts + phone_idx, MAX_BOIDS_PER_PHONE, network_stream);
  }

#if DEBUG_PACKET_LENGTHS
  std::vector<uint32_t> newBoidCounts(networkPhoneCount);
  std::vector<int32_t> visibleLastIndices(networkPhoneCount);
  cudaMemcpyAsync(newBoidCounts.data(), dev_phoneNewCounts,
    networkPhoneCount * sizeof(uint32_t), cudaMemcpyDeviceToHost, network_stream);
  checkCUDAErrorWithLine("cudaMemcpy compacted new-boid counts failed!");
  cudaMemcpyAsync(visibleLastIndices.data(), dev_phoneVisibleCounts,
    networkPhoneCount * sizeof(int32_t), cudaMemcpyDeviceToHost, network_stream);
  checkCUDAErrorWithLine("cudaMemcpy visible boid counts failed!");
  cudaStreamSynchronize(network_stream);
  for (uint32_t phone_idx = 0; phone_idx < networkPhoneCount; ++phone_idx) {
    const uint32_t selectedCount = visibleLastIndices[phone_idx] < 0 ? 0u :
      static_cast<uint32_t>(visibleLastIndices[phone_idx]) + 1u;
    const uint32_t updateCount = selectedCount - newBoidCounts[phone_idx];
    printf("Phone %u compacted packet data: %u boid updates, %u new boids\n", phone_idx,
      updateCount, newBoidCounts[phone_idx]);
  }
#endif

  kernAssignNewPacketSlots<<<packetBlocksPerGrid, blockSize, 0, network_stream>>>(networkPhoneCount,
    currentEpoch, dev_pos_snapshot, dev_boidIds_snapshot, dev_networkCameras, dev_phoneSortedBoidIndices,
    dev_newRanksByPhone, dev_phoneNewCounts, dev_freeSlotsByPhone, dev_phoneFreeCounts,
    dev_boidSlotInPacketByPhone, dev_boidIdBySlotInPacket, dev_packetEpochsByPhone,
    dev_clientPositionsByPhone, dev_boidNewPackets);
  checkCUDAErrorWithLine("kernAssignNewPacketSlots failed!");

  kernDeriveNetworkPayloadLengths<<<networkPhoneCount, blockSize, 0, network_stream>>>(networkPhoneCount,
    currentEpoch, dev_packetEpochsByPhone, dev_phoneNewCounts,
    dev_phoneUpdatePayloadLengths, dev_phoneNewPayloadLengths);
  checkCUDAErrorWithLine("kernDeriveNetworkPayloadLengths failed!");

  cudaStreamSynchronize(network_stream);
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
    cudaMemcpyAsync(dev_networkCameras + phone_idx, cameras + phone_idx,
      sizeof(CameraCullData), cudaMemcpyHostToDevice, network_stream);
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


void releaseNetworkBuffers() {
  stop_phone_packet_send_loop = true;
  for (const auto &semaphore : phone_packet_ready_semaphore) {
    semaphore->release();
  }
  for (std::thread &thread : phone_packet_send_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  phone_packet_send_threads.clear();
  phone_packet_ready_semaphore.clear();
  for (uint32_t phoneIndex = 0; phoneIndex < networkPhoneCount; ++phoneIndex) {
    if (host_phone_update_packets != nullptr) cudaFreeHost(host_phone_update_packets[phoneIndex]);
    if (host_phone_new_packets != nullptr) cudaFreeHost(host_phone_new_packets[phoneIndex]);
  }
  if (host_phone_update_packets != nullptr) cudaFreeHost(host_phone_update_packets);
  if (host_phone_new_packets != nullptr) cudaFreeHost(host_phone_new_packets);
  if (host_phone_update_payload_lengths != nullptr) cudaFreeHost(host_phone_update_payload_lengths);
  if (host_phone_new_payload_lengths != nullptr) cudaFreeHost(host_phone_new_payload_lengths);
  host_phone_update_packets = nullptr;
  host_phone_new_packets = nullptr;
  host_phone_update_payload_lengths = nullptr;
  host_phone_new_payload_lengths = nullptr;
  for (cudaEvent_t event : phone_packet_copy_ready_events) {
    cudaEventDestroy(event);
  }
  phone_packet_copy_ready_events.clear();
  if (phone_packet_sizes_ready_event != nullptr) {
    cudaEventDestroy(phone_packet_sizes_ready_event);
    phone_packet_sizes_ready_event = nullptr;
  }

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
  if (network_stream != nullptr) {
    cudaStreamDestroy(network_stream);
    network_stream = nullptr;
  }

}

void Boids::requestNetworkPacket(const CameraCullData *cameras, bool *cameraChanged,
  uint32_t phoneCount) {
  bool expected = false;
  if (!network_packet_request_outstanding.compare_exchange_strong(expected, true,
      std::memory_order_acq_rel)) {
    return;
  }

  prepareSimulationSnapshotsForNetwork();
  queuedNetworkCameras.assign(cameras, cameras + phoneCount);
  queuedNetworkCameraChanged = std::make_unique<bool[]>(phoneCount);
  for (uint32_t phoneIndex = 0; phoneIndex < phoneCount; ++phoneIndex) {
    queuedNetworkCameraChanged[phoneIndex] = cameraChanged[phoneIndex];
    cameraChanged[phoneIndex] = false;
  }
  queuedNetworkPhoneCount = phoneCount;
  ready_create_network_packet_semaphore.release();
}

void Boids::stopNetworkPacketCreationLoop() {
  stop_network_packet_creation_loop = true;
  ready_create_network_packet_semaphore.release();
}

void Boids::networkPacketCreationLoop() {
  while (true) {
    std::vector<CameraCullData> cameras;
    std::unique_ptr<bool[]> cameraChanged;
    uint32_t phoneCount = 0;

    // wait until ready for next packet
    ready_create_network_packet_semaphore.acquire();
    if (stop_network_packet_creation_loop) {
      return;
    }

    cameras = std::move(queuedNetworkCameras);
    phoneCount = queuedNetworkPhoneCount;
    cameraChanged = std::move(queuedNetworkCameraChanged);

    cullBoidsByCameras(cameras.data(), cameraChanged.get(), phoneCount);
    cudaMemcpyAsync(host_phone_update_payload_lengths, dev_phoneUpdatePayloadLengths,
      phoneCount * sizeof(uint32_t), cudaMemcpyDeviceToHost, network_stream);
    cudaMemcpyAsync(host_phone_new_payload_lengths, dev_phoneNewPayloadLengths,
      phoneCount * sizeof(uint32_t), cudaMemcpyDeviceToHost, network_stream);
#if DEBUG_MODE
    completedNetworkPackets.fetch_add(1, std::memory_order_relaxed);
#endif
    network_packet_request_outstanding.store(false, std::memory_order_release);

    cudaEventRecord(phone_packet_sizes_ready_event, network_stream);
    cudaEventSynchronize(phone_packet_sizes_ready_event);
    for (uint32_t phone = 0; phone < phoneCount; phone++) {
        phone_packet_ready_semaphore[phone]->release();
    }
  }
}

void Boids::phoneNetworkSendLoop(int phone_idx) {
  while (!stop_phone_packet_send_loop) {
    phone_packet_ready_semaphore[phone_idx]->acquire();
    if (stop_phone_packet_send_loop) {
      return;
    }


    const uint32_t updatePacketBytes = host_phone_update_payload_lengths[phone_idx];
    const uint32_t newPacketBytes = host_phone_new_payload_lengths[phone_idx];
    cudaMemcpyAsync(host_phone_update_packets[phone_idx], hostBoidUpdatePackets[phone_idx],
        updatePacketBytes, cudaMemcpyDeviceToHost,
        network_stream);
    cudaMemcpyAsync(host_phone_new_packets[phone_idx], hostBoidNewPackets[phone_idx],
        newPacketBytes, cudaMemcpyDeviceToHost,
        network_stream);

    cudaEventRecord(phone_packet_copy_ready_events[phone_idx], network_stream);
    cudaEventSynchronize(phone_packet_copy_ready_events[phone_idx]);

    Boids::UpdateBoidData *updatePacket = host_phone_update_packets[phone_idx];
    Boids::NewBoidData *newPacket = host_phone_new_packets[phone_idx];

    // actually send packets over network
  }
}
