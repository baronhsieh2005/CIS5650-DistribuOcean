#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <atomic>
#include <latch>
#include <memory>
#include <mutex>
#include <semaphore>

#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#define DEBUG_PACKET_LENGTHS 0
#endif

// Set to 1 only when profiling individual network packet pipeline stages.
#ifndef DEBUG_NETWORK_PIPELINE_TIMINGS
#define DEBUG_NETWORK_PIPELINE_TIMINGS 1
#endif

// Temporarily isolate network timings from simulation GPU work.
// Set to 0 to restore overlapping simulation and packet creation.
#ifndef DEBUG_PAUSE_SIMULATION_FOR_NETWORK
#define DEBUG_PAUSE_SIMULATION_FOR_NETWORK 1
#endif

namespace Boids {
    constexpr float cullDistance = 900.0f;
    constexpr uint16_t NEW_BOID_POLAR_ENCODING_BIT = 0x8000u;
    constexpr uint16_t NEW_BOID_SLOT_MASK = 0x7FFFu;

    struct ViewPositionEncoding {
        uint8_t x_pos;
        uint8_t y_pos;
        uint16_t depth;
    };

    struct PolarPositionEncoding {
        uint8_t azimuth;
        uint8_t elevation;
        uint16_t radial_depth;
    };

    struct CameraCullData {
        glm::vec3 position;
        glm::mat4 projMat;
    };

    struct NewBoidData {
        // Bit 15 is 1 for polar encoding and 0 for view-grid encoding.
        // Bits 0-14 hold the packet slot.
        uint16_t new_idx_slot;
        union {
            ViewPositionEncoding view;
            PolarPositionEncoding polar;
        } position;
    };
    static_assert(sizeof(NewBoidData) == 6, "New boid records must occupy six bytes");

    struct UpdateBoidData {
        int16_t pos_delta[3];
    };
    static_assert(sizeof(UpdateBoidData) == 6, "Update boid records must occupy six bytes");

    void initSimulation(int N);
    void stepSimulationCoherentGrid(float dt);
    void initializeNetworkPhones(const CameraCullData *cameras, uint32_t phoneCount);
    void createNetworkPackets();
    void cullBoidsByCameras(const CameraCullData *cameras, bool *cameraChanged,
                            uint32_t phoneCount);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    const uint32_t* getDeviceBoidIds();
    uint32_t* const* getDevicePhoneSelectedBoidIndices();
    const int32_t* getDevicePhoneVisibleCounts();
    const uint32_t* getDevicePhoneUpdatePayloadLengths();
    const uint32_t* getDevicePhoneNewPayloadLengths();
    UpdateBoidData* const* getDevicePhoneUpdatePayloads();
    NewBoidData* const* getDevicePhoneNewPayloads();
    void prepareSimulationSnapshotsForNetwork();

    void endSimulation();

    inline std::binary_semaphore ready_create_network_packet_semaphore{0};
#if DEBUG_PAUSE_SIMULATION_FOR_NETWORK
    inline std::binary_semaphore network_packet_build_done{0};
#endif
    inline std::atomic<bool> network_packet_request_outstanding{false};
    inline bool stop_network_packet_creation_loop = false;
    inline bool ready_request_network_packet = false;
    inline std::mutex ready_request_network_packet_lock;
    inline std::unique_ptr<std::latch> network_data_done_sending_latch;
    inline std::atomic<uint32_t> completedNetworkPackets{0};
    void requestNetworkPacket(const CameraCullData *cameras, bool *cameraChanged,
                              uint32_t phoneCount);
    void stopNetworkPacketCreationLoop();
    void stopPhoneNetworkSendLoops();
    void networkPacketCreationLoop();
    void phoneNetworkSendLoop(int phone_idx);
}

void batch_stream_compaction(int num_slots, int num_phones, int num_arrays, uint32_t *const *odata, uint32_t *const *idata, uint8_t *const *map, uint32_t *scan,
    uint32_t *block_sums, uint32_t *block_offsets,
    uint32_t *const *size_out, cudaStream_t stream);
