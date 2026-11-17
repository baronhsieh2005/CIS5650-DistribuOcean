#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#define DEBUG_PACKET_LENGTHS 0
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
    float* const* getDevicePhoneDistances();
    const int32_t* getDevicePhoneVisibleCounts();
    const uint32_t* getDevicePhoneUpdatePayloadLengths();
    const uint32_t* getDevicePhoneNewPayloadLengths();
    UpdateBoidData* const* getDevicePhoneUpdatePayloads();
    NewBoidData* const* getDevicePhoneNewPayloads();
    void prepareSimulationSnapshotsForNetwork();

    void endSimulation();
}
