#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace Boids {
    struct Flock {
        glm::vec3 position;
        float radius;
        glm::vec3 direction;
        uint32_t startIndex;
        uint32_t numBoids;
    };

    struct CompressedBoid {
        uint8_t radius;
        uint8_t azimuth;
        uint8_t polar;
        uint8_t rotationX;
        uint8_t rotationY;
        uint8_t rotationZ;
    };
    static_assert(sizeof(CompressedBoid) == 6, "Compressed boids must occupy six bytes");

    struct CameraCullData {
        glm::vec3 position;
        float maxDistance;
        glm::mat4 projMat;
    };

    struct NetworkFlock {
        glm::vec3 position;
        float radius;
        glm::vec3 direction;
        uint32_t numBoids;
    };
    static_assert(sizeof(NetworkFlock) == 32, "Network flock metadata must occupy 32 bytes");

    void initSimulation(int N);
    void stepSimulationCoherentGrid(float dt);
    void computeFlocks();
    void initializeNetworkPhones(uint32_t phoneCount);
    void createNetworkPackets(const CameraCullData *cameras, uint32_t phoneCount);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    const Flock* getDeviceFlocks();
    const CompressedBoid* getDeviceCompressedBoids();
    const uint32_t* getDeviceFlockCount();
    uint8_t* const* getDevicePhonePackets();
    const uint32_t* getDevicePhonePacketSizes();

    void endSimulation();
}
