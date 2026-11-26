#pragma once

#include <cstdint>

// Scene and camera setup.
inline constexpr int N_FOR_VIS = 200000;
inline constexpr float DT = 0.2f;
inline constexpr uint32_t NUM_VIRTUAL_CAMERAS = 20;
// Camera offsets within the interior of the boids' starting regions.
inline constexpr float VIRTUAL_CAMERA_POSITION_EXTENT = 45.0f;
inline constexpr uint32_t VIRTUAL_CAMERA_RANDOM_SEED = 5650;
#ifndef VISUALIZE
#define VISUALIZE 0
#endif
inline constexpr double NETWORK_UPDATE_INTERVAL_SECONDS = 1.0 / 20.0;

// Simulation and shared CUDA launch configuration.
inline constexpr int blockSize = 128;
// Number of positions cached per G-PICS shared-memory tile.
#define GPIC_SMEM_TILE_SIZE 128
#define rule1Distance 15.0f
#define rule2Distance 9.0f
#define rule3Distance 15.0f
#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f
#define maxSpeed 3.0f
#define scene_scale 300.0f
#define spawnFlockRadius 90.0f

// Culling implementation and visibility predicate are independent choices.
#define NAIVE_CULLING 0
#define GPIC_CULLING 1
#ifndef CULLING_TYPE
#define CULLING_TYPE GPIC_CULLING
#endif

#define DIST_VIEW 0
#define DIST_SPHERE 1
#define MIXED 2
#ifndef CULL_TYPE
#define CULL_TYPE DIST_VIEW
#endif

#define cullDistanceTolerance 36.0f
#define cullFrustumTolerance 0.12f
#define mixedCullSphereFraction 0.5f

// Network packet capacity and encoding. Delta components use 1/1024 units.
#define MAX_BOIDS_PER_PHONE 20000
#define networkPositionDeltaScale (1.0f / 1024.0f)
#define networkPositionDepthMaxDistance (Boids::cullDistance + cullDistanceTolerance)
#define networkUpdateFreeComponent 0xFFFFu

namespace Boids {
    inline constexpr float cullDistance = 900.0f;
    inline constexpr uint16_t NEW_BOID_POLAR_ENCODING_BIT = 0x8000u;
    inline constexpr uint16_t NEW_BOID_SLOT_MASK = 0x7FFFu;
}

// Debug switches can also be overridden through compiler definitions.
#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#endif
#ifndef DEBUG_PACKET_LENGTHS
#define DEBUG_PACKET_LENGTHS 0
#endif
#ifndef DEBUG_NETWORK_PIPELINE_TIMINGS
#define DEBUG_NETWORK_PIPELINE_TIMINGS 1
#endif
// Set to 0 to overlap simulation with packet creation.
#ifndef DEBUG_PAUSE_SIMULATION_FOR_NETWORK
#define DEBUG_PAUSE_SIMULATION_FOR_NETWORK 1
#endif
