#pragma once

#include "config.h"

#if VISUALIZE && SIMULATION_TYPE == GLINT_SIMULATION
#define GLM_FORCE_CUDA
#include <cuda.h>
#include "glintVisualizer.h"
#include "glslUtility.hpp"
#include "packet_creation/glint.h"
#include <cuda_gl_interop.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

extern GlintOctree glintOctree;
extern cudaStream_t simulation_stream;

namespace {
constexpr int verticesPerLeaf = 24;
struct Vertex {
    glm::vec4 position;
    glm::vec4 color;
};
bool visible = false;
GLuint vao = 0;
GLuint vbo = 0;
cudaGraphicsResource *resource = nullptr;
int leafCapacity = 0;
int vertexCount = 0;
uint64_t lastRevision = std::numeric_limits<uint64_t>::max();

void checkCUDA(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("GLINT visualizer: ") + cudaGetErrorString(result));
}

__global__ void buildWireframe(int leafCount, const int *activeLeaves,
    const GlintOctree::GlintNode *nodes, Vertex *vertices) {
    const int leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= leafCount) return;
    const auto &node = nodes[activeLeaves[leafIdx]];
    const float depth = log2f((2.0f * scene_scale) / (node.max_coords.x - node.min_coords.x));
    const float shade = fminf(depth / 10.0f, 1.0f);
    const glm::vec4 color(0.1f + 0.5f * shade, 1.0f - 0.4f * shade, 1.0f, 1.0f);
    int output = leafIdx * verticesPerLeaf;
    // Each of the 12 cube edges connects corners differing in one axis bit.
    for (int axis = 0; axis < 3; ++axis) {
        for (int corner = 0; corner < 8; ++corner) {
            if (corner & (1 << axis)) continue;
            const int other = corner | (1 << axis);
            const glm::vec3 a((corner & 1) ? node.max_coords.x : node.min_coords.x,
                (corner & 2) ? node.max_coords.y : node.min_coords.y,
                (corner & 4) ? node.max_coords.z : node.min_coords.z);
            const glm::vec3 b((other & 1) ? node.max_coords.x : node.min_coords.x,
                (other & 2) ? node.max_coords.y : node.min_coords.y,
                (other & 4) ? node.max_coords.z : node.min_coords.z);
            vertices[output++] = {glm::vec4(a, 1.0f), color};
            vertices[output++] = {glm::vec4(b, 1.0f), color};
        }
    }
}

void reserveVertices(int leafCount) {
    if (leafCount <= leafCapacity) return;
    if (resource) {
        checkCUDA(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
    }
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }
    leafCapacity = std::max(leafCount, leafCapacity * 2);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<size_t>(leafCapacity) * verticesPerLeaf * sizeof(Vertex),
        nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void *>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void *>(offsetof(Vertex, color)));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    checkCUDA(cudaGraphicsGLRegisterBuffer(&resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
}
}

void GlintVisualizer::toggle() {
    visible = !visible;
}

void GlintVisualizer::draw(unsigned int shaderProgram) {
    if (!visible) return;
    const uint64_t revision = glintOctree.get_topology_revision();
    if (revision != lastRevision) {
        const auto leaves = glintOctree.get_leaf_data();
        reserveVertices(leaves.count);
        vertexCount = leaves.count * verticesPerLeaf;
        if (vertexCount) {
            checkCUDA(cudaGraphicsMapResources(1, &resource, simulation_stream));
            Vertex *vertices = nullptr;
            size_t bytes = 0;
            checkCUDA(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&vertices), &bytes, resource));
            buildWireframe<<<(leaves.count + blockSize - 1) / blockSize, blockSize, 0, simulation_stream>>>(
                leaves.count, leaves.active_leaves, leaves.nodes, vertices);
            checkCUDA(cudaGetLastError());
            checkCUDA(cudaGraphicsUnmapResources(1, &resource, simulation_stream));
        }
        lastRevision = revision;
    }
    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_LINES, 0, vertexCount);
    glBindVertexArray(0);
}

void GlintVisualizer::shutdown() {
    if (resource) checkCUDA(cudaGraphicsUnregisterResource(resource));
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    resource = nullptr;
    vao = vbo = 0;
    leafCapacity = vertexCount = 0;
    visible = false;
    lastRevision = std::numeric_limits<uint64_t>::max();
}
#endif
