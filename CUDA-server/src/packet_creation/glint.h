#pragma once

// Initial tuning values; physical capacity is separate from split/merge thresholds.
#define ROOT_NODE_IDX 0
#define NO_PARTITION (-1)
#define NUM_BOIDS_PER_PARTITION 256
#define PARTITION_GRANULARITY_SPLIT 128
#define PARTITION_GRANULARITY_MERGE 64
#define NODE_STK_MAX_SIZE 65536
#define PART_STK_MAX_SIZE 65536
#define GLINT_INITIAL_MAX_DEPTH 5
#define GLINT_THREADS_PER_BLOCK 128
// Number of pending traversal entries reserved per filter block in shared memory.
#define GLINT_FILTER_STACK_CAPACITY 1024
// Initial refinement scratch capacity; filtering must enforce this limit.
#define GLINT_REFINE_PAIRS_PER_BOID 32
#define MAX_BOID_INTERACTION_DISTANCE \
    ((rule1Distance > rule2Distance) \
        ? ((rule1Distance > rule3Distance) ? rule1Distance : rule3Distance) \
        : ((rule2Distance > rule3Distance) ? rule2Distance : rule3Distance))
#define MAX_BOID_INTERACTION_DISTANCE_SQUARED \
    (MAX_BOID_INTERACTION_DISTANCE * MAX_BOID_INTERACTION_DISTANCE)

#include <cuda_runtime.h>
#include <cstdint>
#include <glm/glm.hpp>
#include "../config.h"

class GlintOctree {
public:
    struct LeafOutputCounts {
        int leaves;
        int parents;
    };

    struct GlintNode {
        int part_idx;
        glm::vec3 min_coords;
        glm::vec3 max_coords;
        int child_node_idx[8];
        int size;

        __host__ __device__ GlintNode()
            : part_idx(NO_PARTITION), min_coords(0.0f), max_coords(0.0f),
              child_node_idx{-1, -1, -1, -1, -1, -1, -1, -1}, size(-1)
        {}

        __host__ __device__ GlintNode(int size)
            : part_idx(NO_PARTITION), min_coords(0.0f), max_coords(0.0f),
              child_node_idx{-1, -1, -1, -1, -1, -1, -1, -1}, size(size)
        {}

        __host__ __device__ bool isLeaf() const {
            return child_node_idx[0] == -1;
        }
    };

    struct RefinePair {
        int pid;
        GlintNode *node;
    };

    struct FilterEntry {
        int pid;
        GlintNode *node;
    };

    struct PartitionBuff {
        uint32_t boid_idx[NUM_BOIDS_PER_PARTITION];
    };

    struct LeafData {
        const GlintNode *nodes;
        const PartitionBuff *partitions;
        const int *active_leaves;
        int count;
    };
    LeafData get_leaf_data() const {
        return {node_lst, part_lst, active_leaves, host_num_active_leaves};
    }
#if VISUALIZE
    uint64_t get_topology_revision() const { return topology_revision; }
#endif

    void init(int boid_count, const glm::vec3 *device_positions,
        int depth = GLINT_INITIAL_MAX_DEPTH);
    void deletion();

    void update_partition(const glm::vec3 *dev_positions, cudaStream_t simulation_stream);
    void update_filter(cudaStream_t simulation_stream);
    void update_velocity(const glm::vec3 *dev_vel1, glm::vec3 *dev_vel2, cudaStream_t simulation_stream);
    void update_tree(cudaStream_t simulation_stream);
private:
#if VISUALIZE
    uint64_t topology_revision = 0;
#endif
    // Borrowed simulation positions; all other device allocations below are owned.
    const glm::vec3 *points_lst = nullptr;
    GlintNode *node_lst = nullptr;
    int *node_stk = nullptr;
    PartitionBuff *part_lst = nullptr;
    int *part_stk = nullptr;
    FilterEntry *filter_lst = nullptr;
    RefinePair *refine_lst = nullptr;
    int *filter_lst_size = nullptr;
    int *refine_lst_size = nullptr;
    int host_filter_lst_size = 0;
    int host_refine_lst_size = 0;
    glm::vec3 *position_sum = nullptr;
    glm::vec3 *separation_sum = nullptr;
    glm::vec3 *velocity_sum = nullptr;
    int *cohesion_count = nullptr;
    int *alignment_count = nullptr;
    unsigned int *node_stk_head = nullptr;
    unsigned int *node_stk_tail = nullptr;
    unsigned int *part_stk_head = nullptr;
    unsigned int *part_stk_tail = nullptr;

    uint8_t *active_leaf_map = nullptr;
    uint8_t *potential_leaf_parent_map = nullptr;
    int *active_leaves = nullptr;
    int *next_active_leaves = nullptr;
    uint8_t *leafIsFirstChild = nullptr;
    uint8_t *next_leafIsFirstChild = nullptr;
    int *leaf_parents = nullptr;
    int *next_leaf_parents = nullptr;
    int *old_leaf_emit = nullptr;
    LeafOutputCounts *old_leaf_offsets = nullptr;
    int *potential_leaf_parents = nullptr;
    int *next_potential_leaf_parents = nullptr;
    int *num_active_leaves = nullptr;
    int *num_potential_leaf_parents = nullptr;
    int *next_num_active_leaves = nullptr;
    int *next_num_potential_leaf_parents = nullptr;
    int *topology_changed = nullptr;
    int host_topology_changed = 0;
    int host_num_active_leaves = 0;
    int host_num_potential_leaf_parents = 0;
    void *compact_temp = nullptr;
    size_t compact_temp_bytes = 0;
    int num_boids = 0;
    int max_depth = 0;
    int initial_node_count = 0;
};
