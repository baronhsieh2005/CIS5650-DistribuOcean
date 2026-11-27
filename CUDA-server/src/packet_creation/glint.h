#pragma once

// Initial tuning values; physical capacity is separate from split/merge thresholds.
#define ROOT_NODE_IDX 0
#define NO_PARTITION (-1)
#define NUM_BOIDS_PER_PARTITION 64
#define PARTITION_GRANULARITY_SPLIT 50
#define PARTITION_GRANULARITY_MERGE 25
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

    void init(int boid_count, const glm::vec3 *device_positions,
        int depth = GLINT_INITIAL_MAX_DEPTH);
    void deletion();
    // Run these stages in order, with snapshots/queries before update_tree.
    void update_partition(const glm::vec3 *dev_positions);
    void update_filter();
    void update_velocity(const glm::vec3 *dev_vel1, glm::vec3 *dev_vel2);
    void update_tree();
};

__device__ int getChildFromPos(const glm::vec3 &position,
    const GlintOctree::GlintNode *node);
// Both helpers return squared distances. Boundary distance assumes an interior point.
__device__ float get_distance(const glm::vec3 &position,
    const GlintOctree::GlintNode *node);
__device__ float getMinDistToNode(const glm::vec3 &position,
    const GlintOctree::GlintNode *node);
// Returns -1 for the root or a node not reachable in the current topology.
__device__ int get_parent(int node_idx, const GlintOctree::GlintNode *node_lst);
__global__ void build_init_octree(int num_nodes, int max_depth, GlintOctree::GlintNode *node_lst);
__global__ void init_partition(int num_boids, const glm::vec3 *points_lst, GlintOctree::GlintNode *node_lst);
__global__ void init_leaves(int num_nodes, GlintOctree::GlintNode *node_lst, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, uint8_t *active_leaf_map, uint8_t *potential_leaf_parent_map);
__global__ void partition(int num_boids, const glm::vec3 *points_lst,
    GlintOctree::GlintNode *node_lst, GlintOctree::PartitionBuff *part_lst,
    GlintOctree::FilterEntry *filter_lst, int *filter_lst_size,
    GlintOctree::RefinePair *refine_lst, int *refine_lst_size);
__device__ GlintOctree::FilterEntry *get_block_stack(
    GlintOctree::FilterEntry *storage, int *stack_size, int seed_count);
__global__ void filter(int filter_lst_size, const GlintOctree::FilterEntry *filter_lst,
    const glm::vec3 *points_lst, GlintOctree::GlintNode *node_lst,
    GlintOctree::RefinePair *refine_lst, int *refine_lst_size);
__global__ void reset_interaction_accumulators(int num_boids, glm::vec3 *position_sum,
    glm::vec3 *separation_sum, glm::vec3 *velocity_sum, int *cohesion_count, int *alignment_count);
__global__ void refinement(int num_pairs, const GlintOctree::RefinePair *refine_lst,
    const glm::vec3 *points_lst, const GlintOctree::PartitionBuff *part_lst,
    const glm::vec3 *vel1, glm::vec3 *position_sum, glm::vec3 *separation_sum,
    glm::vec3 *velocity_sum, int *cohesion_count, int *alignment_count);
__global__ void finalize_velocity(int num_boids, const glm::vec3 *points_lst,
    const glm::vec3 *vel1, glm::vec3 *vel2, const glm::vec3 *position_sum,
    const glm::vec3 *separation_sum, const glm::vec3 *velocity_sum,
    const int *cohesion_count, const int *alignment_count);
__global__ void split_node(int num_active_leaves, int *active_leaves, GlintOctree::GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, int *topology_changed, int *old_leaf_emit);
__global__ void merge_node(int num_potential_leaf_parents, const int *potential_leaf_parents, GlintOctree::GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, bool alloc_partition, int *topology_changed);
__global__ void merge_depth(int first_node, int node_count, GlintOctree::GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, bool alloc_partition);

__global__ void reset_leaf_count(int num_active_leaves, const int *active_leaves, GlintOctree::GlintNode *node_lst);

__global__ void update_leaf_merge_events(int num_active_leaves, int *active_leaves,
    GlintOctree::GlintNode *node_lst, int *leaf_parents, uint8_t *leafIsFirstChild, int *old_leaf_emit);

__global__ void update_active_leaf_events(int num_active_leaves, const int *active_leaves,
    const int *leaf_parents, const uint8_t *leafIsFirstChild, const GlintOctree::GlintNode *node_lst,
    int *new_active_leaves, int *new_leaf_parents, uint8_t *new_leafIsFirstChild,
    int *new_num_active_leaves, const GlintOctree::LeafOutputCounts *new_idx, const int *old_leaf_emit,
    int *new_potential_leaf_parents, int *new_num_potential_leaf_parents);
