#define GLM_FORCE_CUDA
#include <cuda.h>
#include "glint.h"
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <cstdio>
#if DEBUG_GLINT_PIPELINE_TIMINGS || DEBUG_GLINT_TREE_UPDATES
#include <sstream>
#endif

#if DEBUG_GLINT_PIPELINE_TIMINGS
#define GLINT_TIME_BEGIN(stage) begin_debug_stage(GLINT_TIMING_##stage, simulation_stream)
#define GLINT_TIME_END(stage) end_debug_stage(GLINT_TIMING_##stage, simulation_stream)
#else
#define GLINT_TIME_BEGIN(stage) ((void)0)
#define GLINT_TIME_END(stage) ((void)0)
#endif

#if VISUALIZE && SIMULATION_TYPE == GLINT_SIMULATION
#include "../glintVisualizer.cuh"
#endif

using GlintNode = GlintOctree::GlintNode;
using PartitionBuff = GlintOctree::PartitionBuff;
using RefinePair = GlintOctree::RefinePair;
using FilterEntry = GlintOctree::FilterEntry;
using LeafOutputCounts = GlintOctree::LeafOutputCounts;

struct GetLeafOutputCounts {
    const int *old_leaf_emit;
    const int *leaf_parents;
    const uint8_t *leafIsFirstChild;

    __host__ __device__ LeafOutputCounts operator()(int leaf_idx) const {
        const int emit = old_leaf_emit[leaf_idx];
        const int parent_emit = emit == 8 ||
            (emit == 1 && leafIsFirstChild[leaf_idx] && leaf_parents[leaf_idx] >= 0);
        return {emit, parent_emit};
    }
};

struct AddLeafOutputCounts {
    __host__ __device__ LeafOutputCounts operator()(LeafOutputCounts a,
        LeafOutputCounts b) const {
        return {a.leaves + b.leaves, a.parents + b.parents};
    }
};

__device__ void push(int *stk, int val, unsigned int *head,
    unsigned int *tail, unsigned int max_size);
__device__ int pop(int *stk, unsigned int *head,
    unsigned int *tail, unsigned int max_size);

__device__ int getChildFromPos(const glm::vec3 &position, const GlintNode *node) {
    const glm::vec3 midpoint = node->min_coords +
        (node->max_coords - node->min_coords) * 0.5f;
    return (position.x >= midpoint.x ? 1 : 0) |
           (position.y >= midpoint.y ? 2 : 0) |
           (position.z >= midpoint.z ? 4 : 0);
}

__device__ float get_distance(const glm::vec3 &position, const GlintNode *node) {
    const glm::vec3 delta = glm::max(glm::max(node->min_coords - position,
        position - node->max_coords), glm::vec3(0.0f));
    return glm::dot(delta, delta);
}

__device__ float getMinDistToNode(const glm::vec3 &position, const GlintNode *node) {
    const glm::vec3 face_distances = glm::min(position - node->min_coords,
        node->max_coords - position);
    const float distance = glm::max(0.0f,
        glm::min(face_distances.x, glm::min(face_distances.y, face_distances.z)));
    return distance * distance;
}

__device__ int get_parent(int node_idx, const GlintNode *node_lst) {
    if (node_idx == ROOT_NODE_IDX || node_idx < 0) return -1;
    const glm::vec3 center = node_lst[node_idx].min_coords +
        (node_lst[node_idx].max_coords - node_lst[node_idx].min_coords) * 0.5f;
    int current = ROOT_NODE_IDX;
    while (!node_lst[current].isLeaf()) {
        const int child = node_lst[current].child_node_idx[
            getChildFromPos(center, &node_lst[current])];
        if (child == node_idx) return current;
        if (child < 0) return -1;
        current = child;
    }
    return -1;
}

__global__ void init_leaf_first_child(int leaf_count, const int *active_leaves,
    const GlintNode *node_lst, uint8_t *leafIsFirstChild, int *leaf_parents) {
    const int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= leaf_count) return;
    const int node_idx = active_leaves[leaf_idx];
    const int parent = get_parent(node_idx, node_lst);
    leaf_parents[leaf_idx] = parent;
    leafIsFirstChild[leaf_idx] = parent >= 0 &&
        node_lst[parent].child_node_idx[0] == node_idx;
}

__global__ void init_pools(GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, int initial_node_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NODE_STK_MAX_SIZE) {
        node_lst[i] = GlintNode();
        node_stk[i] = i;
    }
    if (i < PART_STK_MAX_SIZE) part_stk[i] = i;
    if (i == 0) {
        *node_stk_head = initial_node_count;
        *node_stk_tail = NODE_STK_MAX_SIZE-1;
        *part_stk_head = 0;
        *part_stk_tail = PART_STK_MAX_SIZE-1;
    }
}

__global__ void build_init_octree(int num_nodes, int max_depth, GlintNode *node_lst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    int depth = 0, level_start = 0, level_count = 1;
    while (i >= level_start + level_count) {
        level_start += level_count;
        level_count *= 8;
        ++depth;
    }
    GlintNode node(0);
    node.min_coords = glm::vec3(-scene_scale);
    node.max_coords = glm::vec3(scene_scale);
    const int local_idx = i - level_start;
    for (int d = depth - 1; d >= 0; --d) {
        const int octant = (local_idx >> (3 * d)) & 7;
        const glm::vec3 midpoint = node.min_coords +
            (node.max_coords - node.min_coords) * 0.5f;
        for (int axis = 0; axis < 3; ++axis) {
            if (octant & (1 << axis)) node.min_coords[axis] = midpoint[axis];
            else node.max_coords[axis] = midpoint[axis];
        }
    }
    if (depth < max_depth)
        for (int c = 0; c < 8; ++c) node.child_node_idx[c] = 8 * i + 1 + c;
    node_lst[i] = node;
}

__global__ void build_init_octree_old(int max_depth, GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        GlintNode *root = &node_lst[ROOT_NODE_IDX];
        *root = GlintNode(0);
        root->min_coords = glm::vec3(-scene_scale);
        root->max_coords = glm::vec3(scene_scale);
    }
    __syncthreads();
    unsigned int curr_layer_max = 8;
    for (int d = 1; d <= max_depth; ++d, curr_layer_max *= 8) {
        unsigned int desired_node_id = tid;
        while (desired_node_id < curr_layer_max) {
            unsigned int lo = 0;
            unsigned int hi = curr_layer_max;
            GlintNode *curr_node = &node_lst[ROOT_NODE_IDX];
            for (int i = 0; i < d - 1; ++i) {
                const unsigned int boundary = (hi - lo) / 8;
                const unsigned int desired_child = (desired_node_id - lo) / boundary;
                lo += boundary * desired_child;
                hi = lo + boundary;
                curr_node = &node_lst[curr_node->child_node_idx[desired_child]];
            }
            const unsigned int child_idx = desired_node_id - lo;
            const int next_node_id = pop(node_stk, node_stk_head,
                node_stk_tail, NODE_STK_MAX_SIZE);
            curr_node->child_node_idx[child_idx] = next_node_id;
            GlintNode *next_node = &node_lst[next_node_id];
            *next_node = GlintNode(0);
            const glm::vec3 midpoint = curr_node->min_coords +
                (curr_node->max_coords - curr_node->min_coords) * 0.5f;
            next_node->min_coords = glm::vec3(
                (child_idx & 1) ? midpoint.x : curr_node->min_coords.x,
                (child_idx & 2) ? midpoint.y : curr_node->min_coords.y,
                (child_idx & 4) ? midpoint.z : curr_node->min_coords.z);
            next_node->max_coords = glm::vec3(
                (child_idx & 1) ? curr_node->max_coords.x : midpoint.x,
                (child_idx & 2) ? curr_node->max_coords.y : midpoint.y,
                (child_idx & 4) ? curr_node->max_coords.z : midpoint.z);
            desired_node_id += blockDim.x;
        }
        __syncthreads();
    }
}

__global__ void init_partition(int num_boids, const glm::vec3 *points_lst, GlintNode *node_lst) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= num_boids) {
        return;
    }

    GlintNode *nd = &node_lst[ROOT_NODE_IDX];
    while (!nd->isLeaf()) {
        nd = &node_lst[nd->child_node_idx[getChildFromPos(points_lst[pid], nd)]];
    }

    atomicAdd(&nd->size, 1);
}

__global__ void init_leaves(int num_nodes, GlintNode *node_lst, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, uint8_t *active_leaf_map, uint8_t *potential_leaf_parent_map) {
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    active_leaf_map[node_idx] = 0;
    potential_leaf_parent_map[node_idx] = 0;
    GlintNode *node = &node_lst[node_idx];
    if (node->size < 0) return;
    if (node->isLeaf()) {
        active_leaf_map[node_idx] = 1;
        node->size = 0;
        node->part_idx = pop(part_stk, part_stk_head,
            part_stk_tail, PART_STK_MAX_SIZE);
    } else if (node_lst[node->child_node_idx[0]].isLeaf()) {
        potential_leaf_parent_map[node_idx] = 1;
    }
}

__global__ void reset_leaf_count(int num_active_leaves, const int *active_leaves, GlintNode *node_lst) {
    const int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_active_leaves) {
        return;
    }

    node_lst[active_leaves[leaf_idx]].size = 0;
}

__global__ void update_leaf_merge_events(int num_active_leaves, int *active_leaves,
    GlintNode *node_lst, int *leaf_parents, uint8_t *leafIsFirstChild, int *old_leaf_emit) {
    const int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_active_leaves) {
        return;
    }

    GlintNode *leaf = &node_lst[active_leaves[leaf_idx]];
    if (leaf->size == -1) {
        if (leafIsFirstChild[leaf_idx]) {
            old_leaf_emit[leaf_idx] = 1;
            int parent_idx = leaf_parents[leaf_idx];
            active_leaves[leaf_idx] = parent_idx;
            leaf_parents[leaf_idx] = get_parent(parent_idx, node_lst);
            const int grandparent = leaf_parents[leaf_idx];
            leafIsFirstChild[leaf_idx] = grandparent >= 0 &&
                node_lst[grandparent].child_node_idx[0] == parent_idx;
        } else {
            old_leaf_emit[leaf_idx] = 0;
        }
    }
}

__global__ void update_active_leaf_events(int num_active_leaves, const int *active_leaves,
        const int *leaf_parents, const uint8_t *leafIsFirstChild, const GlintNode *node_lst,
        int *new_active_leaves, int *new_leaf_parents, uint8_t *new_leafIsFirstChild,
        int *new_num_active_leaves, const LeafOutputCounts *new_idx, const int *old_leaf_emit,
        int *new_potential_leaf_parents, int *new_num_potential_leaf_parents) {
    const int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_active_leaves) return;
    const int emit_count = old_leaf_emit[leaf_idx];
    if (leaf_idx == num_active_leaves - 1) {
        const LeafOutputCounts counts = GetLeafOutputCounts{
            old_leaf_emit, leaf_parents, leafIsFirstChild}(leaf_idx);
        *new_num_active_leaves = new_idx[leaf_idx].leaves + counts.leaves;
        *new_num_potential_leaf_parents = new_idx[leaf_idx].parents + counts.parents;
    }
    if (emit_count == 0) return;

    int new_leaf_idx = new_idx[leaf_idx].leaves;
    if (emit_count == 1) {
        new_active_leaves[new_leaf_idx] = active_leaves[leaf_idx];
        new_leaf_parents[new_leaf_idx] = leaf_parents[leaf_idx];
        new_leafIsFirstChild[new_leaf_idx] = leafIsFirstChild[leaf_idx];
        if (leafIsFirstChild[leaf_idx] && leaf_parents[leaf_idx] >= 0) {
            const int rank = new_idx[leaf_idx].parents;
            new_potential_leaf_parents[rank] = leaf_parents[leaf_idx];
        }
    } else {
        int parent_idx = active_leaves[leaf_idx];
        const GlintNode *parent_node = &node_lst[parent_idx];
        for (int i = 0; i < 8; i++) {
            new_active_leaves[new_leaf_idx+i] = parent_node->child_node_idx[i];
            new_leaf_parents[new_leaf_idx+i] = parent_idx;
            new_leafIsFirstChild[new_leaf_idx+i] = i == 0;
        }
        const int rank = new_idx[leaf_idx].parents;
        new_potential_leaf_parents[rank] = parent_idx;
    }
}

__device__ void push_to_partition(GlintNode *node, int val, PartitionBuff *part_lst) {
    if (node->part_idx == NO_PARTITION) {
        return;
    }

    PartitionBuff *partition = &part_lst[node->part_idx];
    int old_size = atomicAdd(&node->size, 1);
    if (old_size >= NUM_BOIDS_PER_PARTITION) {
        // Error
        return;
    }
    partition->boid_idx[old_size] = val;
}

// Every lane in participating must call this, including lanes that do not emit.
__device__ int reserve_tasks(int *counter, bool emit, unsigned int participating) {
    const unsigned int mask = __ballot_sync(participating, emit);
    if (mask == 0) return -1;
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(mask) - 1;
    int start = 0;
    if (lane == leader) start = atomicAdd(counter, __popc(mask));
    start = __shfl_sync(participating, start, leader);
    return emit ? start + __popc(mask & ((1u << lane) - 1u)) : -1;
}

__global__ void partition(int num_boids, const glm::vec3 *points_lst,
    GlintNode *node_lst, PartitionBuff *part_lst,
    FilterEntry *filter_lst, int *filter_lst_size,
    RefinePair *refine_lst, int *refine_lst_size) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int participating = __ballot_sync(__activemask(), pid < num_boids);
    if (pid >= num_boids) {
        return;
    }
    // Only filter appends refinement tasks, after this kernel has completed.
    if (pid == 0) *refine_lst_size = num_boids;

    GlintNode *nd = &node_lst[ROOT_NODE_IDX];
    GlintNode *filter_root = &node_lst[ROOT_NODE_IDX];
    const glm::vec3 position = points_lst[pid];
    while (!nd->isLeaf()) {
        const float distance_squared = getMinDistToNode(position, nd);
        if (distance_squared > MAX_BOID_INTERACTION_DISTANCE_SQUARED) {
            filter_root = nd;
        }

        nd = &node_lst[nd->child_node_idx[getChildFromPos(position, nd)]];
    }

    push_to_partition(nd, pid, part_lst);
    
    refine_lst[pid] = RefinePair(pid, nd);

    const bool needs_filter = !filter_root->isLeaf() &&
        getMinDistToNode(position, nd) <= MAX_BOID_INTERACTION_DISTANCE_SQUARED;
    const int rank = reserve_tasks(filter_lst_size, needs_filter, participating);
    if (needs_filter) {
        filter_lst[rank] = FilterEntry(pid, static_cast<int>(filter_root - node_lst));
    }
}

__device__ FilterEntry *get_block_stack(FilterEntry *storage,
    int *stack_size, int seed_count) {
    if (threadIdx.x == 0) *stack_size = seed_count;
    __syncthreads();
    return storage;
}

__global__ void filter(int filter_lst_size, const FilterEntry *filter_lst,
    const glm::vec3 *points_lst, GlintNode *node_lst,
    RefinePair *refine_lst, int *refine_lst_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ GlintOctree::FilterEntry stack[];
    __shared__ int stackSize;
    __shared__ int popCount;
    __shared__ int popBase;
    const int remaining = filter_lst_size - blockIdx.x * blockDim.x;
    const int seed_count = remaining < blockDim.x ? remaining : blockDim.x;
    FilterEntry *block_stk = get_block_stack(stack, &stackSize, seed_count);
    if (tid < filter_lst_size) block_stk[threadIdx.x] = filter_lst[tid];

    __syncthreads();
    while (true) {
        if (threadIdx.x == 0) {
            popCount = stackSize < blockDim.x ? stackSize : blockDim.x;
            stackSize -= popCount;
            popBase = stackSize;
        }
        __syncthreads();
        
        if (popCount == 0) {
            break;
        }
        
        const bool has_work = threadIdx.x < popCount;
        FilterEntry entry;
        if (has_work) {
            entry = stack[popBase + threadIdx.x];
        }
        __syncthreads();

        const unsigned int work_mask = __ballot_sync(__activemask(), has_work);
        if (has_work) {
            int pid = entry.pid;
            GlintNode *nd = &node_lst[entry.node_idx];
            const glm::vec3 position = points_lst[pid];

            for (int child_idx : nd->child_node_idx) {
                GlintNode *child = &node_lst[child_idx];
                const float distance_squared = get_distance(position, child);
                const bool intersects = distance_squared <= MAX_BOID_INTERACTION_DISTANCE_SQUARED;
                const bool leaf = child->isLeaf();
                const bool push_child = intersects && !leaf;
                bool refine_child = false;
                if (intersects && leaf && child->size > 0) {
                    const glm::vec3 root_max = node_lst[ROOT_NODE_IDX].max_coords;
                    const bool own_leaf = distance_squared == 0.0f &&
                        (position.x < child->max_coords.x || child->max_coords.x == root_max.x) &&
                        (position.y < child->max_coords.y || child->max_coords.y == root_max.y) &&
                        (position.z < child->max_coords.z || child->max_coords.z == root_max.z);
                    refine_child = !own_leaf;
                }
                const int stack_rank = reserve_tasks(&stackSize, push_child, work_mask);
                if (push_child) stack[stack_rank] = FilterEntry(pid, child_idx);
                const int refine_rank = reserve_tasks(refine_lst_size, refine_child, work_mask);
                if (refine_child) refine_lst[refine_rank] = RefinePair(pid, child);
            }
        }
        __syncthreads();
    }
}

__global__ void reset_interaction_accumulators(int num_boids, glm::vec3 *position_sum,
    glm::vec3 *separation_sum, glm::vec3 *velocity_sum, int *cohesion_count, int *alignment_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boids) return;
    position_sum[i] = glm::vec3(0.0f);
    separation_sum[i] = glm::vec3(0.0f);
    velocity_sum[i] = glm::vec3(0.0f);
    cohesion_count[i] = 0;
    alignment_count[i] = 0;
}

__global__ void refinement(int num_pairs, const RefinePair *refine_lst,
    const glm::vec3 *points_lst, const PartitionBuff *part_lst,
    const glm::vec3 *vel1, glm::vec3 *position_sum, glm::vec3 *separation_sum,
    glm::vec3 *velocity_sum, int *cohesion_count, int *alignment_count) {
    int pair_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (pair_idx >= num_pairs) {
        return;
    }
    
    const RefinePair *pair = &refine_lst[pair_idx];
    uint32_t pid1 = pair->pid;
    const GlintNode *node = pair->node;
    const PartitionBuff *partition = &part_lst[node->part_idx];
    const glm::vec3 position = points_lst[pid1];
    glm::vec3 local_position_sum(0.0f);
    glm::vec3 local_separation_sum(0.0f);
    glm::vec3 local_velocity_sum(0.0f);
    int local_cohesion_count = 0;
    int local_alignment_count = 0;
    bool has_separation = false;
    for (int i = 0; i < node->size; ++i) {
        const uint32_t pid2 = partition->boid_idx[i];
        if (pid2 == pid1) continue;
        const glm::vec3 neighbor_position = points_lst[pid2];
        const glm::vec3 delta = neighbor_position - position;
        const float distance_squared = glm::dot(delta, delta);
        if (distance_squared < rule1Distance * rule1Distance) {
            local_position_sum += neighbor_position;
            ++local_cohesion_count;
        }
        if (distance_squared < rule2Distance * rule2Distance) {
            local_separation_sum -= delta;
            has_separation = true;
        }
        if (distance_squared < rule3Distance * rule3Distance) {
            local_velocity_sum += vel1[pid2];
            ++local_alignment_count;
        }
    }
    // Publish once per target/leaf task, rather than once per neighbor.
    if (local_cohesion_count) {
        atomicAdd(&position_sum[pid1].x, local_position_sum.x);
        atomicAdd(&position_sum[pid1].y, local_position_sum.y);
        atomicAdd(&position_sum[pid1].z, local_position_sum.z);
        atomicAdd(&cohesion_count[pid1], local_cohesion_count);
    }
    if (has_separation) {
        atomicAdd(&separation_sum[pid1].x, local_separation_sum.x);
        atomicAdd(&separation_sum[pid1].y, local_separation_sum.y);
        atomicAdd(&separation_sum[pid1].z, local_separation_sum.z);
    }
    if (local_alignment_count) {
        atomicAdd(&velocity_sum[pid1].x, local_velocity_sum.x);
        atomicAdd(&velocity_sum[pid1].y, local_velocity_sum.y);
        atomicAdd(&velocity_sum[pid1].z, local_velocity_sum.z);
        atomicAdd(&alignment_count[pid1], local_alignment_count);
    }
}

#if GLINT_COOPERATIVE_REFINEMENT
__global__ void count_refinement_tasks(int num_pairs, const RefinePair *refine_lst,
    int *refine_leaf_counts) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) {
        return;
    }
    atomicAdd(&refine_leaf_counts[refine_lst[pair_idx].node->part_idx], 1);
}

__global__ void scatter_refinement_tasks(int num_pairs, const RefinePair *refine_lst,
    const int *refine_leaf_offsets, int *refine_leaf_counts, int *refine_grouped_pids) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) {
        return;
    }
    const RefinePair pair = refine_lst[pair_idx];
    const int part_idx = pair.node->part_idx;
    const int rank = atomicAdd(&refine_leaf_counts[part_idx], 1);
    refine_grouped_pids[refine_leaf_offsets[part_idx] + rank] = pair.pid;
}

__global__ void refinement_cooperative(int num_leaves, const int *active_leaves,
    const GlintNode *node_lst, const PartitionBuff *part_lst,
    const int *refine_leaf_counts, const int *refine_leaf_offsets,
    const int *refine_grouped_pids, const glm::vec3 *points_lst,
    const glm::vec3 *vel1, glm::vec3 *position_sum, glm::vec3 *separation_sum,
    glm::vec3 *velocity_sum, int *cohesion_count, int *alignment_count) {
    const int leaf_idx = blockIdx.x;
    if (leaf_idx >= num_leaves) return;
    const GlintNode *node = &node_lst[active_leaves[leaf_idx]];
    const int size = node->size;
    const int part_idx = node->part_idx;
    const int count = refine_leaf_counts[part_idx];
    if (size == 0 || count == 0) return;

    __shared__ uint32_t shared_ids[NUM_BOIDS_PER_PARTITION];
    __shared__ float shared_x[NUM_BOIDS_PER_PARTITION];
    __shared__ float shared_y[NUM_BOIDS_PER_PARTITION];
    __shared__ float shared_z[NUM_BOIDS_PER_PARTITION];
    const PartitionBuff *partition = &part_lst[part_idx];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const uint32_t pid = partition->boid_idx[i];
        const glm::vec3 position = points_lst[pid];
        shared_ids[i] = pid;
        shared_x[i] = position.x;
        shared_y[i] = position.y;
        shared_z[i] = position.z;
    }
    __syncthreads();

    const int start = refine_leaf_offsets[part_idx];
    // Read-only shared data needs no additional barriers between target chunks.
    for (int target = threadIdx.x; target < count; target += blockDim.x) {
        const uint32_t pid1 = refine_grouped_pids[start + target];
        const glm::vec3 position = points_lst[pid1];
        glm::vec3 local_position_sum(0.0f);
        glm::vec3 local_separation_sum(0.0f);
        glm::vec3 local_velocity_sum(0.0f);
        int local_cohesion_count = 0;
        int local_alignment_count = 0;
        bool has_separation = false;
        for (int i = 0; i < size; ++i) {
            const uint32_t pid2 = shared_ids[i];
            if (pid2 == pid1) continue;
            const glm::vec3 neighbor_position = glm::vec3(shared_x[i], shared_y[i], shared_z[i]);
            const glm::vec3 delta = neighbor_position - position;
            const float distance_squared = glm::dot(delta, delta);
            if (distance_squared < rule1Distance * rule1Distance) {
                local_position_sum += neighbor_position;
                ++local_cohesion_count;
            }
            if (distance_squared < rule2Distance * rule2Distance) {
                local_separation_sum -= delta;
                has_separation = true;
            }
            if (distance_squared < rule3Distance * rule3Distance) {
                local_velocity_sum += vel1[pid2];
                ++local_alignment_count;
            }
        }
        if (local_cohesion_count) {
            atomicAdd(&position_sum[pid1].x, local_position_sum.x);
            atomicAdd(&position_sum[pid1].y, local_position_sum.y);
            atomicAdd(&position_sum[pid1].z, local_position_sum.z);
            atomicAdd(&cohesion_count[pid1], local_cohesion_count);
        }
        if (has_separation) {
            atomicAdd(&separation_sum[pid1].x, local_separation_sum.x);
            atomicAdd(&separation_sum[pid1].y, local_separation_sum.y);
            atomicAdd(&separation_sum[pid1].z, local_separation_sum.z);
        }
        if (local_alignment_count) {
            atomicAdd(&velocity_sum[pid1].x, local_velocity_sum.x);
            atomicAdd(&velocity_sum[pid1].y, local_velocity_sum.y);
            atomicAdd(&velocity_sum[pid1].z, local_velocity_sum.z);
            atomicAdd(&alignment_count[pid1], local_alignment_count);
        }
    }
}
#endif

__global__ void finalize_velocity(int num_boids, const glm::vec3 *points_lst,
    const glm::vec3 *vel1, glm::vec3 *vel2, const glm::vec3 *position_sum,
    const glm::vec3 *separation_sum, const glm::vec3 *velocity_sum,
    const int *cohesion_count, const int *alignment_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boids) return;
    glm::vec3 cohesion(0.0f), alignment(0.0f);
    if (cohesion_count[i])
        cohesion = (position_sum[i] / static_cast<float>(cohesion_count[i]) - points_lst[i]) * rule1Scale;
    if (alignment_count[i])
        alignment = (velocity_sum[i] / static_cast<float>(alignment_count[i])) * rule3Scale;
    glm::vec3 new_velocity = vel1[i] + cohesion + separation_sum[i] * rule2Scale + alignment;
    const float speed_squared = glm::dot(new_velocity, new_velocity);
    if (speed_squared > maxSpeed * maxSpeed)
        new_velocity *= maxSpeed * rsqrtf(speed_squared);
    vel2[i] = new_velocity;
}

__global__ void split_node(int num_active_leaves, int *active_leaves, GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, int *topology_changed, int *old_leaf_emit) {
    int leaf_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (leaf_idx >= num_active_leaves) {
        return;
    }

    GlintNode *node = &node_lst[active_leaves[leaf_idx]];
    const glm::vec3 child_size = (node->max_coords - node->min_coords) * 0.5f;
    if (node->isLeaf() && node->size > PARTITION_GRANULARITY_SPLIT &&
        child_size.x >= GLINT_MIN_CELL_SIZE && child_size.y >= GLINT_MIN_CELL_SIZE &&
        child_size.z >= GLINT_MIN_CELL_SIZE) {
        const glm::vec3 midpoint = node->min_coords +
            (node->max_coords - node->min_coords) * 0.5f;
        for (int i = 0; i < 8; i++) {
            int child_idx = pop(node_stk, node_stk_head, node_stk_tail, NODE_STK_MAX_SIZE);
            node->child_node_idx[i] = child_idx;
            GlintNode *child_node = &node_lst[child_idx];
            *child_node = GlintNode(0);

            child_node->min_coords = glm::vec3(
                (i & 1) ? midpoint.x : node->min_coords.x,
                (i & 2) ? midpoint.y : node->min_coords.y,
                (i & 4) ? midpoint.z : node->min_coords.z);
            child_node->max_coords = glm::vec3(
                (i & 1) ? node->max_coords.x : midpoint.x,
                (i & 2) ? node->max_coords.y : midpoint.y,
                (i & 4) ? node->max_coords.z : midpoint.z);
            
            if (node->part_idx != NO_PARTITION) {
                child_node->part_idx = node->part_idx;
                node->part_idx = NO_PARTITION;
                node->size = 0;
            } else {
                child_node->part_idx = pop(part_stk, part_stk_head, part_stk_tail, PART_STK_MAX_SIZE);
            }
        }

        old_leaf_emit[leaf_idx] = 8;
        atomicExch(topology_changed, 1);
    } else {
        old_leaf_emit[leaf_idx] = 1;
    }
}

__device__ void push(int *stk, int val, unsigned int *head,
    unsigned int *tail, unsigned int max_size) {
    unsigned int old_tail = atomicInc(tail, max_size - 1);
    unsigned int curr_tail = ++old_tail == max_size ? 0 : old_tail;
    if (curr_tail == *head) {
        // ERROR
        return;
    }

    stk[curr_tail] = val;
}

__device__ int pop(int *stk, unsigned int *head,
    unsigned int *tail, unsigned int max_size) {
    unsigned int old_head = atomicInc(head, max_size - 1);
    unsigned int new_head = old_head + 1 == max_size ? 0 : old_head + 1;
    if (new_head == *tail) {
        // ERROR
        return -1;
    }

    return stk[old_head];
}

__device__ void merge_at(int node_idx, GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, bool alloc_partition, int *topology_changed) {
    GlintNode *node = &node_lst[node_idx];
    if (node->size >= 0 && !node->isLeaf()) {
        int total_size = 0;
        for (int child : node->child_node_idx) {
            GlintNode *child_node = &node_lst[child];
            if (!child_node->isLeaf()) {
                return;
            }
            total_size += child_node->size;
        }

        if (total_size >= PARTITION_GRANULARITY_MERGE) {
            return;
        }

        node->size = total_size;
        if (topology_changed) atomicExch(topology_changed, 1);
        for (int i = 0; i < 8; i++) {
            int child = node->child_node_idx[i];
            node->child_node_idx[i] = -1;
            GlintNode *child_node = &node_lst[child];

            if (alloc_partition) {
                if (node->part_idx == NO_PARTITION) {
                    node->part_idx = child_node->part_idx;
                } else {
                    push(part_stk, child_node->part_idx, part_stk_head, part_stk_tail, PART_STK_MAX_SIZE);
                }
            }

            child_node->size = -1;
            push(node_stk, child, node_stk_head, node_stk_tail, NODE_STK_MAX_SIZE);
        }
    }
}

__global__ void merge_node(int num_potential_leaf_parents, const int *potential_leaf_parents, GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, bool alloc_partition, int *topology_changed) {
    const int parent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (parent_idx >= num_potential_leaf_parents) {
        return;
    }
    merge_at(potential_leaf_parents[parent_idx], node_lst, node_stk, node_stk_head, node_stk_tail, part_stk, part_stk_head, part_stk_tail, alloc_partition, topology_changed);
}

__global__ void merge_depth(int first_node, int node_count, GlintNode *node_lst, int *node_stk, unsigned int *node_stk_head, unsigned int *node_stk_tail, int *part_stk, unsigned int *part_stk_head, unsigned int *part_stk_tail, bool alloc_partition) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) merge_at(first_node + i, node_lst, node_stk, node_stk_head, node_stk_tail, part_stk, part_stk_head, part_stk_tail, alloc_partition, nullptr);
}

namespace {
void checkGlintCUDA(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("GLINT: ") + cudaGetErrorString(result));
}
}

void GlintOctree::init(int boid_count, const glm::vec3 *device_positions, cudaStream_t simulation_stream, int depth) {
    if (device_positions == nullptr || depth < 0)
        throw std::invalid_argument("GLINT: invalid positions or depth");
    int total = 1, width = 1;
    for (int d = 1; d <= depth; ++d) {
        if (width > (NODE_STK_MAX_SIZE - total) / 8)
            throw std::invalid_argument("GLINT: complete initial tree exceeds node pool");
        width *= 8;
        total += width;
    }
    if (width > PART_STK_MAX_SIZE)
        throw std::invalid_argument("GLINT: initial leaves exceed partition pool");

    deletion(simulation_stream);
    points_lst = device_positions;
    num_boids = boid_count;
    max_depth = depth;
    initial_node_count = total;
    const int threads = GLINT_THREADS_PER_BLOCK;
    const int node_blocks = (NODE_STK_MAX_SIZE + threads - 1) / threads;
    const int boid_blocks = (num_boids + threads - 1) / threads;
    auto allocate = [simulation_stream](auto *&pointer, size_t count) {
        checkGlintCUDA(cudaMallocAsync(reinterpret_cast<void **>(&pointer),
            count * sizeof(*pointer), simulation_stream));
    };
    try {
#if DEBUG_GLINT_PIPELINE_TIMINGS
        for (int i = 0; i < GLINT_TIMING_COUNT; ++i) {
            checkGlintCUDA(cudaEventCreate(&timing_start[i]));
            checkGlintCUDA(cudaEventCreate(&timing_end[i]));
        }
#endif
        allocate(node_lst, NODE_STK_MAX_SIZE);
        allocate(node_stk, NODE_STK_MAX_SIZE);
        allocate(part_lst, PART_STK_MAX_SIZE);
        allocate(part_stk, PART_STK_MAX_SIZE);
        allocate(filter_lst, static_cast<size_t>(num_boids));
        allocate(refine_lst, static_cast<size_t>(num_boids) * GLINT_REFINE_PAIRS_PER_BOID);
#if GLINT_COOPERATIVE_REFINEMENT
        allocate(refine_leaf_counts, PART_STK_MAX_SIZE);
        allocate(refine_leaf_offsets, PART_STK_MAX_SIZE);
        allocate(refine_grouped_pids, static_cast<size_t>(num_boids) * GLINT_REFINE_PAIRS_PER_BOID);
        checkGlintCUDA(cub::DeviceScan::ExclusiveSum(nullptr, refine_group_temp_bytes,
            refine_leaf_counts, refine_leaf_offsets, PART_STK_MAX_SIZE, simulation_stream));
        checkGlintCUDA(cudaMallocAsync(&refine_group_temp, refine_group_temp_bytes, simulation_stream));
#endif
        allocate(position_sum, num_boids);
        allocate(separation_sum, num_boids);
        allocate(velocity_sum, num_boids);
        allocate(cohesion_count, num_boids);
        allocate(alignment_count, num_boids);
        allocate(filter_lst_size, 1);
        allocate(refine_lst_size, 1);
        checkGlintCUDA(cudaMemsetAsync(filter_lst_size, 0, sizeof(int), simulation_stream));
        allocate(node_stk_head, 1);
        allocate(node_stk_tail, 1);
        allocate(part_stk_head, 1);
        allocate(part_stk_tail, 1);
        allocate(active_leaf_map, NODE_STK_MAX_SIZE);
        allocate(potential_leaf_parent_map, NODE_STK_MAX_SIZE);
        allocate(active_leaves, NODE_STK_MAX_SIZE);
        allocate(next_active_leaves, NODE_STK_MAX_SIZE);
        allocate(leafIsFirstChild, NODE_STK_MAX_SIZE);
        allocate(next_leafIsFirstChild, NODE_STK_MAX_SIZE);
        allocate(leaf_parents, NODE_STK_MAX_SIZE);
        allocate(next_leaf_parents, NODE_STK_MAX_SIZE);
        allocate(old_leaf_emit, NODE_STK_MAX_SIZE);
        allocate(old_leaf_offsets, NODE_STK_MAX_SIZE);
        allocate(potential_leaf_parents, NODE_STK_MAX_SIZE);
        allocate(next_potential_leaf_parents, NODE_STK_MAX_SIZE);
        allocate(num_active_leaves, 1);
        allocate(num_potential_leaf_parents, 1);
        allocate(next_num_active_leaves, 1);
        allocate(next_num_potential_leaf_parents, 1);
        allocate(topology_changed, 1);
        checkGlintCUDA(cudaMemsetAsync(next_num_active_leaves, 0, sizeof(int), simulation_stream));
        checkGlintCUDA(cudaMemsetAsync(next_num_potential_leaf_parents, 0, sizeof(int), simulation_stream));
        checkGlintCUDA(cudaMemsetAsync(topology_changed, 0, sizeof(int), simulation_stream));

        auto indices = thrust::make_counting_iterator(0);
        checkGlintCUDA(cub::DeviceSelect::Flagged(nullptr, compact_temp_bytes,
            indices, active_leaf_map, active_leaves, num_active_leaves, NODE_STK_MAX_SIZE, simulation_stream));
        size_t scan_temp_bytes = 0;
        auto output_counts = thrust::make_transform_iterator(indices,
            GetLeafOutputCounts{old_leaf_emit, leaf_parents, leafIsFirstChild});
        checkGlintCUDA(cub::DeviceScan::ExclusiveScan(nullptr, scan_temp_bytes,
            output_counts, old_leaf_offsets, AddLeafOutputCounts{},
            LeafOutputCounts{0, 0}, NODE_STK_MAX_SIZE, simulation_stream));
        if (scan_temp_bytes > compact_temp_bytes) compact_temp_bytes = scan_temp_bytes;
        checkGlintCUDA(cudaMallocAsync(&compact_temp, compact_temp_bytes, simulation_stream));

        const int pool_size = NODE_STK_MAX_SIZE > PART_STK_MAX_SIZE
            ? NODE_STK_MAX_SIZE : PART_STK_MAX_SIZE;
        init_pools<<<(pool_size + threads - 1) / threads, threads, 0, simulation_stream>>>(node_lst, node_stk, node_stk_head, node_stk_tail, part_stk, part_stk_head, part_stk_tail, initial_node_count);
        checkGlintCUDA(cudaGetLastError());
        build_init_octree<<<(total + threads - 1) / threads, threads, 0, simulation_stream>>>(total, depth, node_lst);
        checkGlintCUDA(cudaGetLastError());
        init_partition<<<boid_blocks, threads, 0, simulation_stream>>>(num_boids, points_lst, node_lst);
        checkGlintCUDA(cudaGetLastError());

        int level_width = width / 8;
        int level_start = total - width - level_width;
        for (int d = depth - 1; d >= 0; --d) {
            merge_depth<<<(level_width + threads - 1) / threads, threads, 0, simulation_stream>>>(
                level_start, level_width, node_lst, node_stk, node_stk_head, node_stk_tail, part_stk, part_stk_head, part_stk_tail, false);
            checkGlintCUDA(cudaGetLastError());
            level_width /= 8;
            level_start -= level_width;
        }
        init_leaves<<<node_blocks, threads, 0, simulation_stream>>>(NODE_STK_MAX_SIZE, node_lst, part_stk, part_stk_head, part_stk_tail, active_leaf_map, potential_leaf_parent_map);
        checkGlintCUDA(cudaGetLastError());
        checkGlintCUDA(cub::DeviceSelect::Flagged(compact_temp, compact_temp_bytes,
            indices, active_leaf_map, active_leaves, num_active_leaves, NODE_STK_MAX_SIZE, simulation_stream));
        checkGlintCUDA(cudaFreeAsync(active_leaf_map, simulation_stream));
        active_leaf_map = nullptr;
        checkGlintCUDA(cub::DeviceSelect::Flagged(compact_temp, compact_temp_bytes,
            indices, potential_leaf_parent_map, potential_leaf_parents,
            num_potential_leaf_parents, NODE_STK_MAX_SIZE, simulation_stream));
        checkGlintCUDA(cudaMemcpyAsync(&host_num_active_leaves, num_active_leaves,
            sizeof(host_num_active_leaves), cudaMemcpyDeviceToHost, simulation_stream));
        checkGlintCUDA(cudaMemcpyAsync(&host_num_potential_leaf_parents, num_potential_leaf_parents,
            sizeof(host_num_potential_leaf_parents), cudaMemcpyDeviceToHost, simulation_stream));
        checkGlintCUDA(cudaStreamSynchronize(simulation_stream));
        init_leaf_first_child<<<(host_num_active_leaves + threads - 1) / threads, threads, 0, simulation_stream>>>(
            host_num_active_leaves, active_leaves, node_lst, leafIsFirstChild, leaf_parents);
        checkGlintCUDA(cudaGetLastError());
        partition<<<boid_blocks, threads, 0, simulation_stream>>>(num_boids, points_lst, node_lst, part_lst,
            filter_lst, filter_lst_size, refine_lst, refine_lst_size);
        checkGlintCUDA(cudaGetLastError());
        checkGlintCUDA(cudaMemcpyAsync(&host_filter_lst_size, filter_lst_size,
            sizeof(host_filter_lst_size), cudaMemcpyDeviceToHost, simulation_stream));
        checkGlintCUDA(cudaMemcpyAsync(&host_refine_lst_size, refine_lst_size,
            sizeof(host_refine_lst_size), cudaMemcpyDeviceToHost, simulation_stream));
        checkGlintCUDA(cudaStreamSynchronize(simulation_stream));
#if VISUALIZE
        ++topology_revision;
#endif
    } catch (...) {
        deletion(simulation_stream);
        throw;
    }
}

void GlintOctree::deletion(cudaStream_t simulation_stream) {
    auto release = [simulation_stream](auto *&pointer) {
        if (pointer) cudaFreeAsync(pointer, simulation_stream);
        pointer = nullptr;
    };
    points_lst = nullptr;
    release(node_lst);
    release(node_stk);
    release(part_lst);
    release(part_stk);
    release(filter_lst);
    release(refine_lst);
#if GLINT_COOPERATIVE_REFINEMENT
    release(refine_leaf_counts);
    release(refine_leaf_offsets);
    release(refine_grouped_pids);
    release(refine_group_temp);
    refine_group_temp_bytes = 0;
#endif
    release(position_sum);
    release(separation_sum);
    release(velocity_sum);
    release(cohesion_count);
    release(alignment_count);
    release(filter_lst_size);
    release(refine_lst_size);
    release(node_stk_head);
    release(node_stk_tail);
    release(part_stk_head);
    release(part_stk_tail);
    release(active_leaf_map);
    release(potential_leaf_parent_map);
    release(active_leaves);
    release(next_active_leaves);
    release(leafIsFirstChild);
    release(next_leafIsFirstChild);
    release(leaf_parents);
    release(next_leaf_parents);
    release(old_leaf_emit);
    release(old_leaf_offsets);
    release(potential_leaf_parents);
    release(next_potential_leaf_parents);
    release(num_active_leaves);
    release(num_potential_leaf_parents);
    release(next_num_active_leaves);
    release(next_num_potential_leaf_parents);
    release(topology_changed);
    release(compact_temp);
    cudaStreamSynchronize(simulation_stream);
#if DEBUG_GLINT_PIPELINE_TIMINGS
    for (int i = 0; i < GLINT_TIMING_COUNT; ++i) {
        if (timing_start[i]) cudaEventDestroy(timing_start[i]);
        if (timing_end[i]) cudaEventDestroy(timing_end[i]);
        timing_start[i] = timing_end[i] = nullptr;
        timing_recorded[i] = false;
        timing_totals[i] = 0.0;
    }
#endif
#if DEBUG_GLINT_PIPELINE_TIMINGS || DEBUG_GLINT_TREE_UPDATES
    debug_steps = debug_tree_updates = debug_tree_unchanged = 0;
    debug_step_active = false;
#endif
    compact_temp_bytes = 0;
    num_boids = max_depth = initial_node_count = 0;
    host_num_active_leaves = host_num_potential_leaf_parents = 0;
    host_topology_changed = 0;
    host_filter_lst_size = host_refine_lst_size = 0;
}

void GlintOctree::update_partition(const glm::vec3 *dev_positions, cudaStream_t simulation_stream) {
    points_lst = dev_positions;

    const int threads = GLINT_THREADS_PER_BLOCK;
    const int leaf_blocks = (host_num_active_leaves + threads - 1) / threads;
    GLINT_TIME_BEGIN(LEAF_RESET);
    reset_leaf_count<<<leaf_blocks, threads, 0, simulation_stream>>>(host_num_active_leaves, active_leaves, node_lst);
    GLINT_TIME_END(LEAF_RESET);
    checkGlintCUDA(cudaGetLastError());

    GLINT_TIME_BEGIN(FILTER_RESET);
    checkGlintCUDA(cudaMemsetAsync(filter_lst_size, 0, sizeof(int), simulation_stream));
    GLINT_TIME_END(FILTER_RESET);
    GLINT_TIME_BEGIN(PARTITION);
    partition<<<(num_boids + threads - 1) / threads, threads, 0, simulation_stream>>>(num_boids,
        points_lst, node_lst, part_lst, filter_lst, filter_lst_size, refine_lst, refine_lst_size);
    GLINT_TIME_END(PARTITION);
    checkGlintCUDA(cudaGetLastError());
}

void GlintOctree::update_filter(cudaStream_t simulation_stream) {
    const int threads = GLINT_THREADS_PER_BLOCK;
    GLINT_TIME_BEGIN(FILTER_COUNT);
    checkGlintCUDA(cudaMemcpyAsync(&host_filter_lst_size, filter_lst_size,
        sizeof(host_filter_lst_size), cudaMemcpyDeviceToHost, simulation_stream));
    GLINT_TIME_END(FILTER_COUNT);
    checkGlintCUDA(cudaStreamSynchronize(simulation_stream));
    if (host_filter_lst_size > 0) {
        const int filter_blocks = (host_filter_lst_size + threads - 1) / threads;
        const size_t shared_bytes = GLINT_FILTER_STACK_CAPACITY * sizeof(FilterEntry);
        GLINT_TIME_BEGIN(FILTER);
        filter<<<filter_blocks, threads, shared_bytes, simulation_stream>>>(host_filter_lst_size,
            filter_lst, points_lst, node_lst, refine_lst, refine_lst_size);
        GLINT_TIME_END(FILTER);
        checkGlintCUDA(cudaGetLastError());
    }
    GLINT_TIME_BEGIN(REFINE_COUNT);
    checkGlintCUDA(cudaMemcpyAsync(&host_refine_lst_size, refine_lst_size,
        sizeof(host_refine_lst_size), cudaMemcpyDeviceToHost, simulation_stream));
    GLINT_TIME_END(REFINE_COUNT);
    checkGlintCUDA(cudaStreamSynchronize(simulation_stream));
}

void GlintOctree::update_velocity(const glm::vec3 *dev_vel1, glm::vec3 *dev_vel2, cudaStream_t simulation_stream) {
    const int threads = GLINT_THREADS_PER_BLOCK;
    const int boid_blocks = (num_boids + threads - 1) / threads;
    GLINT_TIME_BEGIN(ACCUM_RESET);
    reset_interaction_accumulators<<<boid_blocks, threads, 0, simulation_stream>>>(num_boids,
        position_sum, separation_sum, velocity_sum, cohesion_count, alignment_count);
    GLINT_TIME_END(ACCUM_RESET);
    checkGlintCUDA(cudaGetLastError());
    if (host_refine_lst_size > 0) {
        const int refinement_blocks = (host_refine_lst_size + threads - 1) / threads;
#if GLINT_COOPERATIVE_REFINEMENT
        GLINT_TIME_BEGIN(REFINE_GROUP_COUNT);
        checkGlintCUDA(cudaMemsetAsync(refine_leaf_counts, 0,
            PART_STK_MAX_SIZE * sizeof(int), simulation_stream));
        count_refinement_tasks<<<refinement_blocks, threads, 0, simulation_stream>>>(
            host_refine_lst_size, refine_lst, refine_leaf_counts);
        GLINT_TIME_END(REFINE_GROUP_COUNT);
        checkGlintCUDA(cudaGetLastError());

        GLINT_TIME_BEGIN(REFINE_GROUP_SCAN);
        checkGlintCUDA(cub::DeviceScan::ExclusiveSum(refine_group_temp, refine_group_temp_bytes,
            refine_leaf_counts, refine_leaf_offsets, PART_STK_MAX_SIZE, simulation_stream));
        GLINT_TIME_END(REFINE_GROUP_SCAN);

        GLINT_TIME_BEGIN(REFINE_GROUP_SCATTER);
        checkGlintCUDA(cudaMemsetAsync(refine_leaf_counts, 0,
            PART_STK_MAX_SIZE * sizeof(int), simulation_stream));
        scatter_refinement_tasks<<<refinement_blocks, threads, 0, simulation_stream>>>(
            host_refine_lst_size, refine_lst, refine_leaf_offsets, refine_leaf_counts, refine_grouped_pids);
        GLINT_TIME_END(REFINE_GROUP_SCATTER);
        checkGlintCUDA(cudaGetLastError());

        GLINT_TIME_BEGIN(REFINEMENT);
        refinement_cooperative<<<host_num_active_leaves, threads, 0, simulation_stream>>>(
            host_num_active_leaves, active_leaves, node_lst, part_lst,
            refine_leaf_counts, refine_leaf_offsets, refine_grouped_pids, points_lst, dev_vel1,
            position_sum, separation_sum, velocity_sum, cohesion_count, alignment_count);
#else
        GLINT_TIME_BEGIN(REFINEMENT);
        refinement<<<refinement_blocks, threads, 0, simulation_stream>>>(host_refine_lst_size,
            refine_lst, points_lst, part_lst, dev_vel1,
            position_sum, separation_sum, velocity_sum, cohesion_count, alignment_count);
#endif
        GLINT_TIME_END(REFINEMENT);
        checkGlintCUDA(cudaGetLastError());
    }
    GLINT_TIME_BEGIN(FINALIZE);
    finalize_velocity<<<boid_blocks, threads, 0, simulation_stream>>>(num_boids, points_lst, dev_vel1, dev_vel2,
        position_sum, separation_sum, velocity_sum, cohesion_count, alignment_count);
    GLINT_TIME_END(FINALIZE);
    checkGlintCUDA(cudaGetLastError());
}

void GlintOctree::update_tree(cudaStream_t simulation_stream) {
    const int threads = GLINT_THREADS_PER_BLOCK;
    const int leaf_blocks = (host_num_active_leaves + threads - 1) / threads;
    const int parent_blocks = (host_num_potential_leaf_parents + threads - 1) / threads;
    GLINT_TIME_BEGIN(TREE_RESET);
    checkGlintCUDA(cudaMemsetAsync(topology_changed, 0, sizeof(int), simulation_stream));
    GLINT_TIME_END(TREE_RESET);
    GLINT_TIME_BEGIN(SPLIT);
    split_node<<<leaf_blocks, threads, 0, simulation_stream>>>(host_num_active_leaves, active_leaves, node_lst, node_stk, node_stk_head, node_stk_tail, part_stk, part_stk_head, part_stk_tail, topology_changed, old_leaf_emit);
    GLINT_TIME_END(SPLIT);
    checkGlintCUDA(cudaGetLastError());
    if (host_num_potential_leaf_parents > 0) {
        GLINT_TIME_BEGIN(MERGE);
        merge_node<<<parent_blocks, threads, 0, simulation_stream>>>(host_num_potential_leaf_parents, potential_leaf_parents, node_lst, node_stk, node_stk_head, node_stk_tail, part_stk, part_stk_head, part_stk_tail, true, topology_changed);
        GLINT_TIME_END(MERGE);
        checkGlintCUDA(cudaGetLastError());
    }

    GLINT_TIME_BEGIN(TREE_FLAG);
    checkGlintCUDA(cudaMemcpyAsync(&host_topology_changed, topology_changed,
        sizeof(host_topology_changed), cudaMemcpyDeviceToHost, simulation_stream));
    GLINT_TIME_END(TREE_FLAG);
    checkGlintCUDA(cudaStreamSynchronize(simulation_stream));
    if (!host_topology_changed) return;

    GLINT_TIME_BEGIN(MERGE_EVENTS);
    update_leaf_merge_events<<<leaf_blocks, threads, 0, simulation_stream>>>(host_num_active_leaves,
        active_leaves, node_lst, leaf_parents, leafIsFirstChild, old_leaf_emit);
    GLINT_TIME_END(MERGE_EVENTS);
    checkGlintCUDA(cudaGetLastError());
    // Generate both output counts on demand and scan them together over old leaves.
    auto output_counts = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
        GetLeafOutputCounts{old_leaf_emit, leaf_parents, leafIsFirstChild});
    GLINT_TIME_BEGIN(SCAN);
    checkGlintCUDA(cub::DeviceScan::ExclusiveScan(compact_temp, compact_temp_bytes,
        output_counts, old_leaf_offsets, AddLeafOutputCounts{},
        LeafOutputCounts{0, 0}, host_num_active_leaves, simulation_stream));
    GLINT_TIME_END(SCAN);
    GLINT_TIME_BEGIN(SCATTER);
    update_active_leaf_events<<<leaf_blocks, threads, 0, simulation_stream>>>(host_num_active_leaves,
        active_leaves, leaf_parents, leafIsFirstChild, node_lst,
        next_active_leaves, next_leaf_parents, next_leafIsFirstChild,
        next_num_active_leaves, old_leaf_offsets, old_leaf_emit,
        next_potential_leaf_parents, next_num_potential_leaf_parents);
    GLINT_TIME_END(SCATTER);
    checkGlintCUDA(cudaGetLastError());

    GLINT_TIME_BEGIN(LEAF_COUNTS);
    checkGlintCUDA(cudaMemcpyAsync(&host_num_active_leaves, next_num_active_leaves,
        sizeof(host_num_active_leaves), cudaMemcpyDeviceToHost, simulation_stream));
    checkGlintCUDA(cudaMemcpyAsync(&host_num_potential_leaf_parents, next_num_potential_leaf_parents,
        sizeof(host_num_potential_leaf_parents), cudaMemcpyDeviceToHost, simulation_stream));
    GLINT_TIME_END(LEAF_COUNTS);
    checkGlintCUDA(cudaStreamSynchronize(simulation_stream));
    std::swap(active_leaves, next_active_leaves);
    std::swap(leaf_parents, next_leaf_parents);
    std::swap(leafIsFirstChild, next_leafIsFirstChild);
    std::swap(potential_leaf_parents, next_potential_leaf_parents);
    std::swap(num_active_leaves, next_num_active_leaves);
    std::swap(num_potential_leaf_parents, next_num_potential_leaf_parents);
#if VISUALIZE
    ++topology_revision;
#endif
}

#if DEBUG_GLINT_PIPELINE_TIMINGS || DEBUG_GLINT_TREE_UPDATES
void GlintOctree::begin_debug_step() {
    debug_step_active = true;
#if DEBUG_GLINT_PIPELINE_TIMINGS
    for (bool &recorded : timing_recorded) recorded = false;
#endif
}

void GlintOctree::finish_debug_step() {
    if (!debug_step_active) return;
    ++debug_steps;
    if (host_topology_changed) ++debug_tree_updates;
    else ++debug_tree_unchanged;
#if DEBUG_GLINT_PIPELINE_TIMINGS
    for (int i = 0; i < GLINT_TIMING_COUNT; ++i) {
        if (!timing_recorded[i]) continue;
        float milliseconds = 0.0f;
        checkGlintCUDA(cudaEventElapsedTime(&milliseconds, timing_start[i], timing_end[i]));
        timing_totals[i] += milliseconds;
    }
#endif
    debug_step_active = false;
}

void GlintOctree::print_debug_stats() {
    if (!debug_steps) return;
    std::ostringstream output;
    output.setf(std::ios::fixed);
    output.precision(3);
#if DEBUG_GLINT_PIPELINE_TIMINGS
    static constexpr const char *names[] = {
        "leaf_reset", "filter_reset", "partition", "filter_count", "filter", "refine_count",
        "accum_reset", "refine_group_count", "refine_group_scan", "refine_group_scatter", "refinement", "finalize", "flag_reset", "split", "merge", "flag_read",
        "merge_events", "scan", "scatter", "leaf_counts", "move",
        "copy", "scan", "leaves"
    };
    static_assert(sizeof(names) / sizeof(names[0]) == GLINT_TIMING_COUNT);
    auto print_stages = [&](int first, int last) {
        for (int i = first; i <= last; ++i)
            output << " | " << names[i] << " " << timing_totals[i] / debug_steps;
    };
    double total = 0.0;
    for (double value : timing_totals) total += value;
    output << "GLINT GPU ms/step (" << debug_steps << " steps)";
    print_stages(GLINT_TIMING_LEAF_RESET, GLINT_TIMING_FINALIZE);
    output << " | move " << timing_totals[GLINT_TIMING_MOVE] / debug_steps
        << " | sum " << total / debug_steps << "\nGLINT tree GPU ms/step";
    print_stages(GLINT_TIMING_TREE_RESET, GLINT_TIMING_LEAF_COUNTS);
    output << "\nGLINT snapshot GPU ms/step";
    print_stages(GLINT_TIMING_SNAPSHOT_COPY, GLINT_TIMING_SNAPSHOT_LEAVES);
    output << "\n";
    for (double &value : timing_totals) value = 0.0;
#endif
#if DEBUG_GLINT_TREE_UPDATES
    output << "GLINT tree updates | with updates " << debug_tree_updates
        << " | no updates " << debug_tree_unchanged << " | steps " << debug_steps << "\n";
#endif
    printf("%s", output.str().c_str());
    debug_steps = debug_tree_updates = debug_tree_unchanged = 0;
}
#endif

#if DEBUG_GLINT_PIPELINE_TIMINGS
void GlintOctree::begin_debug_stage(TimingStage stage, cudaStream_t simulation_stream) {
    if (!debug_step_active) return;
    checkGlintCUDA(cudaEventRecord(timing_start[stage], simulation_stream));
    timing_recorded[stage] = true;
}

void GlintOctree::end_debug_stage(TimingStage stage, cudaStream_t simulation_stream) {
    if (!debug_step_active) return;
    checkGlintCUDA(cudaEventRecord(timing_end[stage], simulation_stream));
}
#endif
