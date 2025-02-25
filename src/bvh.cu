#include "../include/bvh.cuh"
#include "../include/scene.cuh"
#include "../include/vec3.cuh"
#include <algorithm>
#include <queue>
#include <unordered_map>
__host__ bvh_node* build_bvh(Scene& scene, unsigned int start, unsigned int end, unsigned int& size, int min_triangles) {
    /*
    Attention: this function is not optimized for the GPU, it is meant to be run on the CPU.
    And it will change the order of the triangles in the index buffer.
    */
    vec3 p0, p1, p2;
    vec3 pmin(1e9, 1e9, 1e9), pmax(-1e9, -1e9, -1e9);
    for (unsigned int i = 0;i < end - start;i++) {
        p0 = scene.vertex_buffer[scene.index_buffer[3 * i + 0]].p;
        p1 = scene.vertex_buffer[scene.index_buffer[3 * i + 1]].p;
        p2 = scene.vertex_buffer[scene.index_buffer[3 * i + 2]].p;
        pmin = vec3min(pmin, vec3min(p0, vec3min(p1, p2)));
        pmax = vec3max(pmax, vec3max(p0, vec3max(p1, p2)));
    }
    
    vec3 d = pmax - pmin;
    int axis = d.maxElementAxis();
    if (end - start <= min_triangles || d[axis] < 1e-4) {
        size++;
        return new bvh_node(pmin, pmax, nullptr, nullptr, start, end, axis);
    }
    float mid = 0.5f * (pmin[axis] + pmax[axis]);
    unsigned int mid_index = start;
    for (unsigned int i = start;i < end;i++) {
        vec3 p0 = scene.vertex_buffer[scene.index_buffer[3 * i + 0]].p;
        vec3 p1 = scene.vertex_buffer[scene.index_buffer[3 * i + 1]].p;
        vec3 p2 = scene.vertex_buffer[scene.index_buffer[3 * i + 2]].p;
        vec3 c = 1.0f / 3.0f * (p0 + p1 + p2);
        // if the center of the triangle is on the left side of the split
        if (c[axis] < mid) {
            std::swap(scene.index_buffer[3 * i + 0], scene.index_buffer[3 * mid_index + 0]);
            std::swap(scene.index_buffer[3 * i + 1], scene.index_buffer[3 * mid_index + 1]);
            std::swap(scene.index_buffer[3 * i + 2], scene.index_buffer[3 * mid_index + 2]);
            mid_index++;
            // sort the triangles so that the ones on the left side of the split are on the left side of the array
        }
    }
    if (mid_index == start || mid_index == end) {
        mid_index = start + (end - start) / 2;
    }// if all the triangles are on one side of the split, just split in the middle
    
    bvh_node* left = build_bvh(scene, start, mid_index, size, min_triangles);
    bvh_node* right = build_bvh(scene, mid_index, end, size, min_triangles);

    size++;
    return new bvh_node(pmin, pmax, left, right, start, end, axis);

}
__host__ void bvhFlatten(bvh_node* root, DeviceBVHNode* device_bvh) {
    if (!root) return ;

    std::queue<bvh_node*> queue;
    std::unordered_map<bvh_node*, int> node_indices;
        
    queue.push(root);
    int node_count = 0;
        
    while (!queue.empty()) {
        bvh_node* node = queue.front();
        queue.pop();
        
        int current_index = node_count++;
        node_indices[node] = current_index;
            
        device_bvh[current_index].pmin = node->pmin;
        device_bvh[current_index].pmax = node->pmax;
        device_bvh[current_index].start = node->start;
        device_bvh[current_index].end = node->end;
        device_bvh[current_index].axis = node->axis;
        device_bvh[current_index].is_leaf = (node->left == nullptr && node->right == nullptr);
            
        device_bvh[current_index].left_idx = UINT_MAX;
        device_bvh[current_index].right_idx = UINT_MAX;
            
        if (node->left) queue.push(node->left);
        if (node->right) queue.push(node->right);
    }
        
    for (const auto& pair : node_indices) {
        bvh_node* node = pair.first;
        int node_idx = pair.second;
            
        if (node->left) {
            device_bvh[node_idx].left_idx = node_indices[node->left];
        }
            
        if (node->right) {
            device_bvh[node_idx].right_idx = node_indices[node->right];
        }
    }
}