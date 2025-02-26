#pragma once
#include "vec3.cuh"
#include "ray.cuh"
struct bvh_node {
    vec3 pmin, pmax;
    bvh_node *left, *right;
    int axis;
    unsigned int start, end;
    __host__ __device__ bvh_node() {}
    __host__ __device__ bvh_node(vec3 pmin, vec3 pmax, bvh_node *left, bvh_node *right, unsigned int start, unsigned int end, int axis) {
        this->pmin = pmin;
        this->pmax = pmax;
        this->left = left;
        this->right = right;
        this->start = start;
        this->end = end;
        this->axis = axis;
    }
};
struct Scene;
__host__ bvh_node* build_bvh(Scene& scene, unsigned int start, unsigned int end, unsigned int& size, int min_triangles = 1);

struct FlattenBVHNode {
    vec3 pmin, pmax;
    unsigned int left_idx, right_idx;
    unsigned int start, end;
    bool is_leaf;
    int axis;
    __device__ bool hit(const ray& r, float tmin, float &tmax) const;
};

__host__ void bvhFlatten(bvh_node* root, FlattenBVHNode* fla_bvh);
