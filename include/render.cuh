#pragma once
#include "vec3.cuh"
#include "vertex.cuh"
#include "ray.cuh"
#include "bvh.cuh"
#include "scene.cuh"
#include "camera.cuh"
#include <curand.h>
#include <curand_kernel.h>

__device__ vec3 color(const ray& r, FlattenBVHNode* bvh_nodes, vertex* vertex_buffer,
      unsigned int* index_buffer, int max_depth, curandState* rand_state);

__global__ void render(vertex* vertex_buffer, unsigned int* index_buffer, FlattenBVHNode* root,
     Camera &cam, int width, int height, int samples_per_pixel, int max_depth, 
     curandState* rand_states);