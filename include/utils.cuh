#pragma once
#include "vec3.cuh"
#include "vertex.cuh"
#include "bvh.cuh"
#include "scene.cuh"
#include <stdio.h>
#include <vector>

__host__ void cpySceneToDevice(Scene& scene) {
    printf("Copying BVH to device\n");
    FlattenBVHNode* host_fla_bvh = new FlattenBVHNode[scene.bvh_size];
    bvhFlatten(scene.root, host_fla_bvh);
    FlattenBVHNode* device_fla_bvh;
    cudaMalloc(&device_fla_bvh, scene.bvh_size * sizeof(FlattenBVHNode));
    cudaMemcpy(device_fla_bvh, host_fla_bvh, scene.bvh_size * sizeof(FlattenBVHNode), cudaMemcpyHostToDevice);
    scene.device_fla_bvh = device_fla_bvh;
    delete[] host_fla_bvh;
    printf("Done\n");
    printf("Copying vertex buffer and index buffer to device\n");
    vertex* device_vertex_buffer;
    unsigned int* device_index_buffer;
    unsigned int* device_offsets;
    cudaMalloc(&device_vertex_buffer, scene.vertex_buffer.size() * sizeof(vertex));
    cudaMalloc(&device_index_buffer, scene.index_buffer.size() * sizeof(unsigned int));
    cudaMalloc(&device_offsets, scene.offsets.size() * sizeof(unsigned int));
    cudaMemcpy(device_vertex_buffer, scene.vertex_buffer.data(), scene.vertex_buffer.size() * sizeof(vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(device_index_buffer, scene.index_buffer.data(), scene.index_buffer.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_offsets, scene.offsets.data(), scene.offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    scene.device_vertex_buffer = device_vertex_buffer;
    scene.device_index_buffer = device_index_buffer;
    scene.device_offsets = device_offsets;
    printf("Done\n");
}