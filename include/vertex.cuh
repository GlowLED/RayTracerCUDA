#pragma once
#include "vec3.cuh"

struct vertex {
	vec3 p;
    vec3 n;
    float u, v;
    __host__ __device__ vertex() {}
    __host__ __device__ vertex(vec3 p, vec3 n, float u, float v) {
        this->p = p;
        this->n = n;
        this->u = u;
        this->v = v;
    }
};
