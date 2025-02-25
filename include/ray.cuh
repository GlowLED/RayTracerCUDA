#pragma once
#include "vec3.cuh"
struct ray{
    vec3 orig;
    vec3 dir;
    __host__ __device__ ray() {}
    __host__ __device__ ray(vec3 o, vec3 d) {
        orig = o;
        dir = d;
    }
    __host__ __device__ __forceinline__ vec3 point_at_parameter(const float t) const{
        return orig + t * dir;
    }
    __host__ __device__ __forceinline__ vec3 origin() const {
        return orig;
    }
    __host__ __device__ __forceinline__ vec3 direction() const {
        return dir;
    }
};
