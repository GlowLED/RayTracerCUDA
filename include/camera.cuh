#pragma once
#include "vec3.cuh"
#include "ray.cuh"

class Camera
{
public:
    __host__ Camera() {}
    __host__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio);
    __device__ ray get_ray(float s, float t) const;
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    
};