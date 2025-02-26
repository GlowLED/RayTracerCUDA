#pragma once
#include "vec3.cuh"
#include "ray.cuh"
struct hit_record;
struct material {
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
};
