#pragma once
#include "vec3.cuh"
#include "ray.cuh"
#include "vertex.cuh"
struct hit_record
{
    vec3 p;
    vec3 n;
    float t;
    float u, v;
};
#define EPSILON 1e-6
__device__ bool triangle_hit(const vertex &v0, const vertex &v1, const vertex &v2, 
                             const ray &r, float t_min, float t_max, hit_record &rec)
{   
    vec3 e1 = v1.p - v0.p;
    vec3 e2 = v2.p - v0.p;
    vec3 p = cross(r.dir, e2);
    float det = dot(e1, p);
    if (det > -EPSILON && det < EPSILON) return false;
    float inv_det = 1.0f / det;
    vec3 t = r.orig - v0.p;
    float u = dot(t, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    vec3 q = cross(t, e1);
    float v = dot(r.dir, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t_hit = dot(e2, q) * inv_det;
    if (t_hit < t_min || t_hit > t_max) return false;
    rec.t = t_hit;
    rec.p = r.point_at_parameter(t_hit);
    rec.n = unit_vector(cross(e1, e2));
    
    float w = 1.0f - u - v;
    rec.u = w * v0.u + u * v1.u + v * v2.u;
    rec.v = w * v0.v + u * v1.v + v * v2.v;


    return true;
}