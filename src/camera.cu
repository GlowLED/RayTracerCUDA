#include "../include/camera.cuh"
#include "../include/vec3.cuh"
#include "../include/ray.cuh"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
__host__ Camera::Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio)
    {
        float theta = vfov * M_PI / 180;
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
    }
 __device__ ray Camera::get_ray(float s, float t) const
{
    return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
}