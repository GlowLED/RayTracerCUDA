#include "../include/render.cuh"
#include "../include/vertex.cuh"
#include "../include/scene.cuh"
#include "../include/bvh.cuh"
#include "../include/vec3.cuh"
#include "../include/ray.cuh"
#include "../include/camera.cuh"
#include "../include/material.cuh"
#include "../include/cudastack.cuh"
#include "../include/hittable.cuh"
#include "../include/material.cuh"
#include <curand.h>
#include <curand_kernel.h>

// 定义tile大小常量



#define TILE_SIZE 16
__global__ void render(vertex* vertex_buffer, unsigned int* index_buffer, FlattenBVHNode* bvh_nodes,
    Camera &cam, int width, int height, int samples_per_pixel, int max_depth, 
    curandState* rand_states)
{
   // 计算tile的起始位置
   int tile_x = blockIdx.x * TILE_SIZE;
   int tile_y = blockIdx.y * TILE_SIZE;
   
   // 当前线程在tile内的位置
   int local_x = threadIdx.x;
   int local_y = threadIdx.y;
   
   // 计算全局坐标
   int i = tile_x + local_x;
   int j = tile_y + local_y;

   if ((i >= width) || (j >= height)) return;

   int index = j * width + i;
   curandState rand_state = rand_states[index];
   vec3 col(0, 0, 0);

   for (int s = 0; s < samples_per_pixel; s++)
   {
       float u = (i + curand_uniform(&rand_state)) / width;
       float v = (j + curand_uniform(&rand_state)) / height;
       ray r = cam.get_ray(u, v);
       col += color(r, bvh_nodes, vertex_buffer, index_buffer, max_depth, &rand_state);
   }
}

__device__ vec3 color(const ray& r, FlattenBVHNode* bvh_nodes, vertex* vertex_buffer, unsigned int* index_buffer,
     int max_depth, curandState* rand_state)
{
    vec3 result(0.0f, 0.0f, 0.0f);
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    ray cur_ray = r;

    CudaStack<FlattenBVHNode, 1024> nodes_stack;
    nodes_stack.push(bvh_nodes[0]);

    for (int depth = 0;depth < max_depth; depth++) {
        hit_record rec;
        float closest_so_far = FLT_MAX;
        bool hit_anything = false;
        
        nodes_stack.push(bvh_nodes[0]); // Reset stack for each bounce

        
        while (!nodes_stack.empty()) {
            FlattenBVHNode node = nodes_stack.pop();
            float temp_t_max = closest_so_far;
            if (node.hit(cur_ray, 0.001f, temp_t_max)) {
                closest_so_far = temp_t_max;
                if (node.is_leaf) {
                    for (int i = node.start; i < node.end; i++) {
                        unsigned int idx0 = index_buffer[i * 3];
                        unsigned int idx1 = index_buffer[i * 3 + 1];
                        unsigned int idx2 = index_buffer[i * 3 + 2];
                        vertex v0 = vertex_buffer[idx0];
                        vertex v1 = vertex_buffer[idx1];
                        vertex v2 = vertex_buffer[idx2];
                        if (triangle_hit(v0, v1, v2, cur_ray, 0.001f, closest_so_far, rec)) {
                            closest_so_far = rec.t;
                            hit_anything = true;
                        }
                    }
                } else {
                    nodes_stack.push(bvh_nodes[node.left_idx]);
                    nodes_stack.push(bvh_nodes[node.right_idx]);
                }
            }
        }
        
        if (!hit_anything) {
            // No hit, return background color
            return vec3(0.0f, 0.0f, 0.0f); // Or any background color logic
        }
        
        // Process material and update ray for next bounce
        // This is placeholder code that needs to be replaced with actual material handling
        vec3 target = rec.p + rec.n;
        cur_ray = ray(rec.p, target - rec.p);
        attenuation *= 0.5f; // Simple attenuation factor
    }
    
    return result; // Return accumulated color
}