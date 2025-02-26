#pragma once
#include "model.cuh"
#include "vec3.cuh"
#include "vertex.cuh"
#include "bvh.cuh"
#include <vector>

class Scene: public Model {
public:
    std::vector<Model> models;
    std::vector<vertex> vertex_buffer;
    std::vector<unsigned int> index_buffer;
    std::vector<unsigned int> offsets;
    vertex* device_vertex_buffer;
    unsigned int* device_index_buffer;
    unsigned int* device_offsets;
    bvh_node* root;
    FlattenBVHNode* device_fla_bvh;
    unsigned int bvh_size = 0;
    __host__ Scene() {}
    __host__ Scene(std::vector<Model> models) {
        this->models = models;
    }
    __host__ void addModel(Model model) {
        models.push_back(model);
    }
    __host__ void deleteModel(int index) {
        models.erase(models.begin() + index);
    }
    __host__ void prepare_buffers();

    __host__ size_t getTriangleCount() const {
        return index_buffer.size() / 3;
    }
    
    __host__ bvh_node* buildBVH(int min_triangles = 1) {
        bvh_size = 0;
        root = build_bvh(*this, 0, index_buffer.size()/3, bvh_size);
        return root;
    }
    
};
