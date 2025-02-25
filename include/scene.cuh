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
    __host__ void prepare_buffers() {
        vertex_buffer.clear();
        index_buffer.clear();
        offsets.push_back(0);  // 第一个模型的偏移量为0

        for (const Model& model : models) {
            // 添加顶点
            vertex_buffer.insert(vertex_buffer.end(), model.vertex_buffer.begin(), model.vertex_buffer.end());
            
            // 调整点索引（考虑顶偏移）
            unsigned int offset = offsets.back();
            for (int index : model.index_buffer) {
                index_buffer.push_back(index + offset);
            }
            
            // 记录下一个模型的顶点偏移量
            offsets.push_back(offset + model.vertex_buffer.size());
        }
    }


    __host__ size_t getTriangleCount() const {
        return index_buffer.size() / 3;
    }
    
    __host__ bvh_node* buildBVH(int min_triangles = 1) {
        bvh_size = 0;
        return build_bvh(*this, 0, index_buffer.size()/3, bvh_size);
    }
    
};
