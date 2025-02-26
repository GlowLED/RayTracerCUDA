#include "../include/scene.cuh"
__host__ void Scene::prepare_buffers() {
    offsets.clear();
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