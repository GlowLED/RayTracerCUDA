#pragma once
#include "vertex.cuh"
#include "vec3.cuh"
#include <vector>
#include <cstdint>
class Model
{
public:
    std::vector<vertex> vertex_buffer;
    std::vector<unsigned int> index_buffer;
    __host__ Model() {}
    __host__ Model(std::vector<vertex> vertex_buffer, std::vector<unsigned int> index_buffer) {
        this->vertex_buffer = vertex_buffer;
        this->index_buffer = index_buffer;
    }
    __host__ Model(std::vector<vertex> vertex_buffer, std::vector<int> index_buffer)
    {
        this->vertex_buffer = vertex_buffer;
        // Convert index_buffer from int to unsigned int
        this->index_buffer.resize(index_buffer.size());
        for (size_t i = 0; i < index_buffer.size(); ++i) {
            this->index_buffer[i] = static_cast<unsigned int>(index_buffer[i]);
        }
    }
    __host__ unsigned int getVertexCount() const { return vertex_buffer.size(); }
    __host__ unsigned int getIndexCount() const { return index_buffer.size(); }
    __host__ unsigned int getTriangleCount() const { return index_buffer.size() / 3; }
};
