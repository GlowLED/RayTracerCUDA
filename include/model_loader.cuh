#pragma once
#include "vec3.cuh"
#include "vertex.cuh"
#include "model.cuh"
#include <fstream>
#include <sstream>
#include <string>

class ModelLoader
{
public:
    std::string path;
    __host__ ModelLoader(std::string load_path) {
        this->path = load_path;
    }
    __host__ Model load() const;
};
