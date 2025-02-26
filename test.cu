#include "include/vec3.cuh"
#include "include/model_loader.cuh"
#include "include/model.cuh"
#include "include/vertex.cuh"
#include "include/scene.cuh"
#include "include/bvh.cuh"
#include "include/utils.cuh"
#include <stdio.h>
#include <string>

int main() {
    std::string path = "D:/Code/C++/RayTracerCUDA/plane.obj";
    printf("Looking for file at: %s\n", path.c_str());
    ModelLoader loader(path);
    Model model = loader.load();
    Scene scene;
    scene.addModel(model);
    scene.prepare_buffers();
    scene.buildBVH();
    cpySceneToDevice(scene);
    return 0;
}
