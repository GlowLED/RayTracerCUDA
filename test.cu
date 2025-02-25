#include "include/vec3.cuh"
#include "include/model_loader.cuh"
#include "include/model.cuh"
#include "include/vertex.cuh"
#include "include/scene.cuh"
#include "include/bvh.cuh"
#include <stdio.h>
#include <string>

int main() {
    std::string path = "D:/Code/C++/RayTracerv2/plane.obj";
    printf("Looking for file at: %s\n", path.c_str());
    ModelLoader loader(path);
    Model model = loader.load();
    Scene scene;
    scene.addModel(model);
    scene.prepare_buffers();
    bvh_node* root = scene.buildBVH(32);
    DeviceBVHNode* device_bvh = new DeviceBVHNode[scene.bvh_size];
    bvhFlatten(root, device_bvh);
    return 0;
}
