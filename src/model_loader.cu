#include "../include/model_loader.cuh"
#include "../include/vec3.cuh"
#include "../include/vertex.cuh"
#include "../include/model.cuh"
#include <fstream>
#include <sstream>
#include <string>

__host__ Model ModelLoader::load() const
{   
    std::vector<unsigned int> index_buffer;
    std::vector<vec3> ps;
    std::vector<vec3> ns;
    std::vector<float> us;
    std::vector<float> vs;
    std::ifstream file(path);
    std::string line;
    if (!file.is_open()) {
        printf("Error: Unable to open file %s\n", path.c_str());
        return Model();
    }
    while (getline(file, line))
    {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v")
        {
            float x, y, z;
            iss >> x >> y >> z;
            ps.push_back(vec3(x, y, z));
        }
        else if (type == "vn")
        {
            float x, y, z;
            iss >> x >> y >> z;
            ns.push_back(vec3(x, y, z));
        }
        else if (type == "vt")
        {
            float u, v;
            iss >> u >> v;
            us.push_back(u);
            vs.push_back(v);
        }
        else if (type == "f")
        {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            int i1, i2, i3;
            int j1, j2, j3;
            int k1, k2, k3;
            sscanf(v1.c_str(), "%d/%d/%d", &i1, &j1, &k1);
            sscanf(v2.c_str(), "%d/%d/%d", &i2, &j2, &k2);
            sscanf(v3.c_str(), "%d/%d/%d", &i3, &j3, &k3);
            index_buffer.push_back(i1 - 1);
            index_buffer.push_back(i2 - 1);
            index_buffer.push_back(i3 - 1);
        }
    }
    std::vector<vertex> vertex_buffer(ps.size());
    for(size_t i = 0; i < ps.size(); i++) {
        vertex_buffer[i].p = ps[i];
        if(i < ns.size()) {
            vertex_buffer[i].n = ns[i];
        }
        if(i < us.size() && i < vs.size()) {
            vertex_buffer[i].u = us[i];
            vertex_buffer[i].v = vs[i];
        }
    }
    return Model(vertex_buffer, index_buffer);
}