#pragma once

struct vec3 {
    float e[3];
    __host__ __device__ __forceinline__ vec3() {
        e[0] = e[1] = e[2] = 0.0f;
    }
    __host__ __device__ __forceinline__ vec3(float e0, float e1, float e2) {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ __forceinline__ float x() const { return e[0]; }
    __host__ __device__ __forceinline__ float y() const { return e[1]; }
    __host__ __device__ __forceinline__ float z() const { return e[2]; }
    __host__ __device__ __forceinline__ float r() const { return e[0]; }
    __host__ __device__ __forceinline__ float g() const { return e[1]; }
    __host__ __device__ __forceinline__ float b() const { return e[2]; }
    __host__ __device__ __forceinline__ const vec3& operator+() const {return *this;}
    __host__ __device__ __forceinline__ float operator[](const int i) const {return e[i];}
    __host__ __device__ __forceinline__ float& operator[](const int i) {return e[i];}

    __host__ __device__ __forceinline__ vec3 operator-() const {
        return vec3(-e[0], -e[1], -e[2]);
    }
    __host__ __device__ __forceinline__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    __host__ __device__ __forceinline__ vec3& operator-=(const vec3& v) {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }
    __host__ __device__ __forceinline__ vec3& operator*=(const vec3& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }
    __host__ __device__ __forceinline__ vec3& operator/=(const vec3& v) {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        return *this;
    }
    __host__ __device__ __forceinline__ vec3& operator*=(const float x) {
        e[0] *= x;
        e[1] *= x;
        e[2] *= x;
        return *this;
    }
    __host__ __device__ __forceinline__ vec3& operator/=(const float x) {
        e[0] /= x;
        e[1] /= x;
        e[2] /= x;
        return *this;
    }
    
    __host__ __device__ __forceinline__ float length() const {
        return sqrt(squared_length());
    }
    
    __host__ __device__ __forceinline__ float squared_length() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
    __host__ __device__ __forceinline__ float maxElement() const {
        return fmax(e[0], fmax(e[1], e[2]));
    }
    __host__ __device__ __forceinline__ float minElement() const {
        return fmin(e[0], fmin(e[1], e[2]));
    }
    __host__ __device__ __forceinline__ int maxElementAxis() const {
        return e[0] > e[1] ? (e[0] > e[2] ? 0 : 2) : (e[1] > e[2] ? 1 : 2);
    }
    __host__ __device__ __forceinline__ int minElementAxis() const {
        return e[0] < e[1] ? (e[0] < e[2] ? 0 : 2) : (e[1] < e[2] ? 1 : 2);
    }
    
};
__host__ __device__ __forceinline__ vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}
__host__ __device__ __forceinline__ vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
}
__host__ __device__ __forceinline__ vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}
__host__ __device__ __forceinline__ vec3 operator/(const vec3& v1, const vec3& v2) {
    return vec3(v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z());
}
__host__ __device__ __forceinline__ vec3 operator*(const float v1, const vec3& v2) {
    return vec3(v1 * v2.x(), v1 * v2.y(), v1 * v2.z());
}
__host__ __device__ __forceinline__ vec3 operator/(const float v1, const vec3& v2) {
    return vec3(v1 / v2.x(), v1 / v2.y(), v1 / v2.z());
}
__host__ __device__ __forceinline__ vec3 operator*(const vec3& v1, const float v2) {
    return vec3(v1.x() * v2, v1.y() * v2, v1.z() * v2);
}
__host__ __device__ __forceinline__ vec3 operator/(const vec3& v1, const float v2) {
    return vec3(v1.x() / v2, v1.y() / v2, v1.z() / v2);
}
__host__ __device__ __forceinline__ float dot(const vec3& v1, const vec3& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}
__host__ __device__ __forceinline__ vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.y() * v2.z() - v1.z() * v2.y(),
                v1.z() * v2.x() - v1.x() * v2.z(),
                v1.x() * v2.y() - v1.y() * v2.x());
}
__host__ __device__ __forceinline__ vec3 unit_vector(vec3 v) {
    return v / v.length();
}
__host__ __device__ __forceinline__ vec3 vec3min(const vec3& v1, const vec3& v2) {
    return vec3(fmin(v1.x(), v2.x()), fmin(v1.y(), v2.y()), fmin(v1.z(), v2.z()));
}
__host__ __device__ __forceinline__ vec3 vec3max(const vec3& v1, const vec3& v2) {
    return vec3(fmax(v1.x(), v2.x()), fmax(v1.y(), v2.y()), fmax(v1.z(), v2.z()));
}
