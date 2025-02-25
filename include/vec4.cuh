#pragma once
struct vec4 {
    float e[4];

    __host__ __device__ __forceinline__ vec4() {
        e[0] = 0;
        e[1] = 0;
        e[2] = 0;
        e[3] = 1;
    }
    __host__ __device__ __forceinline__ vec4(float x, float y, float z, float w) {
        e[0] = x;
        e[1] = y;
        e[2] = z;
        e[3] = w;
    }

    __host__ __device__ __forceinline__ float x() const {return e[0];}
    __host__ __device__ __forceinline__ float y() const {return e[1];}
    __host__ __device__ __forceinline__ float z() const {return e[2];}
    __host__ __device__ __forceinline__ float w() const {return e[3];}

    __host__ __device__ __forceinline__ vec4 operator+() const {return *this;}
    __host__ __device__ __forceinline__ vec4 operator-() const;
    __host__ __device__ __forceinline__ float operator[](const int& i) const {return e[i];}
    __host__ __device__ __forceinline__ float& operator[](const int& i) {return e[i];}

    __host__ __device__ __forceinline__ vec4& operator+=(const vec4&);
    __host__ __device__ __forceinline__ vec4& operator-=(const vec4&);
    __host__ __device__ __forceinline__ vec4& operator*=(const vec4&);
    __host__ __device__ __forceinline__ vec4& operator/=(const vec4&);
    __host__ __device__ __forceinline__ vec4& operator*=(const float);
    __host__ __device__ __forceinline__ vec4& operator/=(const float);
};

__host__ __device__ __forceinline__ vec4 operator+(const vec4&, const vec4&);
__host__ __device__ __forceinline__ vec4 operator-(const vec4&, const vec4&);
__host__ __device__ __forceinline__ vec4 operator*(const vec4&, const vec4&);
__host__ __device__ __forceinline__ vec4 operator/(const vec4&, const vec4&);
__host__ __device__ __forceinline__ vec4 operator*(const float, const vec4&);
__host__ __device__ __forceinline__ vec4 operator/(const float, const vec4&);
__host__ __device__ __forceinline__ vec4 operator*(const vec4&, const float);
__host__ __device__ __forceinline__ vec4 operator/(const vec4&, const float);
