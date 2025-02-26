#pragma once

template<typename T, int MAX_SIZE>
class CudaStack {
private:
    T data[MAX_SIZE];
    int top;

public:
    __device__ CudaStack() : top(-1) {}
    
    __device__ __forceinline__ bool empty() const {
        return top < 0;
    }
    
    __device__ __forceinline__ bool full() const {
        return top >= MAX_SIZE - 1;
    }
    
    __device__ __forceinline__ void push(const T& item) {
        if (!full()) {
            data[++top] = item;
        }
    }
    
    __device__ __forceinline__ T pop() {
        if (!empty()) {
            return data[top--];
        }

        return T();
    }
    
    __device__ __forceinline__ T& peek() {
        return data[top];
    }
    
    __device__ __forceinline__ int size() const {
        return top + 1;
    }
    
    __device__ __forceinline__ void clear() {
        top = -1;
    }
};