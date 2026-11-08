// Print CUDA's own free/total device memory.  Does not allocate a buffer.
// Build: nvcc -O2 -arch=sm_120 meminfo.cu -o build\meminfo.exe
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (e != cudaSuccess) {
        std::printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf(
        "device=%s free=%.3f MiB total=%.3f MiB used=%.3f MiB "
        "reserved_context_overhead=%.3f MiB\n",
        prop.name, free_bytes / 1048576.0, total_bytes / 1048576.0,
        (total_bytes - free_bytes) / 1048576.0,
        (total_bytes - free_bytes) / 1048576.0);
    return 0;
}
