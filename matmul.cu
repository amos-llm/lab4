#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

template <typename T> __host__ __device__ T cdiv(T a, T b) { return (a + b - 1) / b; }

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

constexpr int32_t BLOCK_M = 32;
constexpr int32_t BLOCK_N = 32;
constexpr int32_t BLOCK_K = 32;

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    __shared__ float smem_a[BLOCK_M][BLOCK_K];
    __shared__ float smem_b[BLOCK_K][BLOCK_N];

    float acc = 0.0f;
    for (int tile_k = 0; tile_k < cdiv(size_k, BLOCK_K); ++tile_k) {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BLOCK_M * BLOCK_K;
             i += blockDim.x * blockDim.y) {
            int local_row = i / BLOCK_K;
            int local_col = i % BLOCK_K;
            int global_row = blockIdx.y * BLOCK_M + local_row;
            int global_col = tile_k * BLOCK_K + local_col;
            if (global_row < size_i && global_col < size_k) {
                smem_a[local_row][local_col] = a[global_row * size_k + global_col];
            } else {
                smem_a[local_row][local_col] = 0.0f;
            }
        }
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < BLOCK_K * BLOCK_N;
             i += blockDim.x * blockDim.y) {
            int local_row = i / BLOCK_N;
            int local_col = i % BLOCK_N;
            int global_row = tile_k * BLOCK_K + local_row;
            int global_col = blockIdx.x * BLOCK_N + local_col;
            if (global_row < size_k && global_col < size_j) {
                smem_b[local_row][local_col] = b[global_row * size_j + global_col];
            } else {
                smem_b[local_row][local_col] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            acc += smem_a[threadIdx.y][k] * smem_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    int global_row = blockIdx.y * BLOCK_M + threadIdx.y;
    int global_col = blockIdx.x * BLOCK_N + threadIdx.x;
    if (global_row < size_i && global_col < size_j) {
        c[global_row * size_j + global_col] = acc;
    }
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid(cdiv(size_j, BLOCK_N), cdiv(size_i, BLOCK_M));
    matmul_l1<<<grid, block>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

constexpr int32_t BLOCK_M = 64;
constexpr int32_t BLOCK_N = 64;
constexpr int32_t BLOCK_K = 16;
constexpr int32_t TM = 4;
constexpr int32_t TN = 4;

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    __shared__ float smem_a[BLOCK_M][BLOCK_K];
    __shared__ float smem_b[BLOCK_K][BLOCK_N];

    float acc[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int num_k_tiles = cdiv(size_k, BLOCK_K);

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += threads_per_block) {
            int local_row = i / BLOCK_K;
            int local_col = i % BLOCK_K;
            int global_row = blockIdx.y * BLOCK_M + local_row;
            int global_col = tile_k * BLOCK_K + local_col;
            if (global_row < size_i && global_col < size_k) {
                smem_a[local_row][local_col] = a[global_row * size_k + global_col];
            } else {
                smem_a[local_row][local_col] = 0.0f;
            }
        }
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += threads_per_block) {
            int local_row = i / BLOCK_N;
            int local_col = i % BLOCK_N;
            int global_row = tile_k * BLOCK_K + local_row;
            int global_col = blockIdx.x * BLOCK_N + local_col;
            if (global_row < size_k && global_col < size_j) {
                smem_b[local_row][local_col] = b[global_row * size_j + global_col];
            } else {
                smem_b[local_row][local_col] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = smem_a[threadIdx.y * TM + i][k];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = smem_b[k][threadIdx.x * TN + j];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_row = blockIdx.y * BLOCK_M + threadIdx.y * TM + i;
            int global_col = blockIdx.x * BLOCK_N + threadIdx.x * TN + j;
            if (global_row < size_i && global_col < size_j) {
                c[global_row * size_j + global_col] = acc[i][j];
            }
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    dim3 block(BLOCK_N / TN, BLOCK_M / TM);
    dim3 grid(cdiv(size_j, BLOCK_N), cdiv(size_i, BLOCK_M));
    matmul_l1_reg<<<grid, block>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
