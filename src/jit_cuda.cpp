#include "jit_cuda.h"

namespace jit {
namespace cuda {
std::string generate_program_kernel(const genetic::program &program, size_t i) {
  auto string_repr = genetic::stringify(program);
  std::stringstream cuda_code;

  cuda_code << "extern \"C\" __global__ void k" << i
            << "(float* inp, float* outp, int num_rows, int row_width, int pid) {"
            << std::endl;
  cuda_code << "int row_idx = blockIdx.x * blockDim.x + threadIdx.x;"
            << std::endl;
  cuda_code << "if (row_idx < num_rows) {" << std::endl;
  cuda_code << "float* row = &inp[row_idx*row_width];" << std::endl;
  cuda_code << "outp[pid*num_rows + row_idx] = " << string_repr << ";" << std::endl;
  cuda_code << "}" << std::endl; // end if
  cuda_code << "}" << std::endl; // end kernel

  return cuda_code.str();
}

static const std::string kernel_prelude = R"(
__device__ __forceinline__ float add(float a, float b) {
    return a + b;
}

__device__ __forceinline__ float div(float a, float b) {
    float abs_rhs = fabsf(b);
    return abs_rhs < 0.001f ? 1.0f : (a/b);
}

__device__ __forceinline__ float mult(float a, float b) {
    return a * b;
}

__device__ __forceinline__ float sub(float a, float b) {
    return a - b;
}

// Unary operations
__device__ __forceinline__ float cube(float x) {
    return x * x * x;
}

__device__ __forceinline__ float inv(float x) {
    float abs_val = fabsf(x);
    return abs_val < 0.001f ? 0.f : 1.f / x;
}

__device__ __forceinline__ float neg(float x) {
    return -x;
}

__device__ __forceinline__ float custom_rsqrt(float x) {
    // CUDA has a built-in fast reciprocal square root
    float abs_val = fabsf(x);
    return rsqrtf(abs_val);
}

__device__ __forceinline__ float sq(float x) {
    return x * x;
}

__device__ __forceinline__ float custom_log(float x) {
  float abs_val = fabsf(x);
  return abs_val < 0.001f ? 0.f : logf(abs_val);
}

__device__ __forceinline__ float custom_sqrt(float x) {
  float abs_val = fabsf(x);
  return sqrtf(abs_val);
}


)";

std::string generate_full_cuda_src(const genetic::program_t dprogs, const size_t nprogs) {
  std::stringstream cuda_kernel;
  cuda_kernel << kernel_prelude;
  for (size_t i = 0; i < nprogs; i++) {
    cuda_kernel << generate_program_kernel(dprogs[i], i);
  }         
  return cuda_kernel.str();
}
std::string generate_single_cuda_src(const genetic::program& prog) {
  std::stringstream cuda_kernel;
  cuda_kernel << kernel_prelude;
  cuda_kernel << generate_program_kernel(prog, 0);
  return cuda_kernel.str();
}

CUmodule compile_cuda_src(const std::string& src) {
  nvrtcProgram prog;
  CHECK_NVRTC(nvrtcCreateProgram(&prog, src.c_str(), "kernel.cu", 0, nullptr, nullptr)); // filename doesn't matter
  // https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  const char* options[] = {"--gpu-architecture=compute_52"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, options);
      if (compileResult != NVRTC_SUCCESS) {
        // Get compilation log
        size_t logSize;
        CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &logSize));
        char* log = new char[logSize];
        CHECK_NVRTC(nvrtcGetProgramLog(prog, log));
        std::cerr << "Compilation failed:\n" << log << std::endl;
        delete[] log;
        exit(1);
    }
  size_t ptxSize;
  CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
  char* ptx = new char[ptxSize];
  CHECK_NVRTC(nvrtcGetPTX(prog, ptx));
  // std::cout << src << std::endl;
  CUmodule module;
  // std::cout << ptx << std::endl;
  CHECK_CUDA(cuModuleLoadData(&module, ptx));

  nvrtcDestroyProgram(&prog);
  
  return module;
}

std::pair<CUmodule, CUfunction> jit_single(const genetic::program& prog) {
  auto all_cuda = jit::cuda::generate_single_cuda_src(prog);
  CUmodule module = jit::cuda::compile_cuda_src(all_cuda);
  CUfunction func;
  CHECK_CUDA(cuModuleGetFunction(&func, module, "k0"));
  return std::make_pair(module, func);
}

std::pair<CUmodule, std::vector<CUfunction>> jit_all(const genetic::program_t d_progs, const size_t n_progs) {
  auto all_cuda = jit::cuda::generate_full_cuda_src(d_progs, n_progs);
  CUmodule module = jit::cuda::compile_cuda_src(all_cuda);
  std::vector<CUfunction> res;
  for (size_t i = 0; i < n_progs; i++) {
    CUfunction k;
    std::string fname = std::string("k") + std::to_string(i);
    CHECK_CUDA(cuModuleGetFunction(&k, module, fname.c_str()));
    res.push_back(k);
  }
  return std::make_pair(module, res);
}
} // namespace cuda
} // namespace jit
