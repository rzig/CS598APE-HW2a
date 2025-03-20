#include "jit_cuda.h"

namespace jit {
namespace cuda {
std::string generate_program_kernel(const genetic::program &program, size_t i) {
  auto string_repr = genetic::stringify(program);
  std::stringstream cuda_code;

  cuda_code << "void k" << i
            << "(float* inp, float* outp, int num_rows, int row_width) {"
            << std::endl;
  cuda_code << "int row_idx = blockIdx.x * blockDim.x + threadIdx.x;"
            << std::endl;
  cuda_code << "if (row_idx < num_rows) {" << std::endl;
  cuda_code << "float* row = &inp[row_idx*row_width];" << std::endl;
  cuda_code << "outp[row_idx] = " << string_repr << ";" << std::endl;
  cuda_code << "}" << std::endl; // end if
  cuda_code << "}" << std::endl; // end kernel

  return cuda_code.str();
}
} // namespace cuda
} // namespace jit
