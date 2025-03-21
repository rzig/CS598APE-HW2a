#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
// Include ctimer for end-to-end timing
#include "common.h"
#include "ctimer.h"
#include "data.h"
#include "fitness.h"
#include "genetic.h"
#include "jit_cuda.h"
#include "node.h"
#include "program.h"
#include <sstream>

namespace jit {
namespace cuda {
std::string generate_program_kernel(const genetic::program &program, size_t i);
std::string generate_full_cuda_src(const genetic::program_t dprogs, size_t nprogs);
CUmodule compile_cuda_src(const std::string& src);
std::pair<CUmodule, std::vector<CUfunction>> jit_all(const genetic::program_t dprogs, size_t n_progs);
std::pair<CUmodule, CUfunction> jit_single(const genetic::program& prog);
std::string generate_single_cuda_src(const genetic::program& prog);

} // namespace cuda
} // namespace jit
