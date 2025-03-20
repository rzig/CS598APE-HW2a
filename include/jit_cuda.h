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
} // namespace cuda
} // namespace jit
