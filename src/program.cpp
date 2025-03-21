#include "constants.h"
#include "custom_distributions.h"
#include "data.h"
#include "node_detail.h"
#include "philox_engine.h"
#include "reg_stack.h"
#include "jit_cuda.h"
#include <algorithm>
#include <cstdint>
#include <fitness.h>
#include <iostream>
#include <node.h>
#include <numeric>
#include <omp.h>
#include <program.h>
#include <random>
#include <stack>
#include <chrono>
#include "types.h"

namespace genetic {
std::string stringify(const program &);
/**
 * Execution kernel for a single program. We assume that the input data
 * is stored in column major format.
 */
template <int MaxSize = MAX_STACK_SIZE>
void execute_kernel(const program_t d_progs, const Dataset<float> &data,
                    ypred_t y_pred, const uint64_t n_rows,
                    const uint64_t n_progs) {
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()
            << std::endl;

#ifndef CUDA_MODE
#pragma omp parallel for schedule(dynamic)
#endif
#ifdef CUDA_MODE
int pid_c = 0;
size_t batch_size = 2;
static std::vector<std::pair<CUmodule, CUfunction>> outs(2);
while (pid_c < n_progs) {
  for(int pid = pid_c; pid < pid_c + batch_size && pid < n_progs; pid++) {
int threadsPerBlock = 256;
int blocksPerGrid = (n_rows + threadsPerBlock - 1) / threadsPerBlock;
int row_width = 8;
CUdeviceptr dd = data.device_data();
int size = n_rows;
void* args[] = {&dd, &y_pred, &size, &row_width, &pid};
auto compile_res = jit::cuda::jit_single(d_progs[pid]);
outs[pid_c - pid] = compile_res;
auto fn = compile_res.second;
        CHECK_CUDA(cuLaunchKernel(fn, 
                          blocksPerGrid, 1, 1,    // grid dimensions
                          threadsPerBlock, 1, 1,  // block dimensions
                          0, NULL,                // shared memory size and stream
                          args, 0));              // arguments and extra options
}
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()
            << std::endl;
  CHECK_CUDA(cuCtxSynchronize());
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()
            << std::endl;
if (pid_c + batch_size < n_progs) {
for (size_t i = 0; i < batch_size; i++) {
  cuModuleUnload(outs[i].first);
} 
}
pid_c += batch_size;
      // CHECK_CUDA(cuMemcpyDtoH(y_pred, device_res, sizeof(float)*n_rows));
#else
  for (int pid = 0; pid < n_progs; ++pid) {
    alignas(64) float res[Dataset<float>::batch_size_];
    BatchStack<float, MaxSize, Dataset<float>::batch_size_> stacks;
    for (size_t batch = 0; batch < data.num_batches(); batch++) {
      stacks.clear();
      program_t curr_p = d_progs + pid;
      int end = curr_p->len - 1;
      node *curr_node = curr_p->nodes + end;
      float *lhs = nullptr;
      float *rhs = nullptr;

      while (end >= 0) {
        if (detail::is_nonterminal(curr_node->t)) {
          int ar = detail::arity(curr_node->t);
          lhs = stacks.peek();
          stacks.pop();
          if (ar > 1) {
            rhs = stacks.peek();
            stacks.pop();
          }
        }
        detail::evaluate_node_batched(*curr_node, data, batch, lhs, rhs, res);
        stacks.push(res);
        curr_node--;
        end--;
      }
      auto re = stacks.peek();
      for (size_t i = 0; i < data.batch_size(); i++) {
        if (pid == 0) {
          if(fabsf(y_pred[batch*data.batch_size() + i] - re[i]) > 0.01) {
            std::cout << "y_pred: " << y_pred[batch*data.batch_size() + i] << " re " << re[i] << std::endl;
            std::cout << "idx:" << batch*data.batch_size() + i << std::endl;
            for (size_t j = 0; j < 8; j++) {
              std::cout << "row[" << j << "]=" << data.column_of_batch(batch, j)[0] << std::endl;
            }
            exit(1);
          }
          continue;
        } else 
          y_pred[pid * n_rows + (batch * data.batch_size() + i)] = re[i];
      }
    }
    #endif
  }
  #ifdef CUDA_MODE
  CHECK_CUDA(cuCtxSynchronize());
  #endif
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()
            << std::endl;

}

program::program()
    : len(0), depth(0), raw_fitness_(0.0f), metric(metric_t::mse),
      mut_type(mutation_t::none), nodes(nullptr) {}

program::~program() { delete[] nodes; }

program::program(const program &src)
    : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_),
      metric(src.metric), mut_type(src.mut_type) {
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);
}

program &program::operator=(const program &src) {
  len = src.len;
  depth = src.depth;
  raw_fitness_ = src.raw_fitness_;
  metric = src.metric;
  mut_type = src.mut_type;

  // Copy nodes
  delete[] nodes;
  nodes = new node[len];
  std::copy(src.nodes, src.nodes + src.len, nodes);

  return *this;
}

void compute_metric(int n_rows, int n_progs, const float *y,
                    const ypred_t y_pred, const float *w, float *score,
                    const param &params) {
  exit(1);
  return;
  // Call appropriate metric function based on metric defined in params
  // if (params.metric == metric_t::pearson) {
  //   weightedPearson(n_rows, n_progs, y, y_pred, w, score);
  // } else if (params.metric == metric_t::spearman) {
  //   weightedSpearman(n_rows, n_progs, y, y_pred, w, score);
  // } else if (params.metric == metric_t::mae) {
  //   meanAbsoluteError(n_rows, n_progs, y, y_pred, w, score);
  // } else if (params.metric == metric_t::mse) {
  //   meanSquareError(n_rows, n_progs, y, y_pred, w, score);
  // } else if (params.metric == metric_t::rmse) {
  //   rootMeanSquareError(n_rows, n_progs, y, y_pred, w, score);
  // } else if (params.metric == metric_t::logloss) {
  //   logLoss(n_rows, n_progs, y, y_pred, w, score);
  // } else {
  //   // This should not be reachable
  // }
}

void execute(const program_t &d_progs, const int n_rows, const int n_progs,
             const Dataset<float> &data, ypred_t y_pred) {
  execute_kernel(d_progs, data, y_pred, static_cast<uint64_t>(n_rows),
                 static_cast<uint64_t>(n_progs));
  std::cout << "done w here" << std::endl;
}

void find_fitness(program_t d_prog, float *score, const param &params,
                  const int n_rows, const Dataset<float> &data, const float *y,
                  const float *sample_weights) {

  std::vector<float> y_pred(n_rows);
  // Compute predicted values
  #ifdef CUDA_MODE
  CUdeviceptr d_output;
  CHECK_CUDA(cuMemAlloc(&d_output, n_rows*2*sizeof(float)));
  execute(d_prog, n_rows, 1, data, d_output);
  #else
  execute(d_prog, n_rows, 1, data, y_pred.data());
  #endif
std::cout << "done with ex" << std::endl;
  // Compute error
  #ifdef CUDA_MODE
  compute_metric(n_rows, 1, y, d_output, sample_weights, score, params);
  #else
  compute_metric(n_rows, 1, y, y_pred.data(), sample_weights, score, params);
  #endif
}

void find_batched_fitness(int n_progs, program_t d_progs, float *score,
                          const param &params, const int n_rows,
                          const Dataset<float> &data, const float *y,
                          const float *sample_weights) {

  std::vector<float> y_pred((uint64_t)n_rows * (uint64_t)n_progs);
  #ifdef CUDA_MODE
  CUdeviceptr d_output;
  CHECK_CUDA(cuMemAlloc(&d_output, n_rows*n_progs*sizeof(float)));
  execute(d_progs, n_rows, n_progs, data, d_output);
  #else
  execute(d_progs, n_rows, n_progs, data, y_pred.data());
  #endif
std::cout << "here in find batched" << std::endl;
  #ifdef CUDA_MODE
  // Compute error
  compute_metric(n_rows, n_progs, y, d_output, sample_weights, score,
                 params);
  #else
  compute_metric(n_rows, n_progs, y, y_pred.data(), sample_weights, score,
                 params);
  #endif
}

void set_fitness(program &h_prog, const param &params, const int n_rows,
                 const Dataset<float> &data, const float *y,
                 const float *sample_weights) {

  std::vector<float> score(1);

  find_fitness(&h_prog, score.data(), params, n_rows, data, y, sample_weights);

  // Update host and device score for program
  h_prog.raw_fitness_ = score[0];
}

void set_batched_fitness(int n_progs, std::vector<program> &h_progs,
                         const param &params, const int n_rows,
                         const Dataset<float> &data, const float *y,
                         const float *sample_weights) {

  std::vector<float> score(n_progs);

  find_batched_fitness(n_progs, h_progs.data(), score.data(), params, n_rows,
                       data, y, sample_weights);

  // Update scores on host and device
  // TODO: Find a way to reduce the number of implicit memory transfers
  for (auto i = 0; i < n_progs; ++i) {
    h_progs[i].raw_fitness_ = score[i];
  }
}

float get_fitness(const program &prog, const param &params) {
  int crit = params.criterion();
  float penalty = params.parsimony_coefficient * prog.len * (2 * crit - 1);
  return (prog.raw_fitness_ - penalty);
}

/**
 * @brief Get a random subtree of the current program nodes (on CPU)
 *
 * @param pnodes  AST represented as a list of nodes
 * @param len     The total number of nodes in the AST
 * @param rng     Random number generator for subtree selection
 * @return A tuple [first,last) which contains the required subtree
 */
std::pair<int, int> get_subtree(node *pnodes, int len, PhiloxEngine &rng) {
  int start, end;
  start = end = 0;

  // Specify RNG
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  float bound = dist_uniform(rng);

  // Specify subtree start probs acc to Koza's selection approach
  std::vector<float> node_probs(len, 0.1);
  float sum = 0.1 * len;

  for (int i = 0; i < len; ++i) {
    if (pnodes[i].is_nonterminal()) {
      node_probs[i] = 0.9;
      sum += 0.8;
    }
  }

  // Normalize vector
  for (int i = 0; i < len; ++i) {
    node_probs[i] /= sum;
  }

  // Compute cumulative sum
  std::partial_sum(node_probs.begin(), node_probs.end(), node_probs.begin());

  start = std::lower_bound(node_probs.begin(), node_probs.end(), bound) -
          node_probs.begin();
  end = start;

  // Iterate until all function arguments are satisfied in current subtree
  int num_args = 1;
  while (num_args > end - start) {
    node curr;
    curr = pnodes[end];
    if (curr.is_nonterminal())
      num_args += curr.arity();
    ++end;
  }

  return std::make_pair(start, end);
}

int get_depth(const program &p_out) {
  int depth = 0;
  std::stack<int> arity_stack;
  for (auto i = 0; i < p_out.len; ++i) {
    node curr(p_out.nodes[i]);

    // Update depth
    int sz = arity_stack.size();
    depth = std::max(depth, sz);

    // Update stack
    if (curr.is_nonterminal()) {
      arity_stack.push(curr.arity());
    } else {
      // Only triggered for a depth 0 node
      if (arity_stack.empty())
        break;

      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);

      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty())
          break;

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  return depth;
}

void build_program(program &p_out, const param &params, PhiloxEngine &rng) {
  // Define data structures needed for tree
  std::stack<int> arity_stack;
  std::vector<node> nodelist;
  nodelist.reserve(1 << (MAX_STACK_SIZE));

  // Specify Distributions with parameters
  uniform_int_distribution_custom<int> dist_function(
      0, params.function_set.size() - 1);
  uniform_int_distribution_custom<int> dist_initDepth(params.init_depth[0],
                                                      params.init_depth[1]);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constVal(params.const_range[0],
                                                        params.const_range[1]);
  bernoulli_distribution_custom dist_nodeChoice(params.terminalRatio);
  bernoulli_distribution_custom dist_coinToss(0.5);

  // Initialize nodes
  int max_depth = dist_initDepth(rng);
  node::type func = params.function_set[dist_function(rng)];
  node curr_node(func);
  nodelist.push_back(curr_node);
  arity_stack.push(curr_node.arity());

  init_method_t method = params.init_method;
  if (method == init_method_t::half_and_half) {
    // Choose either grow or full for this tree
    bool choice = dist_coinToss(rng);
    method = choice ? init_method_t::grow : init_method_t::full;
  }

  // Fill tree
  while (!arity_stack.empty()) {
    int depth = arity_stack.size();
    p_out.depth = std::max(depth, p_out.depth);
    bool node_choice = dist_nodeChoice(rng);

    if ((node_choice == false || method == init_method_t::full) &&
        depth < max_depth) {
      // Add a function to node list
      curr_node = node(params.function_set[dist_function(rng)]);
      nodelist.push_back(curr_node);
      arity_stack.push(curr_node.arity());
    } else {
      // Add terminal
      int terminal_choice = dist_terminalChoice(rng);
      if (terminal_choice == params.num_features) {
        // Add constant
        float val = dist_constVal(rng);
        curr_node = node(val);
      } else {
        // Add variable
        int fid = terminal_choice;
        curr_node = node(fid);
      }

      // Modify nodelist
      nodelist.push_back(curr_node);

      // Modify stack
      int e = arity_stack.top();
      arity_stack.pop();
      arity_stack.push(e - 1);
      while (arity_stack.top() == 0) {
        arity_stack.pop();
        if (arity_stack.empty()) {
          break;
        }

        e = arity_stack.top();
        arity_stack.pop();
        arity_stack.push(e - 1);
      }
    }
  }

  // Set new program parameters - need to do a copy as
  // nodelist will be deleted using RAII semantics
  p_out.nodes = new node[nodelist.size()];
  std::copy(nodelist.begin(), nodelist.end(), p_out.nodes);

  p_out.len = nodelist.size();
  p_out.metric = params.metric;
  p_out.raw_fitness_ = 0.0f;
}

void point_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // deep-copy program
  p_out = prog;

  // Specify RNGs
  uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
  uniform_int_distribution_custom<int> dist_terminalChoice(0,
                                                           params.num_features);
  uniform_real_distribution_custom<float> dist_constantVal(
      params.const_range[0], params.const_range[1]);

  // Fill with uniform numbers
  std::vector<float> node_probs(p_out.len);
  std::generate(node_probs.begin(), node_probs.end(),
                [&dist_uniform, &rng] { return dist_uniform(rng); });

  // Mutate nodes
  int len = p_out.len;
  for (int i = 0; i < len; ++i) {
    node curr(prog.nodes[i]);

    if (node_probs[i] < params.p_point_replace) {
      if (curr.is_terminal()) {
        int choice = dist_terminalChoice(rng);

        if (choice == params.num_features) {
          // Add a randomly generated constant
          curr = node(dist_constantVal(rng));
        } else {
          // Add a variable with fid=choice
          curr = node(choice);
        }
      } else if (curr.is_nonterminal()) {
        // Replace current function with another function of the same arity
        int ar = curr.arity();
        // CUML_LOG_DEBUG("Arity is %d, curr function is
        // %d",ar,static_cast<std::underlying_type<node::type>::type>(curr.t));
        std::vector<node::type> fset = params.arity_set.at(ar);
        uniform_int_distribution_custom<int> dist_fset(0, fset.size() - 1);
        int choice = dist_fset(rng);
        curr = node(fset[choice]);
      }

      // Update p_out with updated value
      p_out.nodes[i] = curr;
    }
  }
}

void crossover(const program &prog, const program &donor, program &p_out,
               const param &params, PhiloxEngine &rng) {
  // Get a random subtree of prog to replace
  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  // Set metric of output program
  p_out.metric = prog.metric;

  // MAX_STACK_SIZE can only handle tree of depth MAX_STACK_SIZE -
  // max(func_arity=2) + 1 Thus we continuously hoist the donor subtree. Actual
  // indices in donor
  int donor_start = 0;
  int donor_end = donor.len;
  int output_depth = 0;
  int iter = 0;
  do {
    ++iter;
    // Get donor subtree
    std::pair<int, int> donor_slice =
        get_subtree(donor.nodes + donor_start, donor_end - donor_start, rng);

    // Get indices w.r.t current subspace [donor_start,donor_end)
    int donor_substart = donor_slice.first;
    int donor_subend = donor_slice.second;

    // Update relative indices to global indices
    donor_substart += donor_start;
    donor_subend += donor_start;

    // Update to new subspace
    donor_start = donor_substart;
    donor_end = donor_subend;

    // Evolve on current subspace
    p_out.len =
        (prog_start) + (donor_end - donor_start) + (prog.len - prog_end);
    delete[] p_out.nodes;
    p_out.nodes = new node[p_out.len];

    // Copy slices using std::copy
    std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
    std::copy(donor.nodes + donor_start, donor.nodes + donor_end,
              p_out.nodes + prog_start);
    std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
              p_out.nodes + (prog_start) + (donor_end - donor_start));

    output_depth = get_depth(p_out);
  } while (output_depth >= MAX_STACK_SIZE);

  // Set the depth of the final program
  p_out.depth = output_depth;
}

void subtree_mutation(const program &prog, program &p_out, const param &params,
                      PhiloxEngine &rng) {
  // Generate a random program and perform crossover
  program new_program;
  build_program(new_program, params, rng);
  crossover(prog, new_program, p_out, params, rng);
}

void hoist_mutation(const program &prog, program &p_out, const param &params,
                    PhiloxEngine &rng) {
  // Replace program subtree with a random sub-subtree

  std::pair<int, int> prog_slice = get_subtree(prog.nodes, prog.len, rng);
  int prog_start = prog_slice.first;
  int prog_end = prog_slice.second;

  std::pair<int, int> sub_slice =
      get_subtree(prog.nodes + prog_start, prog_end - prog_start, rng);
  int sub_start = sub_slice.first;
  int sub_end = sub_slice.second;

  // Update subtree indices to global indices
  sub_start += prog_start;
  sub_end += prog_start;

  p_out.len = (prog_start) + (sub_end - sub_start) + (prog.len - prog_end);
  p_out.nodes = new node[p_out.len];
  p_out.metric = prog.metric;

  // Copy node slices using std::copy
  std::copy(prog.nodes, prog.nodes + prog_start, p_out.nodes);
  std::copy(prog.nodes + sub_start, prog.nodes + sub_end,
            p_out.nodes + prog_start);
  std::copy(prog.nodes + prog_end, prog.nodes + prog.len,
            p_out.nodes + (prog_start) + (sub_end - sub_start));

  // Update depth
  p_out.depth = get_depth(p_out);
}

} // namespace genetic
