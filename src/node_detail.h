#pragma once
#ifndef GENETIC_NODE_DETAIL_H
#define GENETIC_NODE_DETAIL_H
#include "data.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <node.h>
#include <type_traits>

namespace genetic {
namespace detail {

static constexpr float MIN_VAL = 0.001f;

inline bool is_terminal(node::type t) {
  return t == node::type::variable || t == node::type::constant;
}

inline bool is_nonterminal(node::type t) { return !is_terminal(t); }

inline int arity(node::type t) {
  if (node::type::unary_begin <= t && t <= node::type::unary_end) {
    return 1;
  }
  if (node::type::binary_begin <= t && t <= node::type::binary_end) {
    return 2;
  }
  return 0;
}

inline void evaluate_node_batched(const node &n, const Dataset<float> &data,
                                  size_t batch, const float *__restrict lhs,
                                  const float *__restrict rhs,
                                  float *__restrict result) {
  if (n.t == node::type::variable) { // avoid a copy;
    float *bat = data.column_of_batch(batch, n.u.fid);
    for (size_t i = 0; i < data.batch_size(); i++) {
      result[i] = bat[i];
    }
    return;
  } else if (n.t == node::type::constant) {
    for (size_t i = 0; i < data.batch_size(); i++) {
      result[i] = n.u.val;
    }
    return;
  } else {
    auto batch_size = data.batch_size();
    switch (n.t) {
    // binary operators
    case node::type::add:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = lhs[i] + rhs[i];
      }
      break;
    case node::type::atan2:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = atan2f(lhs[i], rhs[i]);
      }
      break;
    case node::type::div:
      for (size_t i = 0; i < batch_size; i++) {
        float abs_rhs = fabsf(rhs[i]);
        result[i] = abs_rhs < MIN_VAL ? 1.0f : (lhs[i] / rhs[i]);
      }
      break;
    case node::type::fdim:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = fdimf(lhs[i], rhs[i]);
      }
      break;
    case node::type::max:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = fmaxf(lhs[i], rhs[i]);
      }
      break;
    case node::type::min:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = fminf(lhs[i], rhs[i]);
      }
      break;
    case node::type::mul:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = lhs[i] * rhs[i];
      }
      break;
    case node::type::pow:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = powf(lhs[i], rhs[i]);
      }
      break;
    case node::type::sub:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = lhs[i] - rhs[i];
      }
      break;
    // unary operators
    case node::type::abs:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = fabsf(lhs[i]);
      }
      break;
    case node::type::acos:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = acosf(lhs[i]);
      }
      break;
    case node::type::acosh:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = acoshf(lhs[i]);
      }
      break;
    case node::type::asin:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = asinf(lhs[i]);
      }
      break;
    case node::type::asinh:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = asinhf(lhs[i]);
      }
      break;
    case node::type::atan:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = atanf(lhs[i]);
      }
      break;
    case node::type::atanh:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = atanhf(lhs[i]);
      }
      break;
    case node::type::cbrt:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = cbrtf(lhs[i]);
      }
      break;
    case node::type::cos:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = cosf(lhs[i]);
      }
      break;
    case node::type::cosh:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = coshf(lhs[i]);
      }
      break;
    case node::type::cube:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = lhs[i] * lhs[i] * lhs[i];
      }
      break;
    case node::type::exp:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = expf(lhs[i]);
      }
      break;
    case node::type::inv:
      for (size_t i = 0; i < batch_size; i++) {
        float abs_val = fabsf(lhs[i]);
        result[i] = abs_val < MIN_VAL ? 0.f : 1.f / lhs[i];
      }
      break;
    case node::type::log:
      for (size_t i = 0; i < batch_size; i++) {
        float abs_val = fabsf(lhs[i]);
        result[i] = abs_val < MIN_VAL ? 0.f : logf(abs_val);
      }
      break;
    case node::type::neg:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = -lhs[i];
      }
      break;
    case node::type::rcbrt:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = static_cast<float>(1.0) / cbrtf(lhs[i]);
      }
      break;
    case node::type::rsqrt:
      for (size_t i = 0; i < batch_size; i++) {
        float abs_val = fabsf(lhs[i]);
        result[i] = static_cast<float>(1.0) / sqrtf(abs_val);
      }
      break;
    case node::type::sin:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = sinf(lhs[i]);
      }
      break;
    case node::type::sinh:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = sinhf(lhs[i]);
      }
      break;
    case node::type::sq:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = lhs[i] * lhs[i];
      }
      break;
    case node::type::sqrt:
      for (size_t i = 0; i < batch_size; i++) {
        float abs_val = fabsf(lhs[i]);
        result[i] = sqrtf(abs_val);
      }
      break;
    case node::type::tan:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = tanf(lhs[i]);
      }
      break;
    case node::type::tanh:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = tanhf(lhs[i]);
      }
      break;
    // shouldn't reach here!
    default:
      for (size_t i = 0; i < batch_size; i++) {
        result[i] = 0.f;
      }
      break;
    };
  }
}

// `data` assumed to be stored in col-major format
inline float evaluate_node(const node &n, const float *data,
                           const uint64_t stride, const uint64_t idx,
                           const float *in) {
  if (n.t == node::type::constant) {
    return n.u.val;
  } else if (n.t == node::type::variable) {
    return data[(stride * n.u.fid) + idx];
  } else {
    auto abs_inval = fabsf(in[0]), abs_inval1 = fabsf(in[1]);
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators
    case node::type::add:
      return in[0] + in[1];
    case node::type::atan2:
      return atan2f(in[0], in[1]);
    case node::type::div:
      return abs_inval1 < MIN_VAL ? 1.0f
                                  : (in[0] / in[1]); // fdividef(in[0], in[1]);
    case node::type::fdim:
      return fdimf(in[0], in[1]);
    case node::type::max:
      return fmaxf(in[0], in[1]);
    case node::type::min:
      return fminf(in[0], in[1]);
    case node::type::mul:
      return in[0] * in[1];
    case node::type::pow:
      return powf(in[0], in[1]);
    case node::type::sub:
      return in[0] - in[1];
    // unary operators
    case node::type::abs:
      return abs_inval;
    case node::type::acos:
      return acosf(in[0]);
    case node::type::acosh:
      return acoshf(in[0]);
    case node::type::asin:
      return asinf(in[0]);
    case node::type::asinh:
      return asinhf(in[0]);
    case node::type::atan:
      return atanf(in[0]);
    case node::type::atanh:
      return atanhf(in[0]);
    case node::type::cbrt:
      return cbrtf(in[0]);
    case node::type::cos:
      return cosf(in[0]);
    case node::type::cosh:
      return coshf(in[0]);
    case node::type::cube:
      return in[0] * in[0] * in[0];
    case node::type::exp:
      return expf(in[0]);
    case node::type::inv:
      return abs_inval < MIN_VAL ? 0.f : 1.f / in[0];
    case node::type::log:
      return abs_inval < MIN_VAL ? 0.f : logf(abs_inval);
    case node::type::neg:
      return -in[0];
    case node::type::rcbrt:
      return static_cast<float>(1.0) / cbrtf(in[0]);
    case node::type::rsqrt:
      return static_cast<float>(1.0) / sqrtf(abs_inval);
    case node::type::sin:
      return sinf(in[0]);
    case node::type::sinh:
      return sinhf(in[0]);
    case node::type::sq:
      return in[0] * in[0];
    case node::type::sqrt:
      return sqrtf(abs_inval);
    case node::type::tan:
      return tanf(in[0]);
    case node::type::tanh:
      return tanhf(in[0]);
    // shouldn't reach here!
    default:
      return 0.f;
    };
  }
}

} // namespace detail
} // namespace genetic
#endif
