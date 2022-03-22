#pragma once

#include <ATen/Tensor.h>
#include <torch/library.h>

namespace lazy_mode {

struct LazyNativeFunctions {

static at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
static at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2);
static at::Tensor relu(const at::Tensor & self);

};
} // namespace lazy_mode
