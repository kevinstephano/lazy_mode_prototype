#pragma once

#include <ATen/Tensor.h>
#include <torch/library.h>

namespace lazy_mode {

struct LazyNativeFunctions {

static at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2);
static at::Tensor relu(const at::Tensor & self);
static at::Tensor sqrt(const at::Tensor & self);

};
} // namespace lazy_mode
