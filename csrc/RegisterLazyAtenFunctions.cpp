#include "LazyAtenFunctions.h"

namespace {

TORCH_LIBRARY_IMPL(aten, Lazy, m) {
  m.impl("add", TORCH_FN(lazy_mode::LazyNativeFunctions::add);
  m.impl("mm", TORCH_FN(lazy_mode::LazyNativeFunctions::mm));
  m.impl("relu", TORCH_FN(lazy_mode::LazyNativeFunctions::relu));
}

} // anonymous namespace
