#include "LazyAtenFunctions.h"

namespace {

TORCH_LIBRARY_IMPL(aten, Lazy, m) {
  m.impl("sqrt", TORCH_FN(lazy_mode::LazyNativeFunctions::sqrt));
  m.impl("mm", TORCH_FN(lazy_mode::LazyNativeFunctions::mm));
  m.impl("relu", TORCH_FN(lazy_mode::LazyNativeFunctions::relu));
}

} // anonymous namespace
