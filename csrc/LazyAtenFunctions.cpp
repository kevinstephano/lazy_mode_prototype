#include "LazyAtenFunctions.h"
#include "instrumentation.h"
#include "graph_manager.h"
#include "ATen/Functions.h"

#include <iostream>

namespace lazy_mode {
at::Tensor LazyNativeFunctions::mm(const at::Tensor & self, const at::Tensor & mat2) {
    LAZY_PERF_SCOPE("LazyNativeFunctions::mm");
    auto &gm = GraphManager::GetSingleton();

    auto self_ts_value = gm.getTsValue(self);
    auto mat2_ts_value = gm.getTsValue(mat2);

    auto return_value = gm->makeTsNode(c10::Symbol::fromQualString("aten::mm"), {self_ts_value, mat2_ts_value});

    auto return_tensor = at::empty_like(self);
    setTsValue(return_tensor, return_value);

    return return_tensor;
};

at::Tensor LazyNativeFunctions::relu(const at::Tensor & self) {
    LAZY_PERF_SCOPE("LazyNativeFunctions::relu");
    auto &gm = GraphManager::GetSingleton();
    
    auto self_ts_value = gm.getTsValue(self);

    auto return_value = gm->makeTsNode(c10::Symbol::fromQualString("aten::relu"), {self_ts_value});

    auto return_tensor = at::empty_like(self);
    setTsValue(return_tensor, return_value);

    return return_tensor;
};

at::Tensor LazyNativeFunctions::mul(const at::Tensor & self, const at::Tensor & other) {
    LAZY_PERF_SCOPE("LazyNativeFunctions::add");

    auto self_ts_value = gm.getTsValue(self);
    auto other_ts_value = gm.getTsValue(other);

    auto return_value = gm->makeTsNode(c10::Symbol::fromQualString("aten::mul"), {self_ts_value, other_ts_value});

    auto return_tensor = at::empty_like(self);
    setTsValue(return_tensor, return_value);

    return return_tensor;
};

at::Tensor& LazyNativeFunctions::mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor& out) {
    LAZY_PERF_SCOPE("LazyNativeFunctions::add_out");
    auto &gm = GraphManager::GetSingleton();
    
    auto self_ts_value = gm.getTsValue(self);
    auto other_ts_value = gm.getTsValue(other);
    auto out_ts_value = gm.getTsValue(other);

    auto return_value = gm->makeTsNode(c10::Symbol::fromQualString("aten::mul.out"), {self_ts_value, other_ts_value, out_ts_value});

    setTsValue(out, return_value);

    return out; 
};
} // namespace lazy_mode
