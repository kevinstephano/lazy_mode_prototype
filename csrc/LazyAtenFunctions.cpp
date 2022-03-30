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

    auto return_value = gm.makeTsNode(c10::Symbol::fromQualString("aten::mm"), {self_ts_value, mat2_ts_value});

    auto return_tensor = at::empty_like(self);
    gm.setTsValue(return_tensor, return_value);

    return return_tensor;
};

at::Tensor LazyNativeFunctions::relu(const at::Tensor & self) {
    LAZY_PERF_SCOPE("LazyNativeFunctions::relu");
    auto &gm = GraphManager::GetSingleton();
    
    auto self_ts_value = gm.getTsValue(self);

    auto return_value = gm.makeTsNode(c10::Symbol::fromQualString("aten::relu"), {self_ts_value});

    auto return_tensor = at::empty_like(self);
    gm.setTsValue(return_tensor, return_value);

    return return_tensor;
};

at::Tensor LazyNativeFunctions::sqrt(const at::Tensor& self) {
    LAZY_PERF_SCOPE("LazyNativeFunctions::sqrt");
    auto &gm = GraphManager::GetSingleton();

    auto self_ts_value = gm.getTsValue(self);

    auto return_value = gm.makeTsNode(c10::Symbol::fromQualString("aten::sqrt"), {self_ts_value});

    auto return_tensor = at::empty_like(self);
    gm.setTsValue(return_tensor, return_value);

    return return_tensor;
};

} // namespace lazy_mode
