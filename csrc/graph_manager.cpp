#include "graph_manager.h"
#include <ATen/core/Tensor.h>
#include <iostream>

namespace lazy_mode {

torch::jit::Value* GraphManager::getTsValue(const at::Tensor& aten_tensor) {
  LAZY_PERF_SCOPE("GraphManager::getTsValue");
  torch::jit::Value* result = nullptr;
  TORCH_CHECK(aten_tensor.defined(), "aten_tensor not defined!");
  if (tensor_to_value_map_.count(aten_tensor.getIntrusivePtr()) == 0) {
    std::cout << "No existing tensor found, setting a new one!\n";
    result = ts_graph_->addInput();
    result->inferTypeFrom(aten_tensor);
    setTsValue(aten_tensor, result);
  } else {
    std::cout << "Existing tensor found!\n";
    result = tensor_to_value_map_[aten_tensor.getIntrusivePtr()];
  }
  return result;
}

void GraphManager::setTsValue(const at::Tensor& aten_tensor, torch::jit::Value* ts_value) {
  LAZY_PERF_SCOPE("GraphManager::setTsValue");
  TORCH_CHECK(aten_tensor.defined(), "aten_tensor not defined!");
  tensor_to_value_map_[aten_tensor.getIntrusivePtr()] = ts_value;
}

void GraphManager::createGraph() {
  if (not ts_graph_) {
    ts_graph_ = std::make_shared<torch::jit::Graph>();
    ts_graph_function_ = std::make_shared<torch::jit::GraphFunction>(func_name_, ts_graph_, nullptr);
  }
}

void GraphManager::resetGraph() {
  if (ts_graph_) {
    ts_graph_.reset();
    ts_graph_function_.reset();
  }
}

void GraphManager::print() const {
  TORCH_INTERNAL_ASSERT(ts_graph_, "Graph is null!");
  ts_graph_->print(std::cout);
}

} // nameespace lazy_mode
