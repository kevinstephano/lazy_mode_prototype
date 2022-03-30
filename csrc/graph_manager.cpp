#include "graph_manager.h"
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <iostream>

namespace lazy_mode {

torch::jit::Value* GraphManager::getTsValue(const at::Tensor& aten_tensor) {
  LAZY_PERF_SCOPE("GraphManager::getTsValue");
  torch::jit::Value* result = nullptr;
  TORCH_CHECK(aten_tensor.defined(), "aten_tensor not defined!");

  // If Tensor does not have an IR Value, assume it is an input to the graph
  if (tensor_to_value_map_.count(aten_tensor.getIntrusivePtr()) == 0) {
    result = ts_graph_->addInput();
    result->inferTypeFrom(aten_tensor);
    setTsValue(aten_tensor, result);
    graph_inputs_.push_back(aten_tensor);
  } else {
    result = tensor_to_value_map_[aten_tensor.getIntrusivePtr()];
  }
  return result;
}

void GraphManager::setTsValue(const at::Tensor& aten_tensor, torch::jit::Value* ts_value) {
  LAZY_PERF_SCOPE("GraphManager::setTsValue");
  TORCH_CHECK(aten_tensor.defined(), "aten_tensor not defined!");
  tensor_to_value_map_[aten_tensor.getIntrusivePtr()] = ts_value;
}

void GraphManager::createTsGraph() {
  if (not ts_graph_) {
    LAZY_PERF_SCOPE("GraphManager::createTsGraph"); // purposely put inside the "if"
    ts_graph_ = std::make_shared<torch::jit::Graph>();
    ts_graph_function_ = std::make_shared<torch::jit::GraphFunction>(func_name_, ts_graph_, nullptr);
  }
}

void GraphManager::resetTsGraph() {
  LAZY_PERF_SCOPE("GraphManager::resetTsGraph");
  if (ts_graph_) {
    ts_graph_.reset();
    ts_graph_function_.reset();
    tensor_to_value_map_.clear();
  }
}

// Adds a TorchScript Node to the graph and returns the node's output IR Value.
torch::jit::Value* GraphManager::makeTsNode(c10::Symbol sym, const std::vector<torch::jit::NamedValue>& args) {
  LAZY_PERF_SCOPE("GraphManager::makeTsNode");
  auto builtin =
      std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
  auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
  auto ret = magic_method->call({}, *ts_graph_function_, args, {}, 0);
  auto sv = dynamic_cast<torch::jit::SimpleValue*>(ret.get());
  TORCH_CHECK(sv, "SimpleValue* pointer doesn't exist!");
  //TORCH_CHECK(sv->getValue()->type()->kind() == c10::TypeKind::TupleType, "Tuples are not supported!");
  if (sv->getValue()->type()->kind() == c10::TypeKind::TupleType) {
    const auto tuple_call_result = sv->asTuple({}, *ts_graph_function_);
    std::vector<torch::jit::Value*> tuple_result;
    for (const auto& tuple_component : tuple_call_result) {
      auto tuple_component_sv =
          dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
      tuple_result.push_back(tuple_component_sv->getValue());
    }
    TORCH_CHECK(tuple_result.size() == 1, "Only allow size 1 vectors.");
    return tuple_result[0];
  } else {
  	return sv->getValue();
  }
}

void GraphManager::executeTsGraph() {
  LAZY_PERF_SCOPE("GraphManager::executeTsGraph");
  for (const auto& entry : tensor_to_value_map_) {
    if (!entry.second->hasUses()) {
      ts_graph_->registerOutput(entry.second);
      graph_outputs_.push_back(entry.first);
    }
  }
  printTsGraph();

}

void GraphManager::printTsGraph() const {
  TORCH_INTERNAL_ASSERT(ts_graph_, "Graph is null!");
  ts_graph_->print(std::cout);
}

} // nameespace lazy_mode
