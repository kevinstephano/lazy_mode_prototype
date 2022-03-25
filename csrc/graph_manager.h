#pragma once

#include<unordered_map>
#include<torch/csrc/api/include/torch/jit.h>
#include "instrumentation.h"

namespace lazy_mode {

class GraphManager {
    GraphManager() : 
      func_name_("LazyTSLowering"),
      ts_graph_exec_cache_(),
      ts_graph_(std::make_shared<torch::jit::Graph>()),
      ts_graph_function_(std::make_shared<torch::jit::GraphFunction>(
        func_name_, ts_graph_, nullptr))
    { } 

    // There should only be one instance of this class
    GraphManager(const GraphManager&) = delete;
    GraphManager& operator=(const GraphManager&) = delete;
    GraphManager(GraphManager&&) = delete;
    GraphManager& operator=(GraphManager&&) = delete;

  public:
    static GraphManager& GetSingleton() {
      LAZY_PERF_SCOPE("GraphManager::GetSingleton");
      static GraphManager singleton;
      return singleton;
    }

  private:
    std::string func_name_;
    std::unordered_map<std::string, torch::jit::GraphExecutor> ts_graph_exec_cache_;
    //std::unordered_map<at::Tensor, torch::jit::Value> tensor_to_ir_value_map;

    std::shared_ptr<torch::jit::Graph> ts_graph_;
    std::shared_ptr<torch::jit::GraphFunction> ts_graph_function_;
};

} // nameespace lazy_mode
