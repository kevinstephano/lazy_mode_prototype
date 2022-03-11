#pragma once

#include<unordered_map>
#include<torch/csrc/jit/ir/ir.h>

namespace lazy_mode {

class GraphManager {
    GraphManager() = default;

    // There should only be one instance of this class
    GraphManager(const GraphManager&) = delete;
    GraphManager& operator=(const GraphManager&) = delete;
    GraphManager(GraphManager&&) = delete;
    GraphManager& operator=(GraphManager&&) = delete;

  public:
    static GraphManager&  GetSingleton() {
      static GraphManager singleton;
      return singleton;
    }

  private:
    std::unordered_map<std::string, torch::jit::GraphExecutor> graph_cache;
    std::unordered_map<at::Tensor, torch::jit::Value> tensor_to_ir_value_map;

    torch::jit::Graph current_graph;
};

} // nameespace lazy_mode
