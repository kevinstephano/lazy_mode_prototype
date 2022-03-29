#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "graph_manager.h"

namespace lazy_mode {

void InitPythonBindings(py::module m) {
  m.def("_enable_lazy_mode", []() {
    c10::impl::tls_set_dispatch_key_excluded(c10::DispatchKey::CUDA, true);
    c10::impl::tls_set_dispatch_key_excluded(c10::DispatchKey::AutogradCUDA, true);
    c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::Lazy, true);
    c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::AutogradLazy, true);
  });
  m.def("_disable_lazy_mode", []() {
    c10::impl::tls_set_dispatch_key_excluded(c10::DispatchKey::CUDA, false);
    c10::impl::tls_set_dispatch_key_excluded(c10::DispatchKey::AutogradCUDA, false);
    c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::Lazy, false);
    c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::AutogradLazy, false);
    auto &gm = GraphManager::GetSingleton();
    gm.printTsGraph();
    gm.resetTsGraph();
  });
}

} // namespace lazy_mode


PYBIND11_MODULE(_LAZY, m) {
  lazy_mode::InitPythonBindings(m);
}
