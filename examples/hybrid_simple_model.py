import sys
import torch

import lazy_mode

torch._C._jit_set_nvfuser_horizontal_mode(False)

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(1024,1024))
        self.act = torch.nn.ReLU()

    def forward(self, inputs) :
        out1 = torch.mm(inputs, self.weights)
        out2 = self.act(out1)
        return torch.sqrt(out2)

if __name__ == "__main__" :
    device = 'cuda'
    dtype = torch.float
    steps = 20
    batches = [[torch.randn(5120, 1024, dtype=dtype, device=device)] for _ in range(steps)]

    lazy_model = TestModule().to(device=device)
    eager_model = TestModule().to(device=device)

    with torch.autograd.profiler.emit_nvtx(enabled=True):
        with torch.jit.fuser('fuser2') :
            for step,batch in enumerate(batches) :
                with torch.inference_mode() :
                    torch.cuda.nvtx.range_push("Lazy-Mode") 
                    with lazy_mode.lazy_execute() :
                        lazy_out = lazy_model(*batch)
                    torch.cuda.nvtx.range_pop()
 
                    torch.cuda.nvtx.range_push("Eager-Mode") 
                    eager_out = eager_model(lazy_out)
                    torch.cuda.nvtx.range_pop()

