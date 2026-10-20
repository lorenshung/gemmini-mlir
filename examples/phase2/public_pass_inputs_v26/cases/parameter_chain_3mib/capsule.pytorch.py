"""Public deterministic parameter compilation stress; no performance admission."""
import torch

def _saturate_i8(value):
    lower = torch.maximum(value, torch.full_like(value, -128))
    return torch.minimum(lower, torch.full_like(value, 127)).to(torch.int8)

# Declaration: ac2730e80c597d7e247bb199ab87c6bfa62bc51c79a21ac4c7c02e5de8d9e20c
class Program(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w0 = (((torch.arange(1048576, dtype=torch.int64) + 1) % 17) - 8).to(torch.int8).reshape(1024, 1024)
        self.register_parameter('w0', torch.nn.Parameter(w0, requires_grad=False))
        w1 = (((torch.arange(1048576, dtype=torch.int64) + 4) % 17) - 8).to(torch.int8).reshape(1024, 1024)
        self.register_parameter('w1', torch.nn.Parameter(w1, requires_grad=False))
        w2 = (((torch.arange(1048576, dtype=torch.int64) + 7) % 17) - 8).to(torch.int8).reshape(1024, 1024)
        self.register_parameter('w2', torch.nn.Parameter(w2, requires_grad=False))
    def forward(self, x):
        state = x
        wide = torch._int_mm(state, self.w0)
        state = _saturate_i8(torch.div(wide + 4, 8, rounding_mode='floor'))
        wide = torch._int_mm(state, self.w1)
        state = _saturate_i8(torch.div(wide + 4, 8, rounding_mode='floor'))
        wide = torch._int_mm(state, self.w2)
        state = _saturate_i8(torch.div(wide + 4, 8, rounding_mode='floor'))
        return wide, state

def get_model_and_inputs():
    x = (((torch.arange(1024, dtype=torch.int64)) % 17) - 8).to(torch.int8).reshape(1, 1024)
    return Program().eval(), (x,)
