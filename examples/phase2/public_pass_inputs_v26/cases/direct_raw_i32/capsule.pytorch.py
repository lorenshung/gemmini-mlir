"""Fresh explicit public integer source; no model/timing qualification."""
import torch

def _saturate_i8(value):
    lower = torch.maximum(value, torch.full_like(value, -128))
    return torch.minimum(lower, torch.full_like(value, 127)).to(torch.int8)

# Declaration: a95fb0c58bed501a763d1adae3a2b51aa7fab435b4e8d1c17fb2cadf6e21fc54
class Program(torch.nn.Module):
    def forward(self, lhs,rhs):
        wide = torch._int_mm(lhs, rhs)
        return (wide,)

def get_model_and_inputs():
    lhs = (((torch.arange(272,dtype=torch.int64)*1+0)%17)-8).to(torch.int8).reshape([17, 16])
    rhs = (((torch.arange(512,dtype=torch.int64)*3+1)%17)-8).to(torch.int8).reshape([16, 32])
    return Program().eval(), (lhs,rhs,)
