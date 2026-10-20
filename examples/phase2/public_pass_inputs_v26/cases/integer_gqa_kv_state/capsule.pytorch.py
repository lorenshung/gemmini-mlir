"""Public source legality controls; no physical scheduling qualification."""
import torch

def _saturate_i8(value):
    lower = torch.maximum(value, torch.full_like(value, -128))
    return torch.minimum(lower, torch.full_like(value, 127)).to(torch.int8)

# Declaration: a333dbb362796e4b7c6768029166c6f62cdc23bafbe0e84ec88c3bd90bfcb046
class Program(torch.nn.Module):
    def forward(self,prefix_k,prefix_v,q0,k0,v0,q1,k1,v1):
        cache_k = prefix_k
        cache_v = prefix_v
        cache_k0 = torch.cat((cache_k, k0), dim=1)
        cache_v0 = torch.cat((cache_v, v0), dim=1)
        scores_0_0 = torch._int_mm(q0[0].contiguous(), cache_k0[0].transpose(0, 1).contiguous())
        readout_0_0 = _saturate_i8(torch.div(scores_0_0 + 4, 8, rounding_mode='floor'))
        mixed_0_0 = torch._int_mm(readout_0_0, cache_v0[0].contiguous())
        scores_0_1 = torch._int_mm(q0[1].contiguous(), cache_k0[0].transpose(0, 1).contiguous())
        readout_0_1 = _saturate_i8(torch.div(scores_0_1 + 4, 8, rounding_mode='floor'))
        mixed_0_1 = torch._int_mm(readout_0_1, cache_v0[0].contiguous())
        scores_0_2 = torch._int_mm(q0[2].contiguous(), cache_k0[0].transpose(0, 1).contiguous())
        readout_0_2 = _saturate_i8(torch.div(scores_0_2 + 4, 8, rounding_mode='floor'))
        mixed_0_2 = torch._int_mm(readout_0_2, cache_v0[0].contiguous())
        cache_k = cache_k0
        cache_v = cache_v0
        cache_k1 = torch.cat((cache_k, k1), dim=1)
        cache_v1 = torch.cat((cache_v, v1), dim=1)
        scores_1_0 = torch._int_mm(q1[0].contiguous(), cache_k1[0].transpose(0, 1).contiguous())
        readout_1_0 = _saturate_i8(torch.div(scores_1_0 + 4, 8, rounding_mode='floor'))
        mixed_1_0 = torch._int_mm(readout_1_0, cache_v1[0].contiguous())
        scores_1_1 = torch._int_mm(q1[1].contiguous(), cache_k1[0].transpose(0, 1).contiguous())
        readout_1_1 = _saturate_i8(torch.div(scores_1_1 + 4, 8, rounding_mode='floor'))
        mixed_1_1 = torch._int_mm(readout_1_1, cache_v1[0].contiguous())
        scores_1_2 = torch._int_mm(q1[2].contiguous(), cache_k1[0].transpose(0, 1).contiguous())
        readout_1_2 = _saturate_i8(torch.div(scores_1_2 + 4, 8, rounding_mode='floor'))
        mixed_1_2 = torch._int_mm(readout_1_2, cache_v1[0].contiguous())
        cache_k = cache_k1
        cache_v = cache_v1
        return (scores_0_0,mixed_0_0,scores_0_1,mixed_0_1,scores_0_2,mixed_0_2,cache_k0,cache_v0,scores_1_0,mixed_1_0,scores_1_1,mixed_1_1,scores_1_2,mixed_1_2,cache_k1,cache_v1,)

def get_model_and_inputs():
    prefix_k = (((torch.arange(49,dtype=torch.int64)*1+0)%17)-8).to(torch.int8).reshape([1, 7, 7])
    prefix_v = (((torch.arange(49,dtype=torch.int64)*3+1)%17)-8).to(torch.int8).reshape([1, 7, 7])
    q0 = (((torch.arange(21,dtype=torch.int64)*5+2)%17)-8).to(torch.int8).reshape([3, 1, 7])
    k0 = (((torch.arange(7,dtype=torch.int64)*7+3)%17)-8).to(torch.int8).reshape([1, 1, 7])
    v0 = (((torch.arange(7,dtype=torch.int64)*9+4)%17)-8).to(torch.int8).reshape([1, 1, 7])
    q1 = (((torch.arange(21,dtype=torch.int64)*11+5)%17)-8).to(torch.int8).reshape([3, 1, 7])
    k1 = (((torch.arange(7,dtype=torch.int64)*13+6)%17)-8).to(torch.int8).reshape([1, 1, 7])
    v1 = (((torch.arange(7,dtype=torch.int64)*15+7)%17)-8).to(torch.int8).reshape([1, 1, 7])
    return Program().eval(), (prefix_k,prefix_v,q0,k0,v0,q1,k1,v1,)
