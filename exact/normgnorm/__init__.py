from pathlib import Path
from typing import Tuple

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

def exists(x):
    return x is not None

current_dir = Path(__file__).parent.resolve()
normgnorm_cuda = load(
    name="normgnorm_cuda",
    sources=[
        str(current_dir / Path("csrc/normgnorm_cuda.cu")),
        str(current_dir / Path("csrc/normgnorm_cuda.cpp")),
    ],
)


class PEGLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_upegsqnorm, bias_upegsqnorm, normalized_shape, eps):
        assert len(input.shape) == 3
        assert len(normalized_shape) == 1
        assert input.size(-1) == normalized_shape[0]

        input_shape = input.size()
        output, mean, rstd = layernorm_fwd(input.reshape(-1, normalized_shape[0]), weight, bias, eps)
        output = output.reshape(input_shape)
        mean = mean.reshape(input_shape[:-1])
        rstd = rstd.reshape(input_shape[:-1])
        ctx.save_for_backward(input, weight, mean, rstd, weight_upegsqnorm, bias_upegsqnorm)
        return output

    @staticmethod
    def backward(ctx, grad_input):
        input, weight, mean, rstd, weight_upegsqnorm, bias_upegsqnorm = ctx.saved_tensors

        grad_output, pe_grad_weight, pe_grad_bias = layernorm_bwd(
            grad_input, input, weight, mean, rstd
        )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        grad_weight = pe_grad_weight.sum(0)
        grad_bias = pe_grad_bias.sum(0)

        with torch.cuda.stream(s):

            weight_pegnorm = torch.norm(pe_grad_weight, p=2, dim=-1)
            bias_pegnorm = torch.norm(pe_grad_bias, p=2, dim=-1)
            weight_pegsqnorm = weight_pegnorm ** 2
            bias_pegsqnorm = bias_pegnorm ** 2
            _weight_upegsqnorm = torch.stack([weight_pegsqnorm, torch.ones_like(weight_pegsqnorm)], -1).sum(0)
            _bias_upegsqnorm = torch.stack([bias_pegsqnorm, torch.ones_like(bias_pegsqnorm)], -1).sum(0)

        pe_grad_weight.record_stream(s)
        pe_grad_bias.record_stream(s)

        # update buffers
        if exists(weight_upegsqnorm):
            weight_upegsqnorm.add_(_weight_upegsqnorm)
        if exists(bias_upegsqnorm):
            bias_upegsqnorm.add_(_bias_upegsqnorm)

        #return grad_output, grad_weight, grad_bias, weight_pegsqnorm, bias_pegsqnorm, None, None
        return grad_output, grad_weight, grad_bias, None, None, None, None


def layernorm_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return normgnorm_cuda.layernorm_fwd(x, weight, bias, eps)


def layernorm_bwd(grad_input: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return normgnorm_cuda.layernorm_bwd(grad_input, input, weight, mean, rstd)


if __name__ == '__main__':

    x = torch.randn(4, 4096, 4096, requires_grad=True).cuda()
    weight = torch.randn(4096, requires_grad=True).cuda()
    bias = torch.zeros(4096, requires_grad=True).cuda()
    weight_pegsqnorm = torch.zeros(2, requires_grad=True).cuda()
    bias_pegsqnorm = torch.zeros(2, requires_grad=True).cuda()

    PEGLayerNorm.apply(x, weight, bias, weight_pegsqnorm, bias_pegsqnorm, [4096], 1e-5)
    import time

    out_gt = F.layer_norm(x, [4096], weight=weight, bias=bias, eps=1e-5)
    out_ours = PEGLayerNorm.apply(x, weight, bias, weight_pegsqnorm, bias_pegsqnorm, [4096], 1e-5)
    assert torch.allclose(out_gt, out_ours, atol=1e-5, rtol=1e-5)

    min_dt = float('inf')
    sum_dt = 0
    for i in range(100):
        t0 = time.time()
        out = F.layer_norm(x, [4096], weight, bias, 1e-5)
        out.sum().backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        min_dt = min(min_dt, dt)
        sum_dt += dt

    print(f"Time LN: min: {min_dt * 1000:.3f} ms, avg: {sum_dt * 1000 / 100:.3f} ms")


    x.grad = None
    weight.grad = None
    bias.grad = None
    weight_pegsqnorm.grad = None
    bias_pegsqnorm.grad = None


    min_dt = float('inf')
    sum_dt = 0
    for i in range(100):
        t0 = time.time()
        out = PEGLayerNorm.apply(x, weight, bias, weight_pegsqnorm, bias_pegsqnorm, [4096], 1e-5)
        out.sum().backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        min_dt = min(min_dt, dt)
        sum_dt += dt
    print(f"Time PEGLN: min: {min_dt * 1000:.3f} ms, avg: {sum_dt * 1000 / 100:.3f} ms")

