
import torch

class GatedChoiceNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, v):
        ctx.save_for_backward(g, v)
        return g * v

    @staticmethod
    def backward(ctx, grad_output):
        g, v = ctx.saved_tensors

        return (
            grad_output * v,
            grad_output * g
        )

class GatedChoiceGateFreeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, v):
        ctx.save_for_backward(g, v)
        return g * v

    @staticmethod
    def backward(ctx, grad_output):
        g, v = ctx.saved_tensors

        return (
            grad_output * v,
            grad_output
        )

def gated_choice(g, a, m, mode='normal'):
    if mode == 'normal':
        return g * a + (1 - g) * m
    elif mode == 'gate-free-gradient':
        return GatedChoiceGateFreeGradient.apply(g, a) + GatedChoiceGateFreeGradient.apply(1 - g, m)
    elif mode == 'test':
        return GatedChoiceNormal.apply(g, a) + GatedChoiceNormal.apply(1 - g, m)
    else:
        raise NotADirectoryError(f'the mode {mode} is not implemented')
