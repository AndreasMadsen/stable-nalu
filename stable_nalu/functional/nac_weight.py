
import torch

class NACWeight(torch.autograd.Function):
    r"""Implements the NAC weight operator

    w = tanh(\hat{w}) * sigmoid(\hat{m})
    """

    @staticmethod
    def forward(ctx, w_hat, m_hat):
        tanh_w_hat = torch.tanh(w_hat)
        sigmoid_m_hat = torch.sigmoid(m_hat)
        ctx.save_for_backward(tanh_w_hat, sigmoid_m_hat)
        return tanh_w_hat * sigmoid_m_hat

    @staticmethod
    def backward(ctx, grad_output):
        tanh_w_hat, sigmoid_m_hat = ctx.saved_tensors

        return (
            grad_output * (1 - tanh_w_hat*tanh_w_hat)*sigmoid_m_hat,
            grad_output * tanh_w_hat*sigmoid_m_hat*(1-sigmoid_m_hat)
        )

class NACWeightRescaled(torch.autograd.Function):
    r"""Implements the NAC weight operator but with a hard gradient for \hat{m}

    w = tanh(\hat{w}) * sigmoid(\hat{m})
    dL/d\hat{m} = (dL/dw) (dw/d\hat{m})
                = (dL/dw) * 0.1 * sign(\hat{w}) * sigmoid(\hat{m}) * (1 - sigmoid(\hat{m}))
    """
    @staticmethod
    def forward(ctx, w_hat, m_hat):
        tanh_w_hat = torch.tanh(w_hat)
        sigmoid_m_hat = torch.sigmoid(m_hat)
        ctx.save_for_backward(w_hat, tanh_w_hat, sigmoid_m_hat)
        return tanh_w_hat * sigmoid_m_hat

    @staticmethod
    def backward(ctx, grad_output):
        w_hat, tanh_w_hat, sigmoid_m_hat = ctx.saved_tensors

        return (
            grad_output * (1 - tanh_w_hat*tanh_w_hat)*sigmoid_m_hat,
            grad_output * 0.1*torch.sign(w_hat)*sigmoid_m_hat*(1-sigmoid_m_hat)
        )

class NACWeightIndepdent(torch.autograd.Function):
    r"""Implements the NAC weight operator but with independent optimization.

    The optimiation of \hat{w} is indepdent of \hat{m} and vice versa.

    w = tanh(\hat{w}) * sigmoid(\hat{m})

    dL/d\hat{w} = (dL/dw) (dw/d\hat{w})
                = (dL/dw) (1 - tanh(\hat{w})^2)

    dL/d\hat{m} = (dL/dw) (dw/d\hat{m})
                = (dL/dw) sigmoid(\hat{m}) * (1 - sigmoid(\hat{m}))
    """
    @staticmethod
    def forward(ctx, w_hat, m_hat):
        tanh_w_hat = torch.tanh(w_hat)
        sigmoid_m_hat = torch.sigmoid(m_hat)
        ctx.save_for_backward(tanh_w_hat, sigmoid_m_hat)
        return tanh_w_hat * sigmoid_m_hat

    @staticmethod
    def backward(ctx, grad_output):
        tanh_w_hat, sigmoid_m_hat = ctx.saved_tensors

        return (
            grad_output * (1 - tanh_w_hat*tanh_w_hat),
            grad_output * sigmoid_m_hat*(1-sigmoid_m_hat)
        )
