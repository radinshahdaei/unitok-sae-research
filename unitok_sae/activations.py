"""JumpReLU's exact forward and deliberately surrogate threshold derivatives.

Follows Rajamanoharan et al., arXiv:2407.14435v3, Eqs. 11/12 and Appendix J
(v1/v2 pseudocode had corrected bugs). The rectangular KDE window is open at
both boundaries. Input derivatives are ordinary derivatives away from the jump;
only threshold derivatives are replaced. These functions will intentionally fail
finite-difference gradcheck at the jump: a surrogate is not the true derivative.
"""

import torch


class _Jump(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positive, threshold, bandwidth):
        ctx.save_for_backward(positive, threshold)
        ctx.bandwidth = bandwidth
        return positive * (positive > threshold)

    @staticmethod
    def backward(ctx, upstream):
        positive, threshold = ctx.saved_tensors
        window = ((positive - threshold).abs() < ctx.bandwidth / 2).to(positive.dtype)
        input_grad = upstream * (positive > threshold)
        threshold_grad = -upstream * threshold * window / ctx.bandwidth
        return input_grad, threshold_grad.sum_to_size(threshold.shape), None


class _Count(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positive, threshold, bandwidth):
        ctx.save_for_backward(positive, threshold)
        ctx.bandwidth = bandwidth
        return (positive > threshold).to(positive.dtype)

    @staticmethod
    def backward(ctx, upstream):
        positive, threshold = ctx.saved_tensors
        window = ((positive - threshold).abs() < ctx.bandwidth / 2).to(positive.dtype)
        threshold_grad = -upstream * window / ctx.bandwidth
        return (
            torch.zeros_like(positive),
            threshold_grad.sum_to_size(threshold.shape),
            None,
        )


def jump_relu(positive, threshold, *, bandwidth):
    """Gate nonnegative preactivations with learned positive feature thresholds.

    Pass ``threshold = log_threshold.exp()`` to learn in log space; autograd adds
    that chain-rule factor automatically. bandwidth must be a positive scalar.
    The SAE config validates it. Caller supplies nonnegative preactivations,
    matching Appendix J's ReLU preprocessing before the surrogate operation.
    """
    return _Jump.apply(positive, threshold, bandwidth)


def jump_count(positive, threshold, *, bandwidth):
    """Exact active indicators with zero input gradient and KDE threshold gradient."""
    return _Count.apply(positive, threshold, bandwidth)
