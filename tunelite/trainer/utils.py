import torch


# An approximation of in-place grad update
def inplace_grad(model, lr=5e-4):
    def func(x):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and p.shape != torch.Size([0]):
                    p.data -= (lr * p.grad.data)
                    p.grad = None
        return x

    return func
