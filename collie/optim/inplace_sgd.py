import os
import torch
from torch.optim import Optimizer
import torch.distributed as dist


class InplaceSGD(Optimizer):
    """
    一个自定义的优化器类InplaceSGD，用于在分布式训练中的梯度更新。

    该类实现两个梯度更新函数 :meth:`inplace_sgd` 和 :meth:`inplace_sgd_zero3`，分别用于非ZeRO和ZeRO模式下的梯度更新。
    
    :param model: 待优化的模型
    :param lr: 学习率，默认值为1e-3
    :param zero_enabled: 是否开启ZeRO，默认值是 ``False``，表示不开启ZeRO；否则开启ZeRO
    :param clip_grad_norm: 梯度裁剪的范数阈值

        .. note::

            clip_grad_norm须为正数

    :param clip_grad_value: 梯度裁剪的值域阈值
    """
    def __init__(self, model, lr=1e-3, zero_enabled=False, clip_grad_norm=None, clip_grad_value=None):
        self.model = model
        self.lr = lr
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.zero_enabled = zero_enabled
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(f"clip_grad_norm should be positive, got {self.clip_grad_norm}.")
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        # register hook
        self.grad_func = self.inplace_sgd() if not self.zero_enabled else self.inplace_sgd_zero3()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(lr=lr, zero_enabled=zero_enabled, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)
        super(InplaceSGD, self).__init__(self.model.parameters(), defaults)

    def inplace_sgd(self):
        """
        在非ZeRO模式下更新模型参数的梯度。

        :return: func，一个闭包函数，用于更新模型参数的梯度
        """
        def func(x):
            """
            闭包函数，用于更新模型参数的梯度。
            """
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if self.gather_norm:
                            self.grad_norms.append(torch.norm(p.grad, 2.0))
                            p.grad = None
                        else:
                            if self.clip_grad_value is not None and self.clip_grad_value > 0:
                                # Gradients are modified in-place.
                                p.grad.data.clamp_(min=-self.clip_grad_value,
                                                   max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                p.grad.data.mul_(self.clip_coef)
                            p.data -= (self.lr * p.grad.data)
                            p.grad = None
            return x

        return func

    def inplace_sgd_zero3(self):
        """
        在ZeRO模式下更新模型参数的梯度。
        
        :return: func，一个闭包函数，用于更新模型参数的梯度。
        """
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                        if self.gather_norm:
                            self.grad_norms.append(torch.norm(p.grad, 2.0))
                            p.grad = None
                        else:
                            one_dim_grad = p.grad.view(-1)
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.local_rank
                            end = start + partition_size

                            if end > p.grad.numel():
                                partitioned_grad = one_dim_grad.narrow(0, start, p.grad.numel() - start)
                                # partitioned_grad = torch.cat([partitioned_grad, torch.zeros(end - p.grad.numel()).cuda()])
                                partitioned_p = p.ds_tensor.narrow(0, 0, p.grad.numel() - start)
                                if self.clip_grad_value is not None:
                                    # Gradients are modified in-place.
                                    partitioned_grad.clamp_(min=-self.clip_grad_value,
                                                            max=self.clip_grad_value)
                                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                    partitioned_grad.mul_(self.clip_coef)
                                partitioned_p -= (self.lr * partitioned_grad)
                            else:
                                partitioned_grad = one_dim_grad.narrow(0, start, partition_size)
                                if self.clip_grad_value is not None:
                                    # Gradients are modified in-place.
                                    partitioned_grad.clamp_(min=-self.clip_grad_value,
                                                            max=self.clip_grad_value)
                                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                    partitioned_grad.mul_(self.clip_coef)
                                p.ds_tensor -= (self.lr * partitioned_grad)
                            p.grad = None
            return x

        return func

    def backward_step(self, loss, lr):
        """
        执行一步反向传播更新模型的梯度。
        
        :param loss: 模型的loss值
        :param lr: 学习率
        """
        self.lr = lr
        # User need call grad_norm themselves and then call backward_step
        # if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
        #     self.grad_norm(loss)
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is None:
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm() before backward_step."
            )
        loss.backward()
        # update the last one since the hook function will not be called for the last parameter
        self.grad_func(0)

    def grad_norm(self, loss):
        """
        计算梯度的范数。
        
        :param loss: 模型的loss值
        """
        self.gather_norm = True
        self.grad_norms = []

        loss.backward(retain_graph=True)
        # update the last one since the hook function will not be called for the last parameter
        self.grad_func(0)

        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)
            device = torch.device(f"cuda:{self.local_rank}")
            all_grad_norms = torch.zeros(self.world_size * self.grad_norms.shape[0],
                                         dtype=self.grad_norms.dtype, device=device)
            torch.distributed.all_gather_into_tensor(all_grad_norms, self.grad_norms)

            total_norm = torch.norm(all_grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False
