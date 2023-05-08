from megatron.core import tensor_parallel

import torch

class ColumnParallelLinearWithoutBias(tensor_parallel.ColumnParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]
    
class RowParallelLinearWithoutBias(tensor_parallel.RowParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]
    
class GPTLMLoss(torch.nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)  # ignore <pad> when compute loss

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))