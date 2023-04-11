import tqdm

import torch
from torch.nn import CrossEntropyLoss
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from tunelite.trainer import InplaceTensorTrainer

IGNORE_INDEX = -100


class MyInplaceTensorTrainer(InplaceTensorTrainer):
    def eval_step(self, batch):
        logits = self.model(batch['input_ids'], batch['attention_mask'])
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                        shift_labels.view(-1)).view_as(shift_labels)
        loss = loss.mean(dim=1)
        group_loss = loss.split(batch['split_size'])
        preds = torch.stack([torch.argmin(l) for l in group_loss], dim=0)

        preds = nested_numpify(preds)
        return preds.tolist()
