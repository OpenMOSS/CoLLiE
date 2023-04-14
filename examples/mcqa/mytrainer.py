import tqdm

import torch
from torch.nn import CrossEntropyLoss
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from collie.trainer import InplaceTensorTrainer

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
                        shift_labels.view(-1)).view_as(shift_labels)  # [batch_size * n_choices, seq_len]

        if self.collie_args.length_normalization:
            loss = loss.mean(dim=1)
        else:
            loss = loss.sum(dim=1)
        # loss: [batch_size * n_choices]
        if self.collie_args.unconditional_normalization:
            un_input_ids = batch['input_ids'].masked_fill(batch['un_mask'], self.tokenizer.pad_token_id)
            un_attention_mask = batch['attention_mask'].masked_fill(batch['un_mask'], 0)
            un_logits = self.model(un_input_ids, un_attention_mask)
            # Shift so that tokens < n predict n
            shift_un_logits = un_logits[..., :-1, :].contiguous()
            # Flatten the tokens
            un_loss = loss_fct(shift_un_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                               shift_labels.view(-1)).view_as(shift_labels)  # [batch_size * n_choices, seq_len]

            if self.collie_args.length_normalization:
                un_loss = un_loss.mean(dim=1)
            else:
                un_loss = un_loss.sum(dim=1)

            loss = loss - un_loss

        group_loss = loss.split(batch['split_size'])
        preds = torch.stack([torch.argmin(l) for l in group_loss], dim=0)

        preds = nested_numpify(preds)
        return preds.tolist()
