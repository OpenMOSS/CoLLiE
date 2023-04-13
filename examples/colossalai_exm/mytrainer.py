from itertools import cycle

import torch
from torch.nn import CrossEntropyLoss
from transformers.trainer_pt_utils import nested_numpify

try:
    from colossalai.core import global_context as gpc
    from colossalai.context.parallel_mode import ParallelMode
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")

from collie.trainer.colossalai_trainer import ColossalaiTrainer

IGNORE_INDEX = -100


class MyColossalaiTrainer(ColossalaiTrainer):
    def generate(self, batch, max_length, stop_tokens):
        # 此处 batch 的数目是 batch_Size * 4（一般情况下）
        # 这是因为在 data collator 里进行了处理，将 source 和四个
        # 选项分别进行了连接，使得数据量变大
        self.engine.eval()
        input_ids = batch[0]["input_ids"]
        logits, label, _ = self.engine.execute_schedule(
            cycle([({
                "input_ids": input_ids,
                "use_cache": torch.ones(input_ids.shape[0], dtype=torch.bool)
            }, input_ids)]),
            forward_only=True,
            return_loss=False,
            return_output_label=True,
        )

        preds = torch.zeros(len(batch[0]["split_size"])).cuda()
        if gpc.is_pipeline_last_stage():
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[1][..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                            shift_labels.view(-1)).view_as(shift_labels)
            loss = loss.mean(dim=1)
            # 将 loss 按照 Split_size 分组，得到每一组问题的 loss
            # 进而选出 answer
            group_loss = loss.split(batch[0]['split_size'])
            preds = torch.stack([torch.argmin(l) for l in group_loss], dim=0)
        torch.distributed.broadcast(preds, src=gpc.get_world_size(ParallelMode.PIPELINE) - 1)

        preds = nested_numpify(preds)
        return preds.tolist()
