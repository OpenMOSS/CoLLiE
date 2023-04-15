from itertools import cycle

import torch

try:
    from colossalai.core import global_context as gpc
    from colossalai.context.parallel_mode import ParallelMode
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Detected Colossal-AI is not installed. See https://github.com/hpcaitech/ColossalAI")

from collie.trainer.colossalai_trainer import ColossalaiTrainer

class MyColossalaiTrainer(ColossalaiTrainer):

    def generate(self, input_ids, max_length, stop_tokens, use_cache):
        # 此处 batch 的数目是 batch_size * 4（一般情况下）
        # 这是因为在 data collator 里进行了处理，将 source 和四个
        # 选项分别进行了连接，使得数据量变大
        self.engine.eval()
        logits, label, _ = self.engine.execute_schedule(
            cycle([({
                "input_ids": input_ids,
                "use_cache": torch.zeros(input_ids.shape[0], dtype=torch.bool)
                }, input_ids)]),
            forward_only=True,
            return_loss=False,
            return_output_label=True,
        )
        torch.cuda.empty_cache()

        return logits
