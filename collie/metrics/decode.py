from typing import Any, Dict
from collie.metrics.base import BaseMetric
from collie.utils import env
from collie.log.print import print
import torch

class DecodeMetric(BaseMetric):
    def __init__(self, 
                 tokenizer: Any,
                 verbose: bool = True,
                 save_to_file: bool = False,
                 save_path: str = None,
                 gather_result: bool = True) -> None:
        super().__init__(gather_result)
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.save_path = save_path
        self.tokenizer = tokenizer
    
    def get_metric(self):
        return None

    def update(self, result: Dict):
        # 合并数据
        # if isinstance(result, list):
            # input_ids = [r['input_ids'] for r in result]
        # else:
        input_ids = result['input_ids']
        decode_list = []
        for i in range(len(input_ids)):
            if isinstance(input_ids[i], torch.Tensor):
                if input_ids[i].ndim == 2:
                    decode_list.extend(list(map(lambda x: x.detach().cpu().tolist(), [*input_ids[i]])))
                else:
                    decode_list.append(input_ids[i].detach().cpu().tolist())
            else:
                decode_list.append(input_ids[i])
        sentences = []
        for ids in decode_list:
            sentences.append(self.tokenizer.decode(ids))
        if env.pp_rank == env.pp_size - 1 \
            and env.tp_rank == env.tp_size - 1 \
                and (env.dp_rank == 0 or self.gather_result):
                    if self.verbose:
                        print(sentences)
                    if self.save_to_file:
                        with open(self.save_path, 'a+') as f:
                            f.write('\n'.join(sentences) + '\n')
