from typing import Any
from collie.metrics.base import BaseMetric
from collie.utils import env
import torch

class DecodeMetric(BaseMetric):
    def __init__(self, 
                 tokenizer: Any,
                 verbose: bool = True,
                 save_to_file: bool = False,
                 save_path: str = None,
                 gather_result: bool = False) -> None:
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
                    input_ids[i] = list(map(lambda x: x.detach().cpu().tolist(), [*input_ids[i]]))
                    decode_list.extend(input_ids[i])
                else:
                    input_ids[i] = input_ids[i].detach().cpu().tolist()
                    decode_list.append(input_ids[i])
            else:
                decode_list.append(input_ids[i])
        sentences = []
        for ids in decode_list:
            sentences.append(self.tokenizer.decode(ids))
        if self.verbose:
            print(sentences)
        if self.save_to_file and env.local_rank == 0:
            with open(self.save_path, 'a+') as f:
                f.write('\n'.join(sentences) + '\n')