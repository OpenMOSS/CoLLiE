import os
import torch

from collie.metrics import BaseMetric
from collie.utils import env
from collie.log import logger

class AlpacaDecodeMetric(BaseMetric):
    def __init__(self, tokenizer, gather_result: bool = False) -> None:
        super().__init__(gather_result)
        self.tokenizer = tokenizer
        self.sentences = []
    def update(self, result):
        if isinstance(result, list):
            input_ids = [r[0]['generate'] for r in result]
        else:
            input_ids = [result['generate']]
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
        for ids in decode_list:
            self.sentences.append(self.tokenizer.decode(ids))

    def get_metric(self):
        return self.sentences