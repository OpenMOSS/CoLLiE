import os
import torch

from collie.metrics import BaseMetric
from collie.utils import env
from collie.log import logger

class SFTAccMetric(BaseMetric):
    def __init__(self):
        super(SFTAccMetric, self).__init__(gather_result=True)
        self.right = 0
        self.total = 0

    def reset(self):
        self.right = 0
        self.total = 0

    def update(self, result):
        """

        :param result: dict. Gathered result of eval_fn. Contains `right`,
            `total` in this case.
        """
        self.right += sum(result["right"]).cpu().item()
        self.total += sum(result["total"]).cpu().item()

    def get_metric(self):
        acc = self.right / self.total
        return acc


class SFTDecodeMetric(BaseMetric):
    def __init__(self, tokenizer, gather_result=True):
        super(SFTDecodeMetric, self).__init__(gather_result=gather_result)
        self.tokenizer = tokenizer
        self.sentences = []

    def reset(self):
        self.sentences = []

    def update(self, result):
        """

        :param result: list. Gathered result of eval_fn. Contains `right`,
            `total` in this case.
        """
        input_ids = [r for r in result['generate']]
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
        return {"decode": self.sentences}