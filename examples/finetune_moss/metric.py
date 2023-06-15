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
        if env.dp_size > 1:
            self.right += sum(result["right"]).cpu().item()
            self.total += sum(result["total"]).cpu().item()
        else:
            self.right += result["right"].cpu().item()
            self.total += result["total"].cpu().item()

    def get_metric(self):
        acc = self.right / self.total
        return acc
