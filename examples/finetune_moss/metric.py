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

    def update(self, result):
        """

        :param result: list. Gathered result of eval_fn. Contains `right`,
            `total`, `generate` and `train_meta` in this case.
        """
        res = result
        if not isinstance(res, list):
            res = [result]
        for r in res:
            self.right += r["right"].cpu().item()
            self.total += r["total"].cpu().item()
        with open(f"rank_{env.rank}", "w") as fp:
            fp.write(f"{self.right}, {self.total}, {env.rank}, {len(result)}")

    def get_metric(self, train_meta):
        acc = self.right / self.total
        # logger.info("Epoch {} Batch {} Accuracy:{}".format(
        #     train_meta['epoch_idx'], train_meta['batch_idx'], acc
        # ))
        print(f"\n------------------\n"
              "Epoch {} Batch {} Accuracy:{}"
              "\n------------------\n".format(
             train_meta['epoch_idx'], train_meta['batch_idx'], acc
         ))
        self.right = 0
        self.total = 0


class SFTDecodeMetric(BaseMetric):
    def __init__(self, tokenizer, save_path, gather_result=True):
        super(SFTDecodeMetric, self).__init__(gather_result=gather_result)
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.sentences = []

    def update(self, result):
        """

        :param result: list. Gathered result of eval_fn. Contains `right`,
            `total`, `generate` and `train_meta` in this case.
        """
        if isinstance(result, list):
            # TODO
            # metric 的 gather 功能也不完善导致多个 metric 下这里的 result 是
            # list of list; 暂时规避掉这个问题
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

    def get_metric(self, train_meta):
        filename = "generate_result_epoch{}_step{}_rank{}".format(
            train_meta["epoch_idx"], train_meta["batch_idx"], env.rank
        )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, filename), "w") as fp:
            for sentence in self.sentences:
                fp.write(sentence + "\n-------------------------------\n")
        self.sentences = []