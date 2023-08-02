from torch.utils.data import DistributedSampler, DataLoader
from deepspeed.runtime.data_pipeline.data_sampling.data_sampler import DeepSpeedDataSampler
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.data_pipeline.constants import CURRICULUM_LEARNING, \
    DATA_SAMPLING_NUM_WORKERS, DATA_SAMPLING, CURRICULUM_LEARNING_ENABLED

from collie.utils import env
from .batch_sampler import CollieBatchSampler

class CollieDataLoader(object):
    """
    **CoLLiE** 封装的 DataLoader。
    
    在流水线并行的情景下每次迭代取出 ``batch_size * accumulation_steps`` 个
    sample。

    :param dataset:
    :param batch_size:
    :param pin_memory:
    :param shuffle:
    :param accumulation_steps:
    :param collate_fn:
    :param sampler:
    :param drop_last: 当最后一个 batch 样本数不足时是否丢弃。在流水线情况下如果为
        ``False``，则会补齐最后一个 batch。
    :param data_efficiency_config: DeepSpeed 中关于 ``Data Effiency`` 部分的设置
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 accumulation_steps=1,
                 shuffle=False,
                 pin_memory=True,
                 collate_fn=None,
                 num_workers=None,
                 sampler=None,
                 drop_last=False,
                 data_efficiency_config={}):
        self.batch_size = batch_size
        if env.pp_size > 1:
            self.batch_size *= accumulation_steps

        try:
            self.curriculum_learning_enabled = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]
        except KeyError:
            self.curriculum_learning_enabled = False

        if self.curriculum_learning_enabled:
            sampler = DeepSpeedDataSampler(
                data_efficiency_config, len(dataset), self.batch_size,
                env.dp_rank, env.dp_size, env.dp_group, accumulation_steps,
                env.rank, drop_last=drop_last
            )
            device_count = get_accelerator().device_count()
            num_workers = data_efficiency_config[DATA_SAMPLING][DATA_SAMPLING_NUM_WORKERS]
        else:
            if sampler is None:
                sampler = DistributedSampler(
                    dataset=dataset, num_replicas=env.dp_size,
                    rank=env.dp_rank, shuffle=shuffle
                )
            device_count = 1

            if num_workers is None:
                num_workers = 2 * device_count

        self.num_workers = num_workers
        self.sampler = sampler
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device_count = device_count
        self.pin_memory = pin_memory
        self.data = None
        self.drop_last = drop_last
        self.post_process_func = None

        if self.drop_last:
            self.len = len(self.sampler) // self.batch_size
        else:
            from math import ceil
            self.len = ceil(len(self.sampler) / self.batch_size)

    def __iter__(self):
        # TODO why DeepSpeedDataLoader do so?
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.curriculum_learning_enabled:
            data = next(self.data_iterator)
            if self.post_process_func is not None:
                data = self.post_process_func(data, self.sampler.state_dict())
            return data
        else:
            return next(self.data)

    def _create_dataloader(self):
        if self.curriculum_learning_enabled:
            self.dataloader = DataLoader(
                self.dataset, pin_memory=self.pin_memory,
                batch_sampler=self.sampler, num_workers=self.num_workers,
                collate_fn=self.collate_fn
            )
            self.data_iterator = iter(self.dataloader)
            return self.dataloader
        else:
            if self.drop_last:
                last_batch = "drop"
            elif env.pp_size > 1:
                last_batch = "fill"
            else:
                last_batch = "normal"
            batch_sampler = CollieBatchSampler(self.sampler, self.batch_size,
                                               last_batch)
            self.dataloader = DataLoader(self.dataset,
                                         batch_sampler=batch_sampler,
                                         collate_fn=self.collate_fn,
                                         num_workers=self.num_workers)
            self.data = (x for x in self.dataloader)

            return self.dataloader
