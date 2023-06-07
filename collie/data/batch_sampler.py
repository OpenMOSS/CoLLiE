class CollieBatchSampler:
    """
    Batch Sampler。在最后一个 batch 样本数目不足一个 ``batch size`` 时可以选择
    不处理（normal）、丢弃（drop）或从头补齐（fill）。

    :param sampler:
    :param batch_size:
    :param last_batch: 当最后一个 batch 样本数不足一个 ``batch_size`` 时的处理方式

        * ``'normal'`` - 不进行任何特殊处理。
        * ``'drop'`` - 丢弃最后一个 batch。
        * ``'fill'`` - 将最后一个 batch 补齐到 ``batch_size`` 大小。
    """
    def __init__(self, sampler, batch_size, last_batch="normal"):
        assert last_batch in ["normal", "drop", "fill"]
        self.sampler = sampler
        self.batch_size = batch_size
        self.last_batch = last_batch

    def __iter__(self):
        # torch BatchSampler.__iter__
        if self.last_batch == "drop":
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                if self.last_batch == "normal":
                    yield batch[:idx_in_batch]
                elif self.last_batch == "fill":
                    sampler_iter = iter(self.sampler)
                    while idx_in_batch < self.batch_size:
                        try:
                            batch[idx_in_batch] = next(sampler_iter)
                            idx_in_batch += 1
                        except StopIteration:
                            sampler_iter = iter(self.sampler)
                    yield batch

    def __len__(self) -> int:
        if self.last_batch == "drop":
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]