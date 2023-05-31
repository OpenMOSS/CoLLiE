from typing import Any, Dict

from collie.metrics import BaseMetric
from collie.log import logger


class _MetricsWrapper:
    """
    对输入的 metrics 进行封装以便于支持 Trainer 使用

    :param metrics: 用户传进来的 metric, 类型为 Dict。例如 {'name1': metric1, 'name2': metric2}
    :param trainer: 类型为 :class:`~collie.trainer.Trainer`, 用以对 metrics 进行初始化。
    """

    def __init__(self, metrics, trainer):
        self._metrics = []
        self._metric_names = []
        if metrics is not None:
            if not isinstance(metrics, Dict):
                raise TypeError('Parameter `metrics` can only be `Dict` type.')
            for metric_name, metric in metrics.items():
                if isinstance(metric, BaseMetric):
                    metric.construct(trainer)
                else:
                    raise ValueError(f"{metric_name}:{metric.__class__.__name__} must be instance of BaseMetric, but it not!")
                self._metric_names.append(metric_name)
                self._metrics.append(metric)

    def update(self, result):
        """
        针对一个批次的预测结果做评价指标的累计。

        :param result: 用以计算 metric 的预测结果，其类型只能为 Dict。
        """
        for metric in self._metrics:
            if not isinstance(result, dict):
                raise RuntimeError(
                    'The output of your model is of type:`{}`, please '
                    'either directly return a dict from your model'.
                    format(type(result)))
            # gather 输入
            if metric.gather_result:
                gather_out = metric.gather(result)
            else:
                gather_out = result
            metric.update(gather_out)

    def reset(self):
        """
        将 Metric 中的状态重新设置。
        """
        for metric in self._metrics:
            metric.reset()

    def get_metric(self) -> Dict:
        """调用各个 metric 得到 metric 的结果。

        :return: 返回的类型如 {'metric_name1': metric_results,'metric_name2': metric_results}
        """
        results = {}
        for metric_name, metric in zip(self._metric_names, self._metrics):
            if isinstance(metric, BaseMetric):
                _results = metric.get_metric()
            else:
                raise RuntimeError(f'Not support `{type(metric)}` for now.')
            if _results is not None:
                results[metric_name] = _results
            else:
                logger.warning_once(f'Metric:{metric_name} returns None when '
                                    'getting metric results.')
        return results