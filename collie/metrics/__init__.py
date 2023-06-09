""" **CoLLie** 中的评价指标
"""
from .base import BaseMetric
from .decode import DecodeMetric
from .accuracy import AccuracyMetric
from .ppl import PplMetric

__all__ = [
    'BaseMetric',
    'DecodeMetric',
    'AccuracyMetric',
    'PplMetric'
]