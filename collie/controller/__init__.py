""" **CoLLie** 的训练器模块
"""

from .trainer import Trainer
from .evaluator import Evaluator, PerplexityEvaluator, ClassficationEvaluator

__all__ = ['Trainer', 'Evaluator', 'PerplexityEvaluator', 'ClassficationEvaluator']