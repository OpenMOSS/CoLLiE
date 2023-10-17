from typing import Dict, Optional
from .base import BaseMetric

class PPLMetric(BaseMetric):
    """
    计算困惑度 `Perplexity <https://en.wikipedia.org/wiki/Perplexity>`_ 的 Metric。
    """
    def __init__(self, gather_result: bool = False) -> None:
        super().__init__(gather_result)
        self.ppl = 0.
        self.total = 0
        
    def reset(self):
        self.ppl = 0.
        self.total = 0
        
    def get_metric(self) -> Optional[Dict]:
        return {'ppl': round(self.ppl / (self.total + 1e-12), 6)}
        
    def update(self, result: Dict):
        assert "ppl" in result.keys(), f"ppl not in result!"
        self.ppl += float(result["ppl"].sum())
        self.total += result["ppl"].shape[0]