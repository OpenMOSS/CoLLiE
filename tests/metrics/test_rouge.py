import sys
sys.path.append("../../")

import pytest

from collie.metrics.rouge import RougeMetric

class TestRouge:
    
    def test_v1(self):
        rg = RougeMetric()
        rg.update({'pred': ['The quick brown fox jumps over the lazy dog'], "target": ['The quick brown dog jumps on the log.']})
        rg.get_metric()
    
    def test_v2(self):
        rg = RougeMetric(metrics=['rouge-1', 'rouge-3', 'rouge-4'])
        rg.update({'pred': ['The quick brown fox jumps over the lazy dog'], "target": ['The quick brown dog jumps on the log.']})
        result = rg.get_metric()
        print(result)
        rg.reset()
        
        rg.update({'pred': ['The quick brown dog jumps on the log.'], "target": ['The quick brown dog jumps on the log.']})
        result = rg.get_metric()
        print(result)
