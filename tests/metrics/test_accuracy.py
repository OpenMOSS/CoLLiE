import sys
sys.path.append("../..")
import pytest
from collie.metrics.accuracy import Accuracy
import torch

class TestAccuracy:

    def test_demo1(self):
        """
            测试输入为[B, n_classes] 的 pred 和 [B] 的 target
        """
        input_result = {'pred': torch.tensor([[0.2, 0.4, 0.3, 0.1], [0.8, 0.5, 0.4, 0.3],
                                              [0.6, 0.1, 0.2, 0.3], [0.3, 0.6, 0.65, 0.5]]),
                       'target': torch.tensor([0, 0, 0, 0])}
        metric = Accuracy()
        
        metric.update(input_result)
        metric_result = metric.get_metric()
        print(metric_result)
        assert metric_result == {'acc': 0.5, 'total': 4, 'correct': 2}

    def test_demo2(self):
        """
            测试输入为 pred 的 shape > target 的 shape
        """
        input_result = {'pred': torch.zeros(4, 3, 2),
                        'target': torch.zeros(4)}
        metric = Accuracy()

        with pytest.raises(RuntimeError) as e:
            metric.update(input_result)
        exec_msg = e.value.args[0]
        print(exec_msg)
        assert exec_msg == 'when pred have size:torch.Size([4, 3, 2]), target should have size: torch.Size([4, 3, 2]) or torch.Size([4, 3]), got torch.Size([4]).'

    def test_demo3(self):
        """
            测试输入带冗余的参数
        """
        input_result = {'pred': torch.tensor([[0.2, 0.4, 0.3, 0.1], [0.8, 0.5, 0.4, 0.3],
                                              [0.6, 0.1, 0.2, 0.3], [0.7, 0.6, 0.65, 0.5]]),
                         'target': torch.tensor([0, 0, 0, 0]),
                         'unused': 1}
        metric = Accuracy()
        
        metric.update(input_result)
        metric_result = metric.get_metric()
        print(metric_result)
        assert metric_result == {'acc': 0.75, 'total': 4, 'correct': 3}

    def test_demo4(self):
        input_result = {'pred': torch.tensor([  [[0.1831, 0.3912],
                                                [0.6588, 0.1285],
                                                [0.9351, 0.0895]],

                                                [[0.9126, 0.4580],
                                                [0.0580, 0.9839],
                                                [0.2595, 0.6489]],

                                                [[0.4953, 0.3380],
                                                [0.0509, 0.4378],
                                                [0.6848, 0.4253]],

                                                [[0.8520, 0.4735],
                                                [0.1479, 0.6368],
                                                [0.6576, 0.9255]]]),
                         'target': torch.tensor([[1, 0, 0],
                                                 [1, 0, -1],
                                                 [0, 1, -1],
                                                 [1, -1, -1]]),
                         'seq_len': torch.tensor([3, 2, 2, 1])}

        metric = Accuracy()
        
        metric.update(input_result)
        metric_result = metric.get_metric()
        print(metric_result)
        assert metric_result == {'acc': 0.625, 'total': 8, 'correct': 5}