import sys
sys.path.append("/mnt/petrelfs/zhangshuo/projects/collie")
from collie.metrics import BaseMetric

import torch
import math

from typing import Sequence, Any, Tuple, Dict
from collections import Counter

__all__ = [
    "BlueMetric"
]

def ngrams(sequence: Sequence[Any], n: int) -> Counter:
    """
    Generate the ngrams from a sequence of items

    Args:
        sequence: sequence of items
        n: n-gram order

    Returns:
        A counter of ngram objects

    .. versionadded:: 0.4.5
    """
    return Counter([tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)])


def lcs(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """
    Compute the length of the longest common subsequence in two sequence of items
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem

    Args:
        seq_a: first sequence of items
        seq_b: second sequence of items

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    m = len(seq_a)
    n = len(seq_b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def modified_precision(references: Sequence[Sequence[Any]], candidate: Any, n: int) -> Tuple[int, int]:
    """
    Compute the modified precision

    .. math::
       p_{n} = \frac{m_{n}}{l_{n}}

    where m_{n} is the number of matched n-grams between translation T and its reference R, and l_{n} is the
    total number of n-grams in the translation T.

    More details can be found in `Papineni et al. 2002`__.

    __ https://www.aclweb.org/anthology/P02-1040.pdf

    Args:
        references: list of references R
        candidate: translation T
        n: n-gram order

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    # ngrams of the candidate
    counts = ngrams(candidate, n)

    # union of ngrams of references
    max_counts: Counter = Counter()
    for reference in references:
        max_counts |= ngrams(reference, n)

    # clipped count of the candidate and references
    clipped_counts = counts & max_counts

    return sum(clipped_counts.values()), sum(counts.values())

def _closest_ref_length(references: Sequence[Sequence[Any]], hyp_len: int) -> int:
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


class _Smoother:
    """
    Smoothing helper
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """

    def __init__(self, method: str):
        valid = ["no_smooth", "smooth1", "nltk_smooth2", "smooth2"]
        if method not in valid:
            raise ValueError(f"Smooth is not valid (expected: {valid}, got: {method})")
        self.smooth = method

    def __call__(self, numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        method = getattr(self, self.smooth)
        return method(numerators, denominators)

    @staticmethod
    def smooth1(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        epsilon = 0.1
        denominators_ = [max(1, d.item()) for d in denominators]
        return [n.item() / d if n != 0 else epsilon / d for n, d in zip(numerators, denominators_)]

    @staticmethod
    def nltk_smooth2(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        denominators_ = torch.tensor([max(1, d.item()) for d in denominators])
        return _Smoother._smooth2(numerators, denominators_)

    @staticmethod
    def smooth2(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        return _Smoother._smooth2(numerators, denominators)

    @staticmethod
    def _smooth2(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        return [
            (n.item() + 1) / (d.item() + 1) if i != 0 else n.item() / d.item()
            for i, (n, d) in enumerate(zip(numerators, denominators))
        ]

    @staticmethod
    def no_smooth(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        denominators_ = [max(1, d) for d in denominators]
        return [n.item() / d for n, d in zip(numerators, denominators_)]


class BleuMetric(BaseMetric):
    r"""计算 `BLEU 得分 <https://en.wikipedia.org/wiki/BLEU>`_.

    .. math::
       \text{BLEU} = b_{p} \cdot \exp \left( \sum_{n=1}^{N} w_{n} \: \log p_{n} \right)

    这里的 :math:`N` 是 `n-gram` 的阶数, :math:`b_{p}` 是句子的简洁性惩罚, 
    :math:`w_{n}` 是加起来为一的正权重, 而 :math:`p_{n}` 是修改过的 n-gram 精确度.
    
    

    更多的细节可以查看 `Papineni et al. 2002 <https://www.aclweb.org/anthology/P02-1040>`_.

    另外, 关于平滑 (`smoothing`) 的技术可以查看 `Chen et al. 2014 <https://aclanthology.org/W14-3346.pdf>`_

    ``result`` 中需要包含 ``pred`` 和 ``target`` 字段，例如:

    .. code-block:: python
        result = {
            "pred": ["the the the the the the the", "cat cat cat cat cat cat cat"],
            "target": [["the cat is on the mat", "there is a cat on the mat"], 
            ["the cat is on the mat", "there is a cat on the mat"]]
        }

    :param ngram: n-gram 的阶数
    :param smooth: 是否使用平滑技术。可选值为 ``no_smooth``, ``smooth1``, ``nltk_smooth2`` 或 ``smooth2``. 默认为 ``no_smooth``
    :param average: 指定使用哪种类型的平均值 (宏平均或微平均)。更多细节可以查看 https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    :param gather_result: 是否对 DP 中不同进程的结果进行聚合。默认为 ``True``
    :param split_fn: 用于分割字符串的函数。默认以空格分割

    例子:

        .. code-block:: python

            from collie.metrics import BleuMetric

            m = BleuMetric(ngram=4, smooth="smooth1")

            result = {
                "pred": ["the the the the the the the"],
                "target": [["the cat is on the mat", "there is a cat on the mat"]]
            }

            m.update(result)

            print(m.get_metric())
    """

    def __init__(
        self,
        split_fn: callable = lambda x: x.split(" "),
        ngram: int = 4,
        smooth: str = "no_smooth",
        average: str = "macro",
        gather_result: bool=True
    ):
        if ngram <= 0:
            raise ValueError(f"ngram order must be greater than zero (got: {ngram})")
        self.ngrams_order = ngram
        self.weights = [1 / self.ngrams_order] * self.ngrams_order
        self.smoother = _Smoother(method=smooth)

        if average not in ["macro", "micro"]:
            raise ValueError(f'Average must be either "macro" or "micro" (got: {average})')
        self.average = average
        self.split_fn = split_fn
        self.reset()

        super(BleuMetric, self).__init__(gather_result=gather_result)

    def _n_gram_counter(
        self,
        references: Sequence[Sequence[Sequence[Any]]],
        candidates: Sequence[Sequence[Any]],
        p_numerators: torch.Tensor,
        p_denominators: torch.Tensor,
    ) -> Tuple[int, int]:
        if len(references) != len(candidates):
            raise ValueError(
                f"nb of candidates should be equal to nb of reference lists ({len(candidates)} != "
                f"{len(references)})"
            )

        hyp_lengths = 0
        ref_lengths = 0

        # Iterate through each hypothesis and their corresponding references.
        for refs, hyp in zip(references, candidates):
            # For each order of ngram, calculate the numerator and
            # denominator for the corpus-level modified precision.
            for i in range(1, self.ngrams_order + 1):
                numerator, denominator = modified_precision(refs, hyp, i)
                p_numerators[i] += numerator
                p_denominators[i] += denominator

            # Calculate the hypothesis lengths
            hyp_lengths += len(hyp)

            # Calculate the closest reference lengths.
            ref_lengths += _closest_ref_length(refs, len(hyp))

        return hyp_lengths, ref_lengths

    def _brevity_penalty_smoothing(
        self, p_numerators: torch.Tensor, p_denominators: torch.Tensor, hyp_length_sum: int, ref_length_sum: int
    ) -> float:
        # Returns 0 if there's no matching n-grams
        # We only need to check for p_numerators[1] == 0, since if there's
        # no unigrams, there won't be any higher order ngrams.
        if p_numerators[1] == 0:
            return 0

        # If no smoother, returns 0 if there's at least one a not matching n-grams]
        if self.smoother.smooth == "no_smooth" and min(p_numerators[1:]).item() == 0:
            return 0

        # Calculate corpus-level brevity penalty.
        if hyp_length_sum < ref_length_sum:
            bp = math.exp(1 - ref_length_sum / hyp_length_sum) if hyp_length_sum > 0 else 0.0
        else:
            bp = 1.0

        # Smoothing
        p_n = self.smoother(p_numerators[1:], p_denominators[1:])

        # Compute the geometric mean
        s = [w_i * math.log(p_i) for w_i, p_i in zip(self.weights, p_n)]
        gm = bp * math.exp(math.fsum(s))
        return gm

    def _sentence_bleu(self, references: Sequence[Sequence[Any]], candidates: Sequence[Any]) -> float:
        return self._corpus_bleu([references], [candidates])

    def _corpus_bleu(self, references: Sequence[Sequence[Sequence[Any]]], candidates: Sequence[Sequence[Any]]) -> float:
        p_numerators: torch.Tensor = torch.zeros(self.ngrams_order + 1)
        p_denominators: torch.Tensor = torch.zeros(self.ngrams_order + 1)

        hyp_length_sum, ref_length_sum = self._n_gram_counter(
            references=references, candidates=candidates, p_numerators=p_numerators, p_denominators=p_denominators
        )
        bleu_score = self._brevity_penalty_smoothing(
            p_numerators=p_numerators,
            p_denominators=p_denominators,
            hyp_length_sum=hyp_length_sum,
            ref_length_sum=ref_length_sum,
        )

        return bleu_score

    def reset(self) -> None:
        if self.average == "macro":
            self._sum_of_bleu = torch.tensor(0.0, dtype=torch.double)
            self._num_sentences = 0

        if self.average == "micro":
            self.p_numerators = torch.zeros(self.ngrams_order + 1)
            self.p_denominators = torch.zeros(self.ngrams_order + 1)
            self.hyp_length_sum = 0
            self.ref_length_sum = 0

    def update(self, result: Dict):
        assert "pred" in result.keys() and "target" in result.keys(), "result must contain pred and target"
        pred = [self.split_fn(x) for x in result["pred"]]
        target = [[self.split_fn(x) for x in y] for y in result["target"]]
        if self.average == "macro":
            for refs, hyp in zip(target, pred):
                self._sum_of_bleu += self._sentence_bleu(references=refs, candidates=hyp)
                self._num_sentences += 1

        elif self.average == "micro":
            hyp_lengths, ref_lengths = self._n_gram_counter(
                references=target, candidates=pred, p_numerators=self.p_numerators, p_denominators=self.p_denominators
            )
            self.hyp_length_sum += hyp_lengths
            self.ref_length_sum += ref_lengths

    def _compute_macro(self) -> torch.Tensor:
        if self._num_sentences == 0:
            raise RuntimeError("Bleu must have at least one example before it can be computed.")

        return self._sum_of_bleu / self._num_sentences

    def _compute_micro(self) -> float:
        bleu_score = self._brevity_penalty_smoothing(
            p_numerators=self.p_numerators,
            p_denominators=self.p_denominators,
            hyp_length_sum=self.hyp_length_sum,
            ref_length_sum=self.ref_length_sum,
        )
        return bleu_score

    def get_metric(self) -> None:
        if self.average == "macro":
            return self._compute_macro().item()
        elif self.average == "micro":
            return self._compute_micro().item()
        
if __name__ == "__main__":
    m = BleuMetric(ngram=2)
    result = {
        "pred": ['10 or 15 years ago, here at Ted, Peter Skillman presented a design challenge called the Marshmallow Challenge.'],
        "target": [['Several years ago here at TED, Peter Skillman introduced a design challenge called the marshmallow challenge.']]
    }
    m.update(result)
    print(m.get_metric())