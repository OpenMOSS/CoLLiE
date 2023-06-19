from typing import Any, Dict, Callable, List, Optional
from collections import Counter
import itertools

from collie.metrics import BaseMetric


class Ngrams(object):
    """
        Ngrams datastructure based on `set` 
    """

    def __init__(self, ngrams={}):
        self._ngrams = set(ngrams)

    def add(self, o):
        self._ngrams.add(o)

    def __len__(self):
        return len(self._ngrams)

    def intersection(self, o):
        inter_set = self._ngrams.intersection(o._ngrams)
        return Ngrams(inter_set)

    def union(self, *ngrams):
        union_set = self._ngrams
        for o in ngrams:
            union_set = union_set.union(o._ngrams)
        return Ngrams(union_set)

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = Ngrams()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _split_into_words(sentences, split_fn: Callable):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[split_fn(_) for _ in sentences]))

def _get_word_ngrams(n, sentences, split_fn: Callable):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0   
    assert n > 0

    words = _split_into_words(sentences, split_fn)
    return _get_ngrams(n, words)

def ngrams(sequence, n):
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

def rouge_n(evaluated_sentences: List, reference_sentences: List, split_fn: Callable=None, n: int=2, raw_results: bool=False):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.

    Returns:
      A tuple (f1, precision, recall) for ROUGE-N

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Hypothesis is empty.")
    if len(reference_sentences) <= 0:
        raise ValueError("Reference is empty.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences, split_fn)
    reference_ngrams = _get_word_ngrams(n, reference_sentences, split_fn)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)
    if raw_results:
        o = {
            "hyp": evaluated_count,
            "ref": reference_count,
            "overlap": overlapping_count
        }
        return o
    else:
        return f_r_p_rouge_n(
            evaluated_count, reference_count, overlapping_count)

def f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count):
    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return {"f": f1_score, "p": precision, "r": recall}

def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    return Ngrams(recon_list)

def _union_lcs(evaluated_sentences, reference_sentence, prev_union=None, split_fn: Callable = None):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C.
    For example:
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8
    and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1
    is "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5".
    The union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5"
    and LCS_u(r_i, C) = 4/5.

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      float: LCS_u(r_i, C)

    ValueError:
      Raises exception if a param has len <= 0
    """
    if prev_union is None:
        prev_union = Ngrams()

    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = prev_union
    prev_count = len(prev_union)
    reference_words = _split_into_words([reference_sentence],  split_fn)

    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s],  split_fn)
        lcs = _recon_lcs(reference_words, evaluated_words)
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    new_lcs_count = len(lcs_union) - prev_count
    return new_lcs_count, lcs_union

def rouge_l_summary_level(
        evaluated_sentences: List, reference_sentences: List, split_fn: Callable=None, raw_results=False, **_):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = (2*R_lcs*P_lcs) / (R_lcs * P_lcs)

    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    m = len(
        Ngrams(
            _split_into_words(reference_sentences, split_fn=split_fn)))

    # total number of words in evaluated sentences
    n = len(
        Ngrams(
            _split_into_words(evaluated_sentences, split_fn=split_fn)))

    # print("m,n %d %d" % (m, n))
    union_lcs_sum_across_all_references = 0
    union = Ngrams()
    for ref_s in reference_sentences:
        lcs_count, union = _union_lcs(evaluated_sentences,
                                      ref_s,
                                      prev_union=union)
        union_lcs_sum_across_all_references += lcs_count

    llcs = union_lcs_sum_across_all_references
    r_lcs = llcs / m
    p_lcs = llcs / n

    f_lcs = 2.0 * ((p_lcs * r_lcs) / (p_lcs + r_lcs + 1e-8))

    if raw_results:
        o = {
            "hyp": n,
            "ref": m,
            "overlap": llcs
        }
        return o
    else:
        return {"f": f_lcs, "p": p_lcs, "r": r_lcs}


class RougeMetric(BaseMetric):
    
    DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-l"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref, **k: rouge_n(hyp, ref, 1, **k),
        "rouge-2": lambda hyp, ref, **k: rouge_n(hyp, ref, 2, **k),
        "rouge-3": lambda hyp, ref, **k: rouge_n(hyp, ref, 3, **k),
        "rouge-4": lambda hyp, ref, **k: rouge_n(hyp, ref, 4, **k),
        "rouge-5": lambda hyp, ref, **k: rouge_n(hyp, ref, 5, **k),
        "rouge-l": lambda hyp, ref, **k:
            rouge_l_summary_level(hyp, ref, **k),
    }
    DEFAULT_STATS = ["r", "p", "f"]
    AVAILABLE_STATS = ["r", "p", "f"]
    
    def __init__(self, metrics: List=None, stats=None, split_fn: Callable = lambda x: x.split(" "), gather_result: bool = False) -> None:
        super().__init__(gather_result)
        # 检查用户传进来的 metrics 参数是否正确
        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]
            
            for m in self.metrics:
                if m not in RougeMetric.AVAILABLE_METRICS:
                    raise ValueError("Unknow metric '%s'" % m)
        else:
            self.metrics = RougeMetric.DEFAULT_METRICS
        
        if stats is not None:
            self.stats = [s.lower() for s in stats]

            for s in self.stats:
                if s not in RougeMetric.AVAILABLE_STATS:
                    raise ValueError("Unknown stat '%s'" % s)
        else:
            self.stats = RougeMetric.DEFAULT_STATS
        
        self.split_fn = split_fn
        
        self.scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
        self.total = 0
    
    def update(self, result: Dict):
        assert "pred" in result.keys() and "target" in result.keys(), "result must contain pred and target"
        pred = [self.split_fn(x) for x in result["pred"]]
        target = [self.split_fn(y) for y in result["target"]]
        for hyp, ref in zip(pred, target):
            # 按照 . 将句子划分， 用于摘要
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
            for m in self.metrics:
                fn = RougeMetric.AVAILABLE_METRICS[m]
                sc = fn(hyp=hyp, ref=ref)
                self.scores[m] = {s: self.scores[m][s] + sc[s] for s in self.stats}
        self.total += 1
    
    def reset(self):
        self.scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
    
    def get_metric(self) -> Optional[Dict]:
        avg_scores = {
            m: {s: self.scores[m][s] / self.total for s in self.stats}
            for m in self.metrics
        }
        return avg_scores

