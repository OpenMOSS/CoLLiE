from typing import Optional, Dict, Callable, Sequence, Tuple, Any, List, Iterable

import torch
from torch import nn
import torch.distributed as dist
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.accelerator import get_accelerator
from transformers.generation.utils import GenerationConfig
from transformers import PreTrainedTokenizerBase
from peft import PeftModel

from collie.module import PipelineGenerationMixin, GPTLMLoss, PipelineModel
from collie.data.dataloader import CollieDataLoader
from collie.utils.rich_progress import f_rich_progress
from collie.log import logger
from collie.config import CollieConfig
from collie.utils import progress, env, setup_ds_engine, BaseProvider, _GenerationStreamer, _MetricsWrapper, BaseMonitor, _MultiMonitors, broadcast_tensor, ColliePadder, auto_param_call
from .server import Server

class Evaluator:
    """
    **CoLLie** 评测器，用于评测模型的性能

    :param model: 用于训练和验证的模型，可以使用 **CoLLie** 实现的模型或 transformers 提供的模型：

        * **CoLLie** 实现的模型 :class:`.CollieModelForCausalLM` 可支持的并行方式包括：张量并行、流水线并行、`ZeRO`
        * transformers 提供的模型 ``transformers.PreTrainedModel`` 只支持 `ZeRO`
    :param dataset: 用于验证的数据集。
        **CoLLie** 可接收的 ``dataset`` 为可迭代对象，例如 ``torch.utils.data.Dataset``
        
        .. note::

            当未提供  ``collate_fn`` 时，``dataset`` 的取值应当为 `Dict` 类型
            
    :param tokenizer: 用于训练和验证的分词器，该分词器将用于:
        * 使用 :class:`~collie.controller.evaluator.Evaluator` 进行基于生成的验证时，使用 `tokenizer` 对生成的结果进行解码
        若无上述需求，可不传入 `tokenizer`
    :param eval_fn: 您可以传入该参数来定制每次评测一个 batch 的数据时所执行的函数。该函数应接受的两个参数为 ``evaluator`` 和 ``batch``，
        返回值为 Dict 类型；若为 `None`, 默认调用 Evaluator 自带的 eval_fn。       
    :param config: 用于验证的配置，必须删除关于 ``optimizer`` 的字段
    :param collate_fn: 用于验证数据集的 `collate_fn`。
        ``collate_fn`` 只可接受一个参数，为  ``dataset`` 迭代值组成的 ``List``。
        
        .. note::

            ``collate_fn`` 的返回值必须是为 `Dict` 类型
                
        例如:

        .. code-block:: python
    
            from transformers import AutoTokenizer
            def collate_fn(batch):
                # batch = ["样本1", "样本2", ...]
                tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", padding_side="left", trust_remote_code=True)
                input_ids = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
                # 第二个 input_ids 会被用于 loss_fn 的 label
                return {"input_ids": input_ids, "labels": input_ids}
    :param data_provider: 额外的数据提供器，可在 ``dataset`` 之外额外注入验证数据，例如通过前端网页或 http 请求等， 详见 :class:`~collie.utils.data_provider.BaseProvider`
    :param monitors: 用于监控训练过程的监控器，详见 :class:`~collie.utils.monitor.BaseMonitor`
    """

    def __init__(self, model, dataset: torch.utils.data.Dataset, tokenizer: Optional[PreTrainedTokenizerBase] = None, metrics: Optional[Dict] = None, eval_fn: Optional[Callable]=None,
                 config: Optional[CollieConfig] = None, collate_fn: Optional[Callable] = ColliePadder(padding_left=True), data_provider: Optional[BaseProvider] = None,
                 monitors: Sequence[BaseMonitor] = []):
        self.engine = None
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.metric_wrapper = _MetricsWrapper(self.metrics, self)
        self.config = config
        self.eval_fn = self.eval_fn if eval_fn is None else eval_fn
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.eval_dataloader = None
        self.data_provider = data_provider
        self.server = None
        if self.data_provider is not None:
            self.server = Server(model=self.model, data_provider=self.data_provider)
        self.monitor = _MultiMonitors(monitors)
        self.global_batch_idx = 0

    def init_engine(self):
        """
        初始化 engine。config 中 的 optimizer 手动删掉， 不然会自动调用
        """
        if dist.get_world_size() != self.config.tp_size * self.config.dp_size * self.config.pp_size:
            logger.rank_zero_warning("The world size is not equal to the product of the parallel sizes set."
                                     f"{dist.get_world_size()} != {self.config.tp_size} * {self.config.dp_size} * {self.config.dp_size}.")
            self.config.dp_size = dist.get_world_size() // (self.config.tp_size * self.config.pp_size)
            logger.rank_zero_warning(f"Set dp_size to {self.config.dp_size}.")
        object.__setattr__(self, 
                           "engine", 
                           setup_ds_engine(config=self.config, model=self.model)[0])
        if isinstance(self.engine.module, PipelineGenerationMixin):
            self.engine.module.set_engine(self.engine)
        if isinstance(self.engine.module, PeftModel) and isinstance(self.engine.module.get_base_model(), PipelineGenerationMixin):
            self.engine.module.get_base_model().set_engine(self.engine)
    
    def eval(self, dataloader: Optional[Iterable] = None):
        """
        对数据集进行一次 eval 测试并返回 metric 的结果。需要注意的是如果 ``Evaluator`` 中的 engine 没有初始化，那么默认会自动初始化一个 engine。

        :param dataloader: 用于 eval 的数据集，为 ``Iterable`` 对象 ，当为 ``None`` 时，使用默认的 ``dataset`` 生成的 ``eval_dataloader``
        """
        if self.engine is None:
            self.init_engine()
        if self.server is not None:
            self.server.start()
        if self.eval_dataloader is None:
            self.eval_dataloader = CollieDataLoader(
                self.dataset, self.config.eval_batch_size,
                self.config.gradient_accumulation_steps, shuffle=False,
                collate_fn=self.collate_fn
            )
            self.eval_steps = len(self.eval_dataloader)
        eval_dataloader = self.eval_dataloader
        if dataloader is not None:
            eval_dataloader = dataloader
        with progress(eval_dataloader, desc="Evaluating Batch: ", disable=env.rank != 0, total=self.eval_steps) as tqbar_batch:
            for batch_idx, batch in enumerate(tqbar_batch):
                tqbar_batch.set_description(f"Evaluating Batch: {batch_idx} / {self.eval_steps}")
                if self.server is not None:
                    self.server.data_provider_handler()
                self.engine.eval()
                if isinstance(self.engine.module, PipelineModel):
                    self.engine.module.forward_type = "eval"
                if isinstance(self.engine.module, PeftModel) and isinstance(self.engine.module.get_base_model(), PipelineModel):
                    self.engine.module.get_base_model().forward_type = "eval"
                with torch.no_grad():
                    batch['past_key_values'] = None
                    result = self.eval_fn(self, batch)
                self.metric_wrapper.update(result)
        with self.monitor as item:
            metric_results = self.metric_wrapper.get_metric()
            for key in list(metric_results.keys()):
                if isinstance(metric_results[key], dict):
                    for k in list(metric_results[key].keys()):
                        metric_results[f"{key}#{k}"] = metric_results[key][k]
                    del metric_results[key]
            item.update({
                "eval_result": metric_results, 
                "global_batch_idx": self.global_batch_idx,
                "mode": "eval"})
        self.metric_wrapper.reset()

        if len(metric_results) > 0 and env.rank == 0:  # 如果 metric 不为 None 需要 print 。
            f_rich_progress.print_json(metric_results)
            
        return metric_results
    
    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Tuple) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为 ``Dict``

            .. note::

                不同的 Evaluator，需求的数据类型也u偶所不同。
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        raise NotImplementedError

class EvaluatorForGeneration(Evaluator):
    def __init__(self, 
                 generation_config: GenerationConfig = GenerationConfig(),
                 skip_special_tokens: bool = True,
                 *args,
                 **kwargs):
        self.generation_config = generation_config
        self.skip_special_tokens = skip_special_tokens
        super().__init__(*args, **kwargs)
        
    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Tuple) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 ``Dict``，格式为：
        
            .. code-block::
            {
                "input_ids": torch.tensor([[1, 100, 100, 2]]),
                "taregt": torch.tensor([[1, 100, 100, 2]]),
            }
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        if isinstance(evaluator.engine.module, PipelineModel):
            evaluator.engine.module.forward_type = "generate"
        if isinstance(evaluator.engine.module, PeftModel) and isinstance(evaluator.engine.module.get_base_model(), PipelineModel):
            evaluator.engine.module.get_base_model().forward_type = "generate"
        assert evaluator.tokenizer is not None, "You must provide a tokenizer to decode the generated results."
        generated_ids = evaluator.engine.module.generate(**{k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}, generation_config=evaluator.generation_config)
        prompt_length = batch["input_ids"].shape[1]
        result = {"pred": [evaluator.tokenizer.decode(sample[prompt_length:], skip_special_tokens=evaluator.skip_special_tokens) for sample in generated_ids]}
        if "target" in batch.keys():
            result["target"] = [evaluator.tokenizer.decode(sample, skip_special_tokens=evaluator.skip_special_tokens) for sample in batch["target"]]
        return result
        
class EvaluatorForPerplexity(Evaluator):
    def __init__(self, 
                 loss_fn: Callable = GPTLMLoss(),
                 collate_fn: Optional[Callable] = ColliePadder(),
                 *args,
                 **kwargs):
        self.loss_fn = loss_fn
        super().__init__(collate_fn=collate_fn, *args, **kwargs)
        
    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Tuple) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 ``Dict``，格式为：
        
            .. code-block::
            {
                "input_ids": torch.tensor([[1, 100, 100, 2]]),
                "labels": torch.tensor([[1, 100, 100, 2]]),
            }
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        # concat prompt labels for p-tuning
        if evaluator.config.peft_config and evaluator.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
            batch_size = batch["input_ids"].shape[0]
            if "labels" in batch.keys():
                prefix_labels = torch.full((batch_size, evaluator.config.peft_config.num_virtual_tokens), -100).to(batch["labels"].device)
                batch["labels"] = torch.cat((prefix_labels, batch["labels"]), dim=1)
        if evaluator.config.pp_size > 1:
            evaluator.engine.module.forward_type = "eval"
            outputs = evaluator.engine.module(**batch)
        else:
            outputs = evaluator.engine(**batch)
        ppl = torch.exp(auto_param_call(evaluator.loss_fn, {**batch, **outputs}, 
                                        signature_fn=evaluator.loss_fn.forward if isinstance(evaluator.loss_fn, nn.Module) else evaluator.loss_fn))
        return {
            "ppl": ppl.detach().clone().view(1,).cuda(),
            # **{key: value.cuda() for key, value in batch[1].items() if isinstance(value, torch.Tensor)}
        }
        
class EvaluatorForClassfication(EvaluatorForPerplexity):
    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Dict) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 ``Dict``，格式为：
        
            .. code-block::
            {
                "input_ids": [
                    torch.tensor([[1, 100, 100, 2]]),
                    torch.tensor([[1, 100, 100, 2]]),
                    torch.tensor([[1, 100, 100, 2]])    
                ],
                "target": torch.tensor([[0]])
            }
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        assert isinstance(batch["input_ids"], Sequence), f"input_ids must be a list for classification task. But got {type(batch['input_ids'])}."
        assert isinstance(batch["attention_mask"], Sequence), f"input_ids must be a list for classification task. But got {type(batch['attention_mask'])}."
        pred = torch.zeros((batch["input_ids"][0].shape[0], len(batch["input_ids"])))
        for idx, input_ids in enumerate(batch["input_ids"]):
            assert isinstance(input_ids, torch.Tensor), "input_ids must be a list of torch.Tensor for classification task."
            inputs = {"input_ids": input_ids.cuda(), "labels": batch["labels"][idx].cuda(), "attention_mask": batch["attention_mask"][idx].cuda(), 
                      **{key: value.cuda() for key, value in batch.items() if key not in ("input_ids", "attention_mask", "labels")}}
            # concat prompt labels for p-tuning
            if evaluator.config.peft_config and evaluator.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
                batch_size = input_ids.shape[0]
                if "labels" in inputs.keys():
                    prefix_labels = torch.full((batch_size, evaluator.config.peft_config.num_virtual_tokens), -100).to(inputs["labels"].device)
                    inputs["labels"] = torch.cat((prefix_labels, inputs["labels"]), dim=1)
            if evaluator.config.pp_size > 1:
                evaluator.engine.module.forward_type = "eval"
                logits = evaluator.engine.module(**inputs)["logits"]
            else:
                logits = evaluator.engine(**inputs)["logits"]
            for sample_idx in range(input_ids.shape[0]):
                pred[sample_idx, idx] = evaluator.loss_fn(logits[sample_idx: sample_idx + 1, :], inputs["labels"][sample_idx: sample_idx + 1, :]).detach().cpu().item()
        pred = pred.argmin(dim=1)
        return {
            "pred": pred.cuda(),
            "target": batch["target"].squeeze(1).cuda(),
            # **{key: value.cuda() for key, value in labels.items() if isinstance(value, torch.Tensor)}
        }
