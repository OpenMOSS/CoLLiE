from typing import Optional, Dict, Callable, Sequence, Tuple, Any, List, Iterable

import torch
import torch.distributed as dist
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.accelerator import get_accelerator
from transformers.generation.utils import GenerationConfig

from collie.module import PipelineGenerationMixin, GPTLMLoss
from collie.data.dataloader import CollieDataLoader
from collie.utils.rich_progress import f_rich_progress
from collie.log import logger
from collie.config import CollieConfig
from collie.utils import progress, env, setup_ds_engine, BaseProvider, _GenerationStreamer, _MetricsWrapper, BaseMonitor, _MultiMonitors, broadcast_tensor, ColliePadder

class Evaluator:
    """
    **CoLLie** 评测器，用于评测模型的性能

    :param model: 用于训练和验证的模型，可以使用 **CoLLie** 实现的模型或 transformers 提供的模型：

        * **CoLLie** 实现的模型 :class:`.CollieModelForCausalLM` 可支持的并行方式包括：张量并行、流水线并行、`ZeRO`
        * transformers 提供的模型 ``transformers.PreTrainedModel`` 只支持 `ZeRO`
    :param dataset: 用于验证的数据集。
        **CoLLie** 可接收的 ``dataset`` 为可迭代对象，例如 ``torch.utils.data.Dataset``
        
        .. note::

            当未提供  ``collate_fn`` 时，``dataset`` 的取值应当为长度为 **2** 的 `Tuple` 类型，例如 `(a, b)`，其中:
                
            * `a` 即 ``input_ids``， 为 ``torch.Tensor`` 类型，表示模型的的输入
            * `b` 可以为 ``torch.Tensor``，也可以是由 ``torch.Tensor`` 组成的任意长度的 `Tuple`，此项会作为 `loss_fn` 的第二个参数传入
    :param eval_fn: 您可以传入该参数来定制每次评测一个 batch 的数据时所执行的函数。该函数应接受的两个参数为 ``evaluator`` 和 ``batch``，
        返回值为 Dict 类型；若为 `None`, 默认调用 Evaluator 自带的 eval_fn。       
    :param config: 用于验证的配置，必须删除关于 ``optimizer`` 的字段
    :param collate_fn: 用于验证数据集的 `collate_fn`。
        ``collate_fn`` 只可接受一个参数，为  ``dataset`` 迭代值组成的 ``List``。
        
        .. note::

            ``collate_fn`` 的返回值必须是为长度为 **2** 的 `Tuple` 类型，，例如 `(a, b)`，其中:
            
            * `a` 即 ``input_ids``，为 ``torch.Tensor`` 类型，表示模型的的输入
            * `b` 可以为 ``torch.Tensor``，也可以是由 ``torch.Tensor`` 组成的任意长度的 ``Tuple``，此项会作为 ``loss_fn`` 的第二个参数传入
                
        例如:

        .. code-block:: python
    
            from transformers import AutoTokenizer
            def collate_fn(batch):
                # batch = ["样本1", "样本2", ...]
                tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", padding_side="left", trust_remote_code=True)
                input_ids = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
                # 第二个 input_ids 会被用于 loss_fn 的 label
                return input_ids, input_ids
    :param data_provider: 额外的数据提供器，可在 ``dataset`` 之外额外注入验证数据，例如通过前端网页或 http 请求等， 详见 :class:`~collie.utils.data_provider.BaseProvider`
    :param generation_config: 用于验证的配置
        **CoLLie** 默认的 ``eval_fn`` 为进行一次生成过程，因此本项配置主要控制生成过程的参数。当自定义 ``eval_fn`` 时，本项配置将不会生效
    :param monitors: 用于监控训练过程的监控器，详见 :class:`~collie.utils.monitor.BaseMonitor`
    """

    def __init__(self, model, dataset: torch.utils.data.Dataset, metrics: Optional[Dict] = None, eval_fn: Optional[Callable]=None,
                 config: Optional[CollieConfig] = None, collate_fn: Optional[Callable] = ColliePadder(padding_left=True), data_provider: Optional[BaseProvider] = None,
                 generation_config: GenerationConfig = GenerationConfig(), monitors: Sequence[BaseMonitor] = []):
        self.engine = None
        self.model = model
        self.metrics = metrics
        self.metric_wrapper = _MetricsWrapper(self.metrics, self)
        self.config = config
        self.generation_config = generation_config
        self.eval_fn = self.eval_fn if eval_fn is None else eval_fn
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.eval_dataloader = None
        self.data_provider = data_provider
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
        if self.config.pp_size > 1:
            # GPTLMLoss 是 Module，会被 nn.Module 加入 _Modules
            # 如果 loss_fn 是一个函数就会在此时报错
            if not isinstance(self.loss_fn, torch.nn.Module):
                del self.model.loss_fn
            self.model.loss_fn = self.loss_fn
        self.engine, _, _, _ = setup_ds_engine(
            model=self.model,
            config=self.config,
        )
    
    def eval(self, dataloader: Optional[Iterable] = None):
        """
        对数据集进行一次 eval 测试并返回 metric 的结果。需要注意的是如果 ``Evaluator`` 中的 engine 没有初始化，那么默认会自动初始化一个 engine。

        :param dataloader: 用于 eval 的数据集，为 ``Iterable`` 对象 ，当为 ``None`` 时，使用默认的 ``dataset`` 生成的 ``eval_dataloader``
        """
        if self.engine is None:
            self.init_engine()
        if self.data_provider is not None and dist.get_rank() == 0:
            if not self.data_provider.provider_started:
                self.data_provider.start_provider()
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
                self.data_provider_handler()
                self.engine.eval()
                with torch.no_grad():
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
        :param batch: 一个 batch 的数据，类型为长度为 2 的 ``Tuple``，其中第一个元素为 ``input_ids``，第二个元素为 ``labels``

            .. note::

                根据提供的 ``dataset`` 和 ``collate_fn`` 的不同，``labels`` 的类型也会有所不同。
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        input_ids, labels = batch
        if isinstance(evaluator.engine, PipelineEngine):
            generation_model = PipelineGenerationMixin(
                engine=evaluator.engine
            )
        else:
            generation_model = evaluator.engine.module
        generated_ids = generation_model.generate(input_ids=input_ids.cuda(), attention_mask=torch.ones_like(input_ids).cuda(), 
                                              generation_config=evaluator.generation_config)
        return {
            "generated_ids": generated_ids,
            # **labels
        }

    def data_provider_handler(self):
        """当初始化 :class:`collie.Evaluator` 的过程中提供了 ``data_provider`` 时会使用此方法。
            ``data_provider`` 中维持一个异步队列 ``queue.Queue``，该方法会不断从中取出数据，放入模型中进行生成
        """
        if self.data_provider is None:
            return None
        has_data = torch.tensor(False).cuda()
        input_ids = None
        if dist.get_rank() == 0:
            input_ids = self.data_provider.get_data()
            if input_ids is not None:
                has_data = ~has_data
                input_ids = input_ids.cuda()
        dist.broadcast(has_data, 0)
        if not has_data:
            return
        input_ids = broadcast_tensor(input_ids, src=0)
        if isinstance(self.engine, PipelineEngine):
            generation_model = PipelineGenerationMixin(
                engine=self.engine
            )
        else:
            generation_model = self.engine.module
        if not generation_model.can_generate():
            return
        use_stream = self.data_provider.stream
        streamer = _GenerationStreamer(server=self.data_provider)
        generated_ids = generation_model.generate(
            input_ids=input_ids.cuda(), 
            attention_mask=torch.ones_like(input_ids).cuda(), 
            generation_config=self.generation_config,
            streamer=streamer if use_stream else None
        )
        if not use_stream:
            self.data_provider.put_feedback(generated_ids[0].cpu())
        
class PerplexityEvaluator(Evaluator):
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
        :param batch: 一个 batch 的数据，类型为长度为 2 的 ``Tuple``，其中第一个元素为 ``input_ids``，第二个元素为 ``labels``

            .. note::

                根据提供的 ``dataset`` 和 ``collate_fn`` 的不同，``labels`` 的类型也会有所不同。
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        if evaluator.config.pp_size > 1:
            logits = evaluator.engine.eval_batch(batch)
        else:
            input_ids, labels = batch
            # concat prompt labels for p-tuning
            if evaluator.config.peft_config and evaluator.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
                batch_size = input_ids.shape[0]
                if "labels" in labels.keys():
                    prefix_labels = torch.full((batch_size, evaluator.config.peft_config.num_virtual_tokens), -100).to(labels["labels"].device)
                    labels["labels"] = torch.cat((prefix_labels, labels["labels"]), dim=1)
            logits = evaluator.engine(input_ids=input_ids.cuda()).logits
        ppl = evaluator.loss_fn(logits, batch[1])
        return {
            "ppl": ppl.detach().clone().view(1,).cuda(),
            # **{key: value.cuda() for key, value in batch[1].items() if isinstance(value, torch.Tensor)}
        }
        
class ClassficationEvaluator(PerplexityEvaluator):
    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Tuple) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 2 的 ``Tuple``，其中第一个元素为 ``input_ids``，第二个元素为 ``labels``

            .. note::

                根据提供的 ``dataset`` 和 ``collate_fn`` 的不同，``labels`` 的类型也会有所不同。
    
        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        input_ids_group, labels_group = batch
        assert isinstance(input_ids_group, tuple), "input_ids must be a list for classification task."
        pred = torch.zeros((input_ids_group[0].shape[0], len(input_ids_group)))
        for idx, input_ids in enumerate(input_ids_group):
            assert isinstance(input_ids, torch.Tensor), "input_ids must be a list of torch.Tensor for classification task."
            labels = {"labels": labels_group["labels"][idx], 
                      **{key: value for key, value in labels_group.items() if key != "labels"}}
            if evaluator.config.pp_size > 1:
                logits = evaluator.engine.eval_batch((input_ids, labels))
            else:
                # concat prompt labels for p-tuning
                if evaluator.config.peft_config and evaluator.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
                    batch_size = input_ids.shape[0]
                    if "labels" in labels.keys():
                        prefix_labels = torch.full((batch_size, evaluator.config.peft_config.num_virtual_tokens), -100).to(labels["labels"].device)
                        labels["labels"] = torch.cat((prefix_labels, labels["labels"]), dim=1)
                logits = evaluator.engine(input_ids=input_ids.cuda()).logits
            for sample_idx in range(input_ids.shape[0]):
                pred[sample_idx, idx] = evaluator.loss_fn(logits[sample_idx: sample_idx + 1, :], labels["labels"][sample_idx: sample_idx + 1, :]).detach().cpu().item()
        pred = pred.argmin(dim=1)
        return {
            "pred": pred.cuda(),
            "target": labels_group["target"].squeeze(1).cuda(),
            # **{key: value.cuda() for key, value in labels.items() if isinstance(value, torch.Tensor)}
        }
