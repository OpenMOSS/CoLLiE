import json
import math
import os
import warnings

import deepspeed
import torch
from deepspeed.runtime.zero import GatheredParameters
from peft.tuners.prefix_tuning import PrefixEncoder
from transformers import PreTrainedModel
from transformers.utils import ContextManagers

from collie import CollieConfig
from collie.driver.io import IODriver
from peft import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PeftModel,
    PeftModelForCausalLM,
    PeftType,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
    PromptLearningConfig,
    PromptTuningInit,
    TaskType,
    set_peft_model_state_dict,
)


def patch_peft_model(collie_config):
    def _setup_prompt_encoder(self, adapter_name):
        config = self.peft_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = (
                2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
            )
        if isinstance(self.base_model, PreTrainedModel):
            from .dist_utils import is_zero3_enabled

            for named_param, value in list(self.base_model.named_parameters()):
                contexts = []
                if is_zero3_enabled(collie_config):
                    contexts.append(GatheredParameters(value))
                with ContextManagers(contexts):
                    if value.shape[0] == self.base_model.config.vocab_size:
                        self.word_embeddings = self.base_model.get_submodule(
                            named_param.replace(".weight", "")
                        )
                        break
        else:
            self.word_embeddings = self.base_model.get_input_embedding()[1]
        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    PeftModel._setup_prompt_encoder = _setup_prompt_encoder

    def inner_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = (
                torch.ones(batch_size, peft_config.num_virtual_tokens)
                .to(self.device)
                .to(attention_mask.dtype)
            )
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            )
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )
        from .dist_utils import env

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, past_key_values=past_key_values, **kwargs
            )
        elif env.pp_rank == 0:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, peft_config.num_virtual_tokens), -100
                ).to(self.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        else:
            return self.base_model(input_ids=input_ids, **kwargs)

    def outter_forward(self, *args, **kwargs):
        from collie.module import PipelineModel

        if isinstance(self.get_base_model(), PipelineModel) and getattr(
            self.get_base_model(), "inner_forward"
        ):
            return self.get_base_model()(*args, **kwargs)
        else:
            return inner_forward(self, *args, **kwargs)

    PeftModelForCausalLM.forward = outter_forward

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.PREFIX_TUNING:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if (
                model_kwargs.get("past_key_values", None) is None
                and kwargs.get("past_key_values", None) is None
                and peft_config.peft_type == PeftType.PREFIX_TUNING
            ):
                past_key_values = self.get_prompt(
                    batch_size=model_kwargs["input_ids"].shape[0]
                )

                if self.base_model_torch_dtype is not None:
                    # handle the case for Bloom where it outputs tuple of tuples
                    if isinstance(past_key_values[0], tuple):
                        past_key_values = tuple(
                            tuple(
                                past_key_value.to(self.base_model_torch_dtype)
                                for past_key_value in past_key_value_tuple
                            )
                            for past_key_value_tuple in past_key_values
                        )
                    elif isinstance(past_key_values, torch.Tensor):
                        past_key_values = past_key_values.to(
                            self.base_model_torch_dtype
                        )
                    else:
                        past_key_values = tuple(
                            past_key_value.to(self.base_model_torch_dtype)
                            for past_key_value in past_key_values
                        )

                model_kwargs["past_key_values"] = past_key_values
            else:
                if (
                    model_kwargs.get("past_key_values", None) is None
                    and kwargs.get("past_key_values", None) is None
                    and self.word_embeddings is not None
                ):
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(
                        batch_size=model_kwargs["input_ids"].shape[0]
                    )
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat(
                        (prompts, inputs_embeds), dim=1
                    )
                    model_kwargs["input_ids"] = None

        return model_kwargs

    PeftModelForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation


def patch_prompt_tuning():
    def __init__(self, config, word_embeddings):
        super(PromptEmbedding, self).__init__()

        total_virtual_tokens = (
            config.num_virtual_tokens * config.num_transformer_submodules
        )
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer, LlamaTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(
                    config.tokenizer_name_or_path
                )
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]

            if word_embeddings is not None:
                word_embeddings.cuda()  # patched here
                input_ids = torch.LongTensor(init_token_ids).unsqueeze(0).cuda()
                input_ids = torch.flatten(input_ids, start_dim=1)
                word_embedding_weights = (
                    word_embeddings(input_ids).detach().clone()
                )  # patched here
                word_embedding_weights = word_embedding_weights.to(torch.float32)
                word_embedding_weights = word_embedding_weights.view(
                    word_embedding_weights.shape[1:]
                )
                self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    PromptEmbedding.__init__ = __init__


def patch_peft(config):
    """
    改写 ``peft`` 的 `PeftModel` 和 `PromptEmbedding`。

    用于适应 **CoLLiE** 的训练和过程。
    """
    patch_peft_model(config)
    patch_prompt_tuning()


def _is_name_in_current_rank(name):
    # TODO convert hf to pp
    import re

    search = re.search("\.[0-9]+\.", name)
    if search is None:
        # 不可能走到这里
        raise ValueError(f"{name} is not a pipeline state key.")
    layer_idx = int(search.group()[1:-1])
    from .dist_utils import env

    return layer_idx in env.pipeline_layers_idx


def pp_merge_peft(path, prefix, io_driver):
    """
    在 pp 情况下将分开保存的 peft 合并到同一个文件
    """
    from .dist_utils import env

    if env.pp_size == 1:
        return
    full_dict = {}
    for pp in range(env.pp_size):
        cur_name = os.path.join(path, f"{prefix}_{pp}.bin")
        full_dict.update(io_driver.load(cur_name, "b"))
        io_driver.delete(cur_name)
    io_driver.save(full_dict, os.path.join(path, f"{prefix}.bin"))


def _split_peft(state: dict, model):
    """
    在 pp 时选取当前 rank 的 key
    """
    from .dist_utils import env

    if env.pp_size == 1:
        return state
    if isinstance(model.active_peft_config, PromptLearningConfig):
        prefix = "base_model."
    else:
        prefix = "base_model.model."
    pipeline_model = model.get_base_model()
    for name in list(state.keys()):
        name_pp = prefix + pipeline_model.name_to_pipeline(name[len(prefix) :])
        if _is_name_in_current_rank(name_pp):
            state[name_pp] = state[name]
        state.pop(name)

    return state


def load_peft(
    model: PeftModel,
    config: CollieConfig,
    path: str,
    adapter_name="default",
    is_trainable: bool = False,
    protocol: str = "file",
):
    """
    加载 adapter 部分权重，当未使用 ``peft`` 时，该方法等同于 ``load_model``

    :param path: 模型保存路径
    :param adapter_name: 当前加载的 adapter 名称
    :param is_trainable: 是否允许加载的 adapter 进行训练
    :param process_exclusion:
    """
    io_driver = IODriver.from_protocol(protocol)

    save_dir = path if adapter_name == "default" else os.path.join(path, adapter_name)
    peft_config_dict = json.loads(
        io_driver.load(os.path.join(save_dir, "adapter_config.json"), mode="j")
    )
    loaded_peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_config_dict["peft_type"]]()
    for key, value in peft_config_dict.items():
        if hasattr(loaded_peft_config, key):
            setattr(loaded_peft_config, key, value)
    if isinstance(loaded_peft_config, PromptLearningConfig) and is_trainable:
        raise ValueError(
            "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
        )
    else:
        loaded_peft_config.inference_mode = not is_trainable
    # 这里在engine影初始化了之后再添加好像确实会出问题
    if loaded_peft_config.peft_type != model.peft_type:
        raise ValueError(
            f"Cannot combine adapters with different peft types. "
            f"Found {model.peft_type} and "
            f"{loaded_peft_config.peft_type}."
        )
    if adapter_name not in model.peft_config:
        raise ValueError(
            f"Adapter `{adapter_name}` is not found in current "
            "model, please check your checkpoint."
        )
    name = f"adapter_model.bin"
    assert io_driver.exists(os.path.join(path, name)), f"{name} does not exist."
    loaded_state_dict = io_driver.load(os.path.join(path, name), mode="rb")
    if loaded_peft_config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        loaded_state_dict = _split_peft(loaded_state_dict, model)
    if isinstance(loaded_peft_config, PromptLearningConfig):
        parameters = [model.prompt_encoder[adapter_name].embedding.weight]
    else:
        parameters = [
            param
            for name, param in model.named_parameters()
            if any(
                [
                    name.replace(f"{adapter_name}.", "") == k
                    for k in loaded_state_dict.keys()
                ]
            )
        ]

    contexts = []
    from .dist_utils import env, is_zero3_enabled

    if is_zero3_enabled(config):
        contexts.append(deepspeed.zero.GatheredParameters(parameters, modifier_rank=0))
    with ContextManagers(contexts):
        if env.dp_rank == 0 or not is_zero3_enabled(config):
            set_peft_model_state_dict(
                model, loaded_state_dict, adapter_name=adapter_name
            )
