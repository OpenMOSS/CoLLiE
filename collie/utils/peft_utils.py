import os
import math

import torch
from types import MethodType
from typing import Optional
from transformers import PreTrainedModel
from peft import TaskType, PeftType, PromptEmbedding, \
    PromptEncoder, PrefixEncoder, PeftModel, PromptTuningInit, \
    PeftModelForCausalLM, PromptLearningConfig
import warnings


def patch_peft_model():
    def _setup_prompt_encoder(self, adapter_name):
        from collie.models.base import CollieModelForCausalLM
        config = self.peft_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
        self.word_embeddings = self.base_model.get_input_embedding()[1]
        #
        # if env.pp_rank > 0 and self.word_embeddings is None:
        #     class DummyEmbedding(torch.nn.Module):
        #         def forward(self, x):
        #             return torch.tensor([[0.]])
        #     self.word_embeddings = DummyEmbedding()
        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder.update(
            torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()
    PeftModel._setup_prompt_encoder = _setup_prompt_encoder

    def innter_forward(
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
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                self.device).to(attention_mask.dtype)
            attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
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

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, peft_config.num_virtual_tokens), -100).to(self.device)
                kwargs["labels"] = torch.cat(
                    (prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            from .dist_utils import env
            print(f"[DEBUG] Rank: {env.rank} input_ids.shape: {input_ids.shape}")
            print(f"[DEBUG] Rank: {env.rank} kwargs['labels'].shape: {kwargs['labels'].shape}")
            print(f"[DEBUG] Rank: {env.rank} inputs_embeds.shape: {inputs_embeds.shape}")
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        
    def outter_forward(self, *args, **kwargs):
        from collie.module import PipelineModel
        from .dist_utils import env
        if isinstance(self.get_base_model(), PipelineModel) and getattr(self.get_base_model(), "inner_forward"):
            return self.get_base_model()(*args, **kwargs)
        elif env.pp_rank > 0:
            return self.get_base_model()(*args, **kwargs)
        else:
            return innter_forward(self, *args, **kwargs)
    PeftModelForCausalLM.forward = outter_forward
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs.get("past_key_values", None) is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            if self.base_model_torch_dtype is not None:
                # handle the case for Bloom where it outputs tuple of tuples
                if isinstance(past_key_values[0], tuple):
                    past_key_values = tuple(
                        tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_value_tuple
                        )
                        for past_key_value_tuple in past_key_values
                    )
                else:
                    past_key_values = tuple(
                        past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                    )
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs
    PeftModelForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation


def patch_prompt_tuning():
    def __init__(self, config, word_embeddings):
        super(PromptEmbedding, self).__init__()

        total_virtual_tokens = config.num_virtual_tokens * \
            config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(
            total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer
            from transformers import LlamaTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.tokenizer_name_or_path)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(
                    config.tokenizer_name_or_path)
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
                input_ids = torch.LongTensor(
                    init_token_ids).unsqueeze(0).cuda()
                input_ids = torch.flatten(input_ids, start_dim=1)
                word_embedding_weights = word_embeddings(
                    input_ids).detach().clone()  # patched here
                word_embedding_weights = word_embedding_weights.to(
                    torch.float32)
                word_embedding_weights = word_embedding_weights.view(
                    word_embedding_weights.shape[1:])
                self.embedding.weight = torch.nn.Parameter(
                    word_embedding_weights)

    PromptEmbedding.__init__ = __init__


def patch_peft():
    """
        改写 ``peft`` 的 `PeftModel` 和 `PromptEmbedding`。

        用于适应 **CoLLiE** 的训练和过程。
    """
    patch_peft_model()
    patch_prompt_tuning()
