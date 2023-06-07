import os
import math

import torch
from transformers import PreTrainedModel
from peft import TaskType, PeftType, PromptEmbedding, PromptEncoder, PrefixEncoder, PeftModel, PromptTuningInit


def patch_peft_model():
    def _setup_prompt_encoder(self, adapter_name):
        config = self.peft_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            # patched for zero3
            if hasattr(value, "ds_shape"):
                if value.ds_shape[0] == self.base_model.config.vocab_size:
                    self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))

                    # all gather word_embeddings weights
                    device = torch.device(f"cuda:{os.getenv('LOCAL_RANK')}")
                    word_embeddings_weights = torch.zeros(
                        int(os.getenv('WORLD_SIZE')) * value.ds_tensor.shape[0],
                        dtype=value.ds_tensor.dtype,
                        device=device
                    )
                    torch.distributed.all_gather_into_tensor(word_embeddings_weights, value.ds_tensor)
                    word_embeddings_weights = word_embeddings_weights[:value.ds_numel].view(value.ds_shape)
                    self.word_embeddings.weight = torch.nn.Parameter(word_embeddings_weights)
                    self.word_embeddings.weight.requires_grad = False
                    break
            else:
                if value.shape[0] == self.base_model.config.vocab_size:
                    self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                    break

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


def patch_prompt_tuning():
    def __init__(self, config, word_embeddings):
        super(PromptEmbedding, self).__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
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

            word_embeddings.cuda() # patched here
            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids).cuda()).detach().clone() # patched here
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    PromptEmbedding.__init__ = __init__


def patch_peft():
    """
        改写 ``peft`` 的 `PeftModel` 和 `PromptEmbedding`。

        用于适应 **CoLLiE** 的训练和过程。
    """
    patch_peft_model()
    patch_prompt_tuning()
