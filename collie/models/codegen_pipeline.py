import os
import re
from typing import Tuple

import torch
from torch import nn
from transformers.models.codegen.modeling_codegen import CodeGenBlock, CodeGenModel, CodeGenPreTrainedModel, CodeGenForCausalLM
from transformers import CodeGenTokenizer, CodeGenConfig

try:
    from deepspeed.pipe import LayerSpec, PipelineModule
except ModuleNotFoundError:
    PipelineModule = None
    LayerSpec = None

from collie.log import logger
from .checkpoint_engine import get_checkpoint_engine, CollieSDLoader

class EmbeddingPipe(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(EmbeddingPipe, self).__init__(*args, **kwargs)

    def forward(self, input_tuple):
        # Assume the first is `input_ids`
        # input_ids, use_cache, past_keys, past_values
        input_ids = input_tuple[0]
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = super().forward(input_ids)
        # input_ids, output_shape, use_cache, past_key, past_values
        # or input_ids, output_shape, use_cache
        return (inputs_embeds, torch.tensor(input_shape).cuda()) + input_tuple[1:]


class LayerNormPipe(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPipe, self).__init__(*args, **kwargs)

    def forward(self, input_tuple):
        hidden_states, position_ids, output_shape, use_cache_tensor = input_tuple[:4]
        use_cache = use_cache_tensor.item()
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(output_shape.tolist())

        # find past_key_value
        if use_cache:
            past_key_values = input_tuple[4:6]
        else:
            past_key_values = ()

        return (hidden_states, ) + past_key_values

class CodeGenLMHead(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(CodeGenLMHead, self).__init__(*args, **kwargs)

    def forward(self, input_tuple):
        hidden_states = input_tuple[0]
        output = super().forward(hidden_states)

        if len(input_tuple) == 1:
            return output
        else:
            return (output, ) + input_tuple[1:]

class CodeGenBlockPipe(CodeGenBlock):
    def __init__(self, config, gradient_checkpointing, layer_idx):
        super(CodeGenBlockPipe, self).__init__(config)
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.idx = layer_idx

    def forward(self, input_tuple):
        hidden_states, position_ids, output_shape, use_cache_tensor = input_tuple[:4]
        assert len(use_cache_tensor.shape) == 0, use_cache_tensor.shape
        use_cache = use_cache_tensor.bool().item()

        if self.gradient_checkpointing and self.training:
            use_cache = False
            use_cache_tensor = torch.tensor(0).to(hidden_states.device)

        if use_cache and self.idx != 0:
            # Layers before has generated key-value, and they are placed
            # at position 4 and 6
            # note that if this is the first layer, there are no keys and
            # values generated.
            cur_past_key_values = input_tuple[4:6]
            # past_key_value used by users, they are placed at position
            # 6 and 8 (may be a empty tuple)
            past_key_values = input_tuple[6:8]
        else:
            # layers before did not generate key-value
            cur_past_key_values = ()
            # used by users
            past_key_values = input_tuple[4:6]

        if len(past_key_values) == 0:
            layer_past = None
        else:
            assert len(past_key_values) == 2, len(past_key_values)
            past_key, past_value = past_key_values
            layer_past = (past_key[self.idx], past_value[self.idx])
        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(super().forward),
                hidden_states,
                None,
                None,
                position_ids,
            )
        else:
            outputs = super().forward(
                hidden_states,
                position_ids=position_ids,
                layer_past=layer_past,
                attention_mask=None,
                head_mask=None,
                use_cache=use_cache,
                output_attentions=False,
            )

        if use_cache:
            key, value = outputs[1]
            if len(cur_past_key_values) != 0:
                # concat key-value so that the shape finally reaches
                # [layers, batch_size, seq_len]
                assert len(cur_past_key_values) == 2, len(cur_past_key_values)
                cur_past_key, cur_past_value = cur_past_key_values
                cur_past_key_values = (
                    torch.concat([cur_past_key, key.unsqueeze(0)], dim=0),
                    torch.concat([cur_past_value, value.unsqueeze(0)], dim=0),
                )
            else:
                # first layer
                cur_past_key_values = (
                    key.unsqueeze(0), value.unsqueeze(0)
                )
        else:
            cur_past_key_values = ()

        # hidden_states, position_ids, use_cache, 
        # past_key, past_value, past_key(user), past_value(user)
        return (outputs[0], position_ids, output_shape, use_cache_tensor) + cur_past_key_values + past_key_values

class CodeGenModelPipe(CodeGenModel):
    def __init__(self, config):
        if LayerSpec is None:
            raise ModuleNotFoundError(
                "Detected DeepSpeed not installed. Please see "
                "https://github.com/microsoft/DeepSpeed ."
            )
        CodeGenPreTrainedModel.__init__(self, config)
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = EmbeddingPipe(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        # LayerSpec 会推迟，所以要把 gradient_checkpointing 传进去
        # 否则会在 post_init 中被丢掉
        # 通过 to_diff_dict 保存的 config 没有 graidnet_checkpoing
        gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        self.h = [
            LayerSpec(
                CodeGenBlockPipe, config, gradient_checkpointing, i
            ) for i in range(config.n_layer)
        ]
        self.ln_f = LayerNormPipe(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def to_layers(self):
        def pre_forward(input_tuple):
            # 输入为 embedding 输出的 input_embeds, input_shape
            inputs_embeds, input_shape, use_cache = input_tuple[0:3]
            # (past_key, past_values)
            past_key_values = input_tuple[3:]
            hidden_states = inputs_embeds
            hidden_states = self.drop(hidden_states)
            output_shape = input_shape.tolist() + [hidden_states.size(-1),]

            if len(past_key_values) == 0:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            return (hidden_states, position_ids, torch.tensor(output_shape).cuda(), use_cache) + past_key_values
        
        layers = [self.wte, pre_forward]
        for i in range(len(self.h)):
            layers.append(self.h[i])

        layers.append(self.ln_f)

        return layers

class CodeGenForCausalLMPipe(CodeGenForCausalLM):
    def __init__(self, config):
        CodeGenPreTrainedModel.__init__(self, config)
        self.transformer = CodeGenModelPipe(config)
        self.lm_head = CodeGenLMHead(config.n_embd, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def to_layers(self):
        layers = self.transformer.to_layers()
        layers.append(self.lm_head)

        return layers

def get_codegen_pipeline(
        model_name_or_path, tokenizer_path=None, config=None,
        one_by_one=False, protocol="file", **kwargs
    ) -> Tuple[PipelineModule, CodeGenTokenizer]:
    """
    Load pretrained codegen pipeline model and tokenizer.

    :param model_name_or_path:
    :param tokenizer_path: Pretrained tokenizer. If ``tokenizer_path`` is
        ``None``, we will set ``model_name_or_path`` as ``tokenizer_path``.
    :param config: Config to initialize model. If ``config`` is None, we will
        use ``CodeGenConfig.from_pretrained(model_name_or_path)`` as model's
        config.
    :param one_by_one: If load checkpoints stage by stage to avoid CPU OOM.
    :param kwargs: Kwargs to initialize ``deepspeed.pipe.PipelineModule``.
    :return: (PipelineModule, CodeGenTokenizer)
    """
    if PipelineModule is None:
        raise ModuleNotFoundError(
            "Detected DeepSpeed not installed. Please see "
            "https://github.com/microsoft/DeepSpeed ."
        )
    if tokenizer_path is None:
        tokenizer_path = model_name_or_path
    if config is None:
        config = get_codgen_config(model_name_or_path, protocol)

    # TODO currently tokenizer doesn't support load from s3.
    tokenizer = CodeGenTokenizer.from_pretrained(tokenizer_path)

    pipeline_model = PipelineModule(
        CodeGenForCausalLMPipe(config=config).to_layers(),
        **kwargs,
    )
    pipeline_model._collie_config = config
    load_state_dict(pipeline_model, model_name_or_path, one_by_one, protocol)

    return pipeline_model, tokenizer

def get_codgen_config(config_path, protocol, **kwargs):
    """
    Load codegen config from file or s3 group.

    :param config_path:
    :param kwargs: Kwargs to set config.
    """
    if protocol == "s3":
        ckpt_engine = get_checkpoint_engine(protocol)
        config_dict = ckpt_engine.load_json(
            os.path.join(config_path, "config.json")
        )
        config = CodeGenConfig.from_dict(config_dict, **kwargs)
    else:
        # May use remote_url here, so temporarily don't use engine for file 
        # here.
        config = CodeGenConfig.from_pretrained(config_path, **kwargs)

    return config

def load_state_dict(
        pipeline_model, ckpt_dir, one_by_one=False, protocol="file",
    ):
    """
    Load pretrained checkpoint that downloaded from huggingface

    :param pipeline_model: deepspeed.pipe.PipelineModule
    :param ckpt_dir: checkpoints save path.
    :param one_by_one: If load checkpoints stage by stage to avoid CPU OOM.
    :param protocol: ['file', 's3], determinate where the checkpoints
        are from (from local or s3 group).
    """
    assert isinstance(pipeline_model, PipelineModule)
    # find sharded index
    ckpt_engine = get_checkpoint_engine(protocol)
    is_collie = ckpt_engine.isfile(os.path.join(ckpt_dir, "pipeline.json"))
    is_sharded = ckpt_engine.isfile(os.path.join(ckpt_dir, "pytorch_model.bin.index.json"))
    if is_collie and is_sharded:
        raise RuntimeError(
            "Detected both ColliE checkpoint and sharded checkpoint in {}."
            .format(ckpt_dir)
        )
    if is_collie:
        load_state_dict_from_collie(
            pipeline_model, ckpt_dir, ckpt_engine, one_by_one
        )
    elif is_sharded:
        load_state_dict_from_sharded_ckpt(
            pipeline_model, ckpt_dir, ckpt_engine, one_by_one
        )
    else:
        load_state_dict_from_normal(
            pipeline_model, ckpt_dir, ckpt_engine, one_by_one
        )

def load_state_dict_from_collie(
        pipeline_model, ckpt_dir, ckpt_engine, one_by_one=False,
    ):
    """
    Load huggingface sharded checkpoints to pipeline model.

    :param pipeline_model: deepspeed.pipe.PipelineModule
    :param ckpt_dir: checkpoints save path.
    :param one_by_one: This parameter us unused.
    :param ckpt_engine: ``CheckpointEngine``
    :param one_by_one: If load checkpoints stage by stage to avoid CPU OOM.
    """
    isinstance(pipeline_model, PipelineModule)
    logger.info("Loading Model from ColliE chackpoints")
    ckpt_info = ckpt_engine.load_json(os.path.join(ckpt_dir, "pipeline.json"))
    num_stages = ckpt_info["num_stages"]
    n_layers = ckpt_info["n_layers"]
    if n_layers != pipeline_model._num_layers:
        raise ValueError(
            f"Num of layers in {ckpt_dir}({n_layers}) is not equal to "
            f"current model({pipeline_model._num_layers})"
        )
    
    pipeline_load_state_dir(pipeline_model, ckpt_dir, ckpt_engine)

def load_state_dict_from_sharded_ckpt(
        pipeline_model, ckpt_dir, ckpt_engine, one_by_one=False,
    ):
    """
    Load huggingface sharded checkpoints to pipeline model.

    :param pipeline_model: deepspeed.pipe.PipelineModule
    :param ckpt_dir: checkpoints save path.
    :param ckpt_engine: ``CheckpointEngine``
    :param one_by_one: If load checkpoints stage by stage to avoid CPU OOM.
    """
    assert isinstance(pipeline_model, PipelineModule)
    logger.info("Loading Model from sharded chackpoints")
    index_file = ckpt_engine.load_json(os.path.join(ckpt_dir, "pytorch_model.bin.index.json"))
    weight_map = index_file["weight_map"]
    total_size = index_file["metadata"]["total_size"]
    parts = pipeline_model.parts
    cur_load_ckpt = ""
    for cur_stage in range(pipeline_model.num_stages):
        state_dict = None
        if one_by_one:
            logger.info(f"Stage {cur_stage} loading...")
            torch.distributed.barrier()
        if cur_stage != pipeline_model.stage_id:
            continue
        for pipe_key in pipeline_model.state_dict().keys():
            key = convert_pipeline_key_to_normal(pipe_key, parts)
            if cur_load_ckpt != weight_map[key]:
                del state_dict
                cur_load_ckpt = weight_map[key]
                state_dict = ckpt_engine.load(os.path.join(ckpt_dir, cur_load_ckpt))

            param = pipeline_model
            for attr in pipe_key.split("."):
                # get attribute
                param = getattr(param, attr)
            copy_param(param, key, state_dict)

            assert torch.equal(
                pipeline_model.state_dict()[pipe_key].detach().cpu(),
                state_dict[key].detach().cpu()
            ), f"{pipe_key} - {key}"
    
        del state_dict

def load_state_dict_from_normal(
        pipeline_model, ckpt_dir, ckpt_engine, one_by_one=False,
    ):
    """
    Load pretrained checkpoint that downloaded from huggingface

    :param pipeline_model: deepspeed.pipe.PipelineModule
    :param ckpt_dir: checkpoints save path.
    :param ckpt_engine: ``CheckpointEngine``
    :param one_by_one: If load checkpoints stage by stage to avoid CPU OOM.
    """
    assert isinstance(pipeline_model, PipelineModule)
    logger.info("Loading Model from normal chackpoints")
    parts = pipeline_model.parts
    for cur_stage in range(pipeline_model.num_stages):
        if one_by_one:
            logger.info(f"Stage {cur_stage} loading...")
            torch.distributed.barrier()
        if cur_stage != pipeline_model.stage_id:
            continue
        state_dict = ckpt_engine.load(os.path.join(ckpt_dir, "pytorch_model.bin"))
        for pipe_key in pipeline_model.state_dict().keys():
            key = convert_pipeline_key_to_normal(pipe_key, parts)

            param = pipeline_model
            for attr in pipe_key.split("."):
                # get attribute
                param = getattr(param, attr)
            copy_param(param, key, state_dict)

            assert torch.equal(
                pipeline_model.state_dict()[pipe_key].detach().cpu(),
                state_dict[key].detach().cpu()
            ), f"{pipe_key} - {key}"

        del state_dict

def pipeline_load_state_dir(pipeline_model, load_dir, checkpoint_engine):
    # PipelineModule.load_state_dir
    for idx, layer in enumerate(pipeline_model.forward_funcs):
        # Functions, etc. will not have state_dicts
        if not hasattr(layer, 'load_state_dict'):
            continue

        # get all checkpoint files for the layer.
        ### PipelineModule.ckpt_layer_path_list
        # make it compatible with s3 group
        local_layer_idx = idx + pipeline_model._local_start
        layer_ckpt_name = f"layer_{local_layer_idx:02d}-.*model_states.pt"
        filelist = checkpoint_engine.list(load_dir)

        model_ckpt_list = []
        for f in filelist:
            if re.fullmatch(layer_ckpt_name, f):
                model_ckpt_list.append(os.path.join(load_dir, f))
        model_ckpt_list.sort()
        ###

        mp_rank = pipeline_model._grid.get_slice_parallel_rank()
        mp_world_size = pipeline_model._grid.get_slice_parallel_world_size()

        sd_loader = CollieSDLoader(
            model_ckpt_list, version=2.0, checkpoint_engine=checkpoint_engine
        )
        load_path, checkpoint, _ = sd_loader.load(mp_world_size, mp_rank)

        layer.load_state_dict(checkpoint)

    pipeline_model._synchronize_tied_weights()

def copy_param(param, key, state_dict):
    input_param = state_dict[key]
    """copy from module._load_from_state_dict"""
    if not torch.overrides.is_tensor_like(input_param):
        logger.warning('While copying the parameter named "{}", '
              'expected torch.Tensor or Tensor-like object '
              'from checkpoint but received {}'
              .format(key, type(input_param)))

    is_param_lazy = torch.nn.parameter.is_lazy(param)
    # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
    if not is_param_lazy and len(param.shape) == 0 \
            and len(input_param.shape) == 1:
        input_param = input_param[0]

    if not is_param_lazy and input_param.shape != param.shape:
        # local shape should match the one in checkpoint
        logger.warning('size mismatch for {}: copying a param with shape {} from '
              'checkpoint, the shape in current model is {}.'
              .format(key, input_param.shape, param.shape))
    try:
        with torch.no_grad():
            param.copy_(input_param)
    except Exception as ex:
        logger.warning('While copying the parameter named "{}", '
              'whose dimensions in the model are {} and '
              'whose dimensions in the checkpoint are {}, '
              'an exception occurred : {}.'
              .format(key, param.size(), input_param.size(), ex.args))
        
def convert_pipeline_key_to_normal(pipe_key, parts):
    """
    Convert pipeline model's key to normal model.

    Examples: 15.ln_1.bias -> transformer.h.15.ln_1.bias
    """
    pipe_key_seq = pipe_key.split(".")
    layer_pipe_idx = int(pipe_key_seq[0])
    if layer_pipe_idx == 0:
        # 0 -> embedding
        # 1 -> preforward
        key = 'transformer.wte.weight'
    elif layer_pipe_idx == parts[-1] - 2:
        # one before last -> LayerNorm ln_f
        param_type = pipe_key_seq[-1] # weight or bias
        key = 'transformer.ln_f.' + param_type
    elif layer_pipe_idx == parts[-1] - 1:
        # last -> Linear lm_head
        param_type = pipe_key_seq[-1] # weight or bias
        key = 'lm_head.' + param_type
    else:
        # blocks
        block_idx = layer_pipe_idx - 2
        # 15.ln_1.bias -> transformer.h.15.ln_1.bias
        attr_list = ['transformer', 'h', str(block_idx)] + pipe_key_seq[1:]
        key = '.'.join(attr_list)

    return key
