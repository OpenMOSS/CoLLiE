# TODO delete
import sys
sys.path.append("../../../")
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import NewGELUActivation
from deepspeed.pipe import LayerSpec

from collie.log import logger
from collie.module import (ColumnParallelLinearWithoutBias,
                           RowParallelLinearWithoutBias,
                           VocabParallelEmbedding)
from .utils import apply_rotary_pos_emb, create_sinusoidal_positions
from ..base import BaseModel

class MossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        max_positions = args.n_positions
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )

        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)

        self.embed_dim = args.n_embd
        self.num_attention_heads = args.n_head
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.qkv_proj = ColumnParallelLinearWithoutBias(
            self.embed_dim, self.embed_dim * 3, bias=False, gather_output=True
        )

        self.out_proj = RowParallelLinearWithoutBias(
            self.embed_dim, self.embed_dim, bias=False, input_is_parallel=False
            
        )
        self.rotary_dim = args.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        qkv = self.qkv_proj(hidden_states)
        # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions

        sincos = embed_positions[position_ids]
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present


class MossMLP(nn.Module):
    def __init__(self, intermediate_size, args):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = args.n_embd

        self.fc_in = ColumnParallelLinearWithoutBias(
            embed_dim, intermediate_size, gather_output=False)
        self.fc_out = RowParallelLinearWithoutBias(
            intermediate_size, embed_dim, input_is_parallel=True)

        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(args.resid_pdrop)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MossBlock(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        inner_dim = args.n_inner if args.n_inner is not None else 4 * args.n_embd
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.attn = MossAttention(args)
        self.mlp = MossMLP(inner_dim, args)
        self.args = args
        self.idx = layer_idx

        self.layer_past = None

    def _forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)

    def forward(self, hidden_states):

        if self.layer_past is None or self.training:
            past_length = 0
        else:
            past_length = self.layer_past[0].size(-2)

        end_pos = hidden_states.shape[1]
        position_ids = torch.arange(
            past_length, end_pos + past_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).view(-1, end_pos)

        if self.args.checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward),
                hidden_states,
                None,
                None,
                position_ids,
            )
        else:
            outputs = self._forward(
                hidden_states,
                position_ids=position_ids,
                layer_past=self.layer_past,
                attention_mask=None,
                head_mask=None,
                use_cache=not self.training,
            )

        if not self.training:
            self.layer_past = outputs[1]

        # hidden_states 
        return outputs[0]


class MossModel(BaseModel):
    # MossForCausalLM
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.n_embd
        self.vocab_size = args.vocab_size
        self.wte = VocabParallelEmbedding(args.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = nn.ModuleList([
            MossBlock(args, i) for i in range(args.n_layer)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=args.layer_norm_epsilon)
        self.lm_head = ColumnParallelLinearWithoutBias(args.n_embd,
                                                       args.vocab_size)

    def forward(self, input_ids):
        inputs_embed = self.wte(input_ids)
        hidden_states = self.drop(inputs_embed)

        for l in self.h:
            hidden_states = l(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    
    @classmethod
    def pipeline_layers(cls, args):
        layers = [
            VocabParallelEmbedding(args.vocab_size, args.n_embd),
            nn.Dropout(args.embd_pdrop),
        ]
        layers += [
            LayerSpec(MossBlock, args, i) for i in range(args.n_layer)
        ]
        layers += [
            nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon),
            ColumnParallelLinearWithoutBias(args.n_embd, args.vocab_size)
        ]

        return layers
    
    @staticmethod
    def load_parallel_state_dict(args, path):
        """
        Load state_dict from ``path``.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        raise NotImplementedError(
            "Every model should implement `load_parallel_state_dict` "
            "to properly load a state dict for the cuurent rank."
        )
    
    @staticmethod
    def save_parallel_state_dict(args, path):
        """
        Save state_dict to ``path``.
        """
        raise NotImplementedError(
            "Every model should implement `save_parallel_state_dict` "
            "to properly save a state dict for the cuurent rank."
        )
