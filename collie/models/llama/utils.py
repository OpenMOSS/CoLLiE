import os
import re
import tqdm
import time
import json
import copy
import torch
import shutil
from einops import rearrange
import torch.distributed as dist
from collections import OrderedDict
from megatron.core import parallel_state
from collie.driver.io.file import FileIODriver
from collie.driver.io.petrel import PetrelIODriver
from collie.models.llama.arguments import LlamaArguments

def load_parallel_state_dict(folder: str,
                             protocol: str = 'file',
                             format: str = 'hf',
                             offload: bool = False,
                             offload_folder: str = "./offload",
                             args: LlamaArguments = LlamaArguments()
                             ):
    if not dist.is_initialized():
        return load_state_dict(folder, protocol, format, args)
    tempdir = [""]
    if offload:
        tempdir[0] = offload_folder
    else:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            tempdir[0] = f"/dev/shm/Collie-{round(time.time() * 1000)}/"
        dist.broadcast_object_list(tempdir, src=0)
    os.makedirs(tempdir[0], exist_ok=True)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        state_dict = load_state_dict(folder, protocol, format, args)
        if "COLLIE_PP_PARTS" in os.environ.keys():
            for key in list(state_dict.keys()):
                if key.startswith("layers"):
                    layer = int(key.split(".")[2])
                    state_dict[key.replace(f"layers.{layer}", f"{layer + 1}")] = state_dict.pop(key)
                if key.endswith("embed_tokens.weight"):
                    state_dict["tied_modules.embed_tokens.weight"] = state_dict.pop(key)
                if key.endswith("norm.weight"):
                    state_dict[f"{args.num_hidden_layers + 1}.weight"] = state_dict.pop(key)
                if key.endswith("lm_head.weight"):
                    state_dict.pop(key)
        for key in state_dict.keys():
            if key.endswith("q_proj.weight") \
                or key.endswith("k_proj.weight") \
                    or key.endswith("v_proj.weight") \
                        or key.endswith("gate_proj.weight") \
                            or key.endswith("up_proj.weight") \
                                or key.endswith("embed_tokens.weight"):
                                    state_dict[key] = list(torch.chunk(state_dict[key], args.tp_size, dim=0))
            if key.endswith("o_proj.weight") \
                or key.endswith("down_proj.weight"):
                    state_dict[key] = list(torch.chunk(state_dict[key], args.tp_size, dim=1))
        part_state_dicts = []
        if "COLLIE_PP_PARTS" in os.environ.keys():
            parts = json.loads(os.environ["COLLIE_PP_PARTS"])
            for stage in range(1, len(parts)):
                layers = list(range(parts[stage - 1], parts[stage]))
                part_state_dict = {key: value for key, value in state_dict.items() if any([key.startswith(f"{layer}.") for layer in layers])}
                part_state_dict.update({key: value for key, value in state_dict.items() if key.startswith("tied_modules")})
                part_state_dicts.append(part_state_dict)
        else:
            part_state_dicts = [state_dict]
        for pp_stage, part_state_dict in enumerate(part_state_dicts):
            for tp_stage in range(args.tp_size):
                file_path = os.path.join(tempdir[0], f"pp-{pp_stage}-tp-{tp_stage}.bin")
                FileIODriver.save({key: value[tp_stage] if isinstance(value, list) else value for key, value in part_state_dict.items()}, file_path)
        del state_dict
        del part_state_dicts
    dist.barrier()
    if "COLLIE_PP_RANK" in os.environ.keys():
        pp_stage = int(os.environ["COLLIE_PP_RANK"])
    else:
        pp_stage = 0
    tp_stage = parallel_state.get_tensor_model_parallel_rank()
    file_path = os.path.join(tempdir[0], f"pp-{pp_stage}-tp-{tp_stage}.bin")
    state_dict = FileIODriver.load(file_path, mode="rb")
    dist.barrier()
    if os.environ.get("LOCAL_RANK", "0") == "0":
        shutil.rmtree(tempdir[0])
    return state_dict
                
def save_parallel_state_dict(state_dict: dict,
                             folder: str,
                             protocol: str = 'file',
                             offload: bool = False,
                             offload_folder: str = "./offload",
                             args: LlamaArguments = LlamaArguments()):
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
    tempdir = [""]
    if offload:
        tempdir[0] = offload_folder
    else:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            tempdir[0] = f"/dev/shm/Collie-{round(time.time() * 1000)}/"
        dist.broadcast_object_list(tempdir, src=0)
    os.makedirs(tempdir[0], exist_ok=True)
    if "COLLIE_PP_RANK" in os.environ.keys():
        pp_stage = int(os.environ["COLLIE_PP_RANK"])
    else:
        pp_stage = 0
    tp_stage = parallel_state.get_tensor_model_parallel_rank()
    file_path = os.path.join(tempdir[0], f"pp-{pp_stage}-tp-{tp_stage}.bin")
    FileIODriver.save(state_dict, file_path)
    dist.barrier()
    if os.environ.get("LOCAL_RANK", "0") == "0":
        weights = [file for file in FileIODriver.list(tempdir[0]) if file.endswith(".bin")]
        pp_tp_map = OrderedDict({weight: list(map(int, re.match(r'pp_(\d+)_tp(\d+)\.pt', weight).groups())) for weight in weights})
        pp_range = range(min([value[0] for value in pp_tp_map.values()]), max([value[0] for value in pp_tp_map.values()]) + 1)
        tp_range = range(max([value[1] for value in pp_tp_map.values()]) + 1)
    
def load_state_dict(folder: str,
                    protocol: str = 'file',
                    format: str = 'hf',
                    args: LlamaArguments = LlamaArguments()):
    assert format in ["hf", "meta"], "Only support hf and meta format"
    assert protocol in ["file", "petrel"], "Only support file and petrel protocol"
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver
    state_dict = OrderedDict()
    if format == "hf":
        if IODriver.exists(os.path.join(folder, "config.json")):
            config = json.loads(IODriver.load(os.path.join(folder, "config.json"), mode="r"))
            for key, value in {
                "vocab_size": config["vocab_size"],
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "layer_norm_epsilon": config["rms_norm_eps"]
            }.items():
                setattr(args, key, value)
        weights = [weight for weight in IODriver.list(folder) if weight.endswith(".bin")]
        with tqdm.tqdm(weights, desc="Loading state dict", total=len(weights)) as pbar:
            for idx, weight in enumerate(pbar):
                raw_state_dict = IODriver.load(os.path.join(folder, weight), mode="rb")
                for key in list(raw_state_dict.keys()):
                    if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                        raw_state_dict[key] = rearrange(
                            raw_state_dict[key],
                            "(h two t) d -> h two t d",
                            h=args.num_attention_heads,
                            two=2).transpose(1, 2).reshape(
                                args.hidden_size,
                                args.hidden_size)
                    raw_state_dict[key.replace("model.", "")] = raw_state_dict.pop(key)
                state_dict.update(raw_state_dict)
                pbar.set_postfix_str(f"Loaded {idx + 1}/{len(weights)} weights")
    elif format == "meta":
        if IODriver.exists(os.path.join(folder, "params.json")):
            params = json.loads(IODriver.load(os.path.join(folder, "params.json"), mode="r"))
            for key, value in {
                "hidden_size": params["dim"],
                "intermediate_size": params["multiple_of"] * ((int(2 * 4 * args.hidden_size / 3) + params["multiple_of"] - 1) // params["multiple_of"]),
                "num_hidden_layers": params["n_layers"],
                "num_attention_heads": params["n_heads"],
                "layer_norm_epsilon": params["norm_eps"]
            }.items():
                setattr(args, key, value)
        weights = [weight for weight in IODriver.list(folder) if weight.endswith(".pt") or weight.endswith(".pth")]
        with tqdm.tqdm(weights, desc="Loading state dict", total=len(weights)) as pbar:
            for idx, weight in enumerate(pbar):
                raw_state_dict = IODriver.load(os.path.join(folder, weight), mode="rb")
                for key in raw_state_dict.keys():
                    raw_key = key
                    key = key.replace("attention", "self_attn")
                    key = key.replace("wo", "o_proj")
                    key = key.replace("wq", "q_proj")
                    key = key.replace("wk", "k_proj")
                    key = key.replace("wv", "v_proj")
                    key = key.replace("feed_forward", "mlp")
                    key = key.replace("w1", "gate_proj")
                    key = key.replace("w2", "down_proj")
                    key = key.replace("w3", "up_proj")
                    key = key.replace("attention_norm", "input_layernorm")
                    key = key.replace("attention_norm", "post_attention_layernorm")
                    key = key.replace("tok_embeddings", "embed_tokens")
                    key = key.replace("output", "lm_head")
                    raw_state_dict[raw_key] = raw_state_dict.pop(key)
                state_dict.update(raw_state_dict)
                pbar.set_postfix_str(f"Loaded {idx + 1}/{len(weights)} weights")
    return state_dict
                    
    
    
def save_state_dict(state_doct: dict,
                    folder: str,
                    protocol: str = 'file',
                    args: LlamaArguments = LlamaArguments()):
    IODriver = FileIODriver if protocol == 'file' else PetrelIODriver