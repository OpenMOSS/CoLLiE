import os
import io
import json
import shutil

import torch

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from deepspeed.runtime.state_dict_factory import MegatronSDLoader

class CollieS3Engine(CheckpointEngine):
    """
    CheckpointEngine to save and load files from s3.
    """

    def __init__(self, config_params=None):
        super(CollieS3Engine, self).__init__(config_params)

        self.protocol = "s3"
        from petrel_client.client import Client
        self.client = Client()

    def save(self, state_dict, path: str):
        with io.BytesIO() as buffer:
            torch.save(state_dict, buffer)
            buffer.seek(0)
            self.client.put(path, buffer)

        return None

    def load(self, path: str, map_location=None):
        with io.BytesIO() as buffer:
            buffer.write(self.client.get(path))
            buffer.seek(0)
            partition = torch.load(buffer, map_location=map_location)

        return partition
    
    def makedirs(self, path, exist_ok=False):
        pass
    
    def load_json(self, filename):
        with io.BytesIO() as buffer:
            buffer.write(self.client.get(filename))
            buffer.seek(0)
            json_dict = json.load(buffer)
        
        return json_dict
    
    def save_json(self, obj, filename):
        with io.BytesIO() as buffer:
            buffer.write(json.dumps(obj).encode())
            buffer.seek(0)
            self.client.put(filename, buffer)
    
    def isfile(self, path):
        return self.client.contains(path)
    
    def delete(self, path):
        if int(os.getenv("RANK", 0)) != 0:
            return
        if self.isfile(path):
            self.client.delete(path)
        else:
            # dir
            for filename in self.list(path):
                self.client.delete(os.path.join(path, filename))

    def list(self, path):
        return list(self.client.list(path))

    def commit(self, tag):
        return True
    
class CollieFileEngine(CheckpointEngine):
    """
    CheckpointEngine to save and load files from local.

    Same as ``TorchCheckpointEngine`` in ``deepspeed`` with logger removed and
    new methods added.
    """

    def __init__(self, config_params=None):
        super(CollieFileEngine, self).__init__(config_params)

        self.protocol = "file"

    def save(self, state_dict, path: str):
        torch.save(state_dict, path)
        return None

    def load(self, path: str, map_location=None):
        partition = torch.load(path, map_location=map_location)
        return partition
    
    def load_json(self, filename):
        with open(filename) as fp:
            json_dict = json.load(fp)
        return json_dict
    
    def save_json(self, obj, filename):
        with open(filename, "w") as fp:
            json.dump(obj, fp)
    
    def isfile(self, path):
        return os.path.isfile(path)
    
    def delete(self, path):
        if int(os.getenv("RANK", 0)) != 0:
            return
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

    def list(self, path):
        return os.listdir(path)

    def commit(self, tag):
        return True

def get_checkpoint_engine(protocol):
    if protocol == "s3":
        return CollieS3Engine()
    elif protocol == "file":
        return CollieFileEngine()
    else:
        raise ValueError(
            "`protocol should be one of ['file', 's3'], not {}"
            .format(protocol)
        )

class CollieSDLoader(MegatronSDLoader):
    def __init__(self, ckpt_list, version, checkpoint_engine):
        super(CollieSDLoader, self).__init__(
            ckpt_list, version, checkpoint_engine
        )

    def load(self, mp_world_size, mp_rank):
        # Delete some code to make it simple for pipeline
        # delete some default parameters
        self.module_key = None
        num_ckpt = len(self.ckpt_list)
        idx = mp_rank * num_ckpt // mp_world_size

        # delete `if` here
        load_path = self.ckpt_list[idx]

        merge_count = 1
        if num_ckpt == mp_world_size:
            # delete an assert here.
            sd = self.checkpoint_engine.load(
                load_path, map_location=lambda storage, loc: storage
            )
            # delete `if quantize` here
            all_scales = None
        # keep these conditoins in case that we add mp
        elif num_ckpt > mp_world_size:
            # adjust input params
            sd, all_scales, merge_count = self.merge_state_dict(
                mp_world_size, mp_rank, False, 8, 64, True
            )
        else:
            # adjust input params
            sd, all_scales = self.split_state_dict(
                mp_world_size, mp_rank, False, 8, 64, True
            )
        return load_path, sd, (all_scales, merge_count)
