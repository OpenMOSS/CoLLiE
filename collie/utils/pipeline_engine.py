import torch
from torch import nn
import torch.distributed as dist
from deepspeed.runtime.pipe.engine import PipelineEngine, _tensor_bytes
from deepspeed.runtime.pipe import schedule, p2p, PipelineModule
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils.timer import ThroughputTimer
from deepspeed.utils import logger
from peft import PeftModel

from .utils import _split_batch, auto_param_call
from .dist_utils import broadcast_tensor, env
from ..module import PipelineModel

def is_even(number):
    return number % 2 == 0

class ColliePipelineEngine(PipelineEngine):
    def __init__(self, has_bool_tensors=False, *args, **kwargs):
        DeepSpeedEngine.__init__(self, *args, **kwargs)
        assert isinstance(self.module, PipelineModule) or (isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModule)), "model must base PipelineModule"

        assert self.zero_optimization_stage() < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors
        self.eval_return_logits = False
        self.outputs = None

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
                " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.train_batch_size(),
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'labels': [],  # labels from batch input
            'outputs': [],  # activations
            'output_tensors': [],  # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline['activation_checkpoint_interval']

        self.module.checkpoint_parallel_write_pipeline = self._config.checkpoint_parallel_write_pipeline

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        self.has_attention_mask = self.module.__class__.__name__ == 'GPT2ModelPipe'
        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()
        self.buffer_shape = None
        self.inputs_extra = {}
        self.outputs_extra = {}

    def reset_buffer_shape(self, batch):
        if self.buffer_shape is None:
            self.buffer_shape = {}
            for key, value in batch.items():
                self.buffer_shape[key] = value.shape
        else:
            flag = False
            if batch.keys() != self.buffer_shape.keys():
                flag = True
            if not flag:
                for key, value in batch.items():
                    if self.buffer_shape[key] != value.shape:
                        flag = True
            if flag:
                self.buffer_shape = {}
                for key, value in batch.items():
                    self.buffer_shape[key] = value.shape
                self.reset_activation_shape()

    def train_batch(self, batch):
        if isinstance(self.module, PipelineModel):
            self.module.inner_forward = True
            self.module.forward_type = "train"
        if isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModel):
            self.module.get_base_model().inner_forward = True
            self.module.get_base_model().forward_type = "train"
        # batch tuple, batch_size is micro_batch * accumulate_steps
        self.reset_buffer_shape(batch)
        batch = _split_batch(batch, self.train_micro_batch_size_per_gpu(),
                             self.gradient_accumulation_steps())
        data_iter = iter(batch)
        result = super().train_batch(data_iter)
        if isinstance(self.module, PipelineModel):
            self.module.inner_forward = False
        if isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModel):
            self.module.get_base_model().inner_forward = False
        return result
    
    def eval_batch(self, batch):
        if isinstance(self.module, PipelineModel):
            self.module.inner_forward = True
            self.module.forward_type = "eval"
        if isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModel):
            self.module.get_base_model().inner_forward = True
            self.module.get_base_model().forward_type = "eval"
        self.reset_buffer_shape(batch)
        if self.total_loss is not None:
            total_loss = self.total_loss.detach().clone()
        else:
            total_loss = None
        self.total_loss = None

        batch = _split_batch(batch, self.module.config.eval_batch_size,
                             self.gradient_accumulation_steps())
        data_iter = iter(batch)
        logits = super().eval_batch(data_iter, return_logits=False,
                                    compute_loss=False, reduce_output=None)
        # logits: list
        # len(logits) = micro_batch_nums
        # Assume batch first
        logits = self.broadcast_logits(logits)
        self.total_loss = total_loss
        if isinstance(self.module, PipelineModel):
            self.module.inner_forward = False
        if isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModel):
            self.module.get_base_model().inner_forward = False
        return logits
    
    def generate_batch(self, batch, use_cache=True):
        if isinstance(self.module, PipelineModel):
            self.module.inner_forward = True
            self.module.forward_type = "generate"
        if isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModel):
            self.module.get_base_model().inner_forward = True
            self.module.get_base_model().forward_type = "generate"
        self.reset_buffer_shape(batch)

        if self.total_loss is not None:
            total_loss = self.total_loss.detach().clone()
        else:
            total_loss = None
        self.total_loss = None
        # special case for generation
        if "input_ids" in batch.keys():
            gradient_accumulation_steps = batch["input_ids"].shape[0]
        elif "inputs_embeds" in batch.keys():
            gradient_accumulation_steps = batch["inputs_embeds"].shape[0]
        else:
            raise ValueError("Batch must have at least one key of `input_ids` or `input_embeds`!")
        if use_cache:
            batch = [batch]
        else:
            batch = _split_batch(batch, 1, gradient_accumulation_steps)
        data_iter = iter(batch)
        
        self._compute_loss = False
        logits = None
        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        if use_cache:
            sched = schedule.InferenceSchedule(micro_batches=1,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        else:
            sched = schedule.InferenceSchedule(micro_batches=gradient_accumulation_steps,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()
        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            logits = self._reduce_outputs(self.fwd_outputs, reduce=None)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        
        # logits: list
        # len(logits) = micro_batch_nums
        # Assume batch first
        logits = self.broadcast_logits(logits)
        self.total_loss = total_loss
        if isinstance(self.module, PipelineModel):
            self.module.inner_forward = False
        if isinstance(self.module, PeftModel) and isinstance(self.module.get_base_model(), PipelineModel):
            self.module.get_base_model().inner_forward = False
        return logits
    
    def broadcast_logits(self, logits):
        src_rank = self.grid.stage_to_global(self.num_stages - 1)
        if logits is not None:
            assert isinstance(logits, list), type(logits)
            assert isinstance(logits[0], dict), type(logits[0])
            _logits = {}
            for key in logits[0].keys():
                # torch.cat 会导致显存大幅度增长，为了防止 OOM 迁移到 cpu 上
                _logits[key] = [l[key].cpu() for l in logits]
            logits.clear()
            # 广播数目
            count_tensor = torch.LongTensor(data=[len(_logits)]).to(self.device)
            broadcast_tensor(count_tensor, dtype=torch.long, src=src_rank,
                             shape=[1], group=env.pp_group)
            # 广播 dict
            logits = {}
            for key in _logits.keys():
                logits[key] = torch.cat(_logits[key], dim=0).cuda()
                # key
                encode = list(key.encode())
                key_tensor = torch.LongTensor(data=encode).to(self.device)
                broadcast_tensor(key_tensor, dtype=torch.long, src=src_rank,
                                 ndim=1, group=env.pp_group)
                # value
                broadcast_tensor(logits[key], src=src_rank, group=env.pp_group)
            _logits.clear()
            # 考虑到速度暂时不清空
            torch.cuda.empty_cache()
        else:
            logits = {}
            count_tensor = broadcast_tensor(None, dtype=torch.long,
                                            src=src_rank, shape=[1],
                                            group=env.pp_group)
            for i in range(count_tensor.item()):
                # key
                key_tensor = broadcast_tensor(None, dtype=torch.long,
                                              src=src_rank, ndim=1,
                                              group=env.pp_group)
                key = bytes(key_tensor.tolist()).decode()
                logits[key] = broadcast_tensor(None, src=src_rank,
                                               group=env.pp_group)

        return logits

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)
        # buffer['inputs'][buffer_id]: dict
        inputs = {k: v.clone() for k, v in self.pipe_buffers['inputs'][buffer_id].items()}

        # collect the partitioned input from the previous stage
        # TODO Why partition and .full?
        if self.is_pipe_partitioned and not self.is_first_stage() and self.module.training:
            part_input = PartitionedTensor.from_meta(
                meta=self.inputs_extra["_meta"],
                local_part=self.inputs_extra["_local_data"],
                group=self.grid.get_slice_parallel_group()
            )
            inputs[self.inputs_extra["_grad_key"]] = part_input.full()
            inputs[self.inputs_extra["_grad_key"]].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            if isinstance(inputs, dict):
                self.pipe_buffers['inputs'][buffer_id] = {k:v for k, v in inputs.items()}
            else:
                self.pipe_buffers['inputs'][buffer_id] = inputs

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self._zero_grads(inputs)
        # if buffer_id >= 1:
        #     import pdb; pdb.set_trace()
        outputs = super(PipelineEngine, self).forward(inputs)

        if self.pipe_recv_buf is not None and list(outputs.keys()) != list(self.pipe_recv_buf.keys()):
            raise RuntimeError(
                "Output keys of this micro batch are not the same as the "
                "previous ones. Please check your model or data. {} vs {}"
                .format(list(outputs.keys()), list(self.pipe_recv_buf.keys()))
            )
        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.module.training:
            _grad_key = None
            for key, value in outputs.items():
                if value.requires_grad:
                    assert _grad_key is None, "More than one tensors requires grad."
                    if self.is_pipe_partitioned and not self.is_last_stage():
                        part = PartitionedTensor(tensor=value, group=self.grid.get_slice_parallel_group())
                    # Clear the large output data, but save the computation graph
                    # TODO 这里源代码没有.to，但现在必须加.to否则会无法send
                        value.data = torch.zeros(1).to(self.device)
                        self.pipe_buffers['output_tensors'][buffer_id] = value
                        self.outputs_extra["_meta"] = part.to_meta()
                        self.outputs_extra["_local_data"] = part.data()
                    _grad_key = key
            assert _grad_key is not None, "None of the outputs has grad!"
            self.outputs_extra["_grad_key"] = _grad_key
        if isinstance(outputs, dict):
            self.pipe_buffers['outputs'][buffer_id] = {k:v for k, v in outputs.items()}
        else:
            self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                # TODO 参数匹配
                labels = self.pipe_buffers['labels'][buffer_id]
                self.loss = auto_param_call(self.module.loss_fn, {**labels, **outputs}, 
                                            signature_fn=self.module.loss_fn.forward if isinstance(self.module.loss_fn, nn.Module) else self.module.loss_fn)
                # self.loss = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())

                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            elif isinstance(self.loss, dict):
                self.fwd_outputs.append({k:v.detach() for k, v in self.loss.items()})
                
                if self.total_loss is None:
                    self.total_loss = {k:torch.zeros_like(l) for k, l in self.loss.items()}
                for k, l in self.loss.items():
                    self.total_loss[k] += l.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super(PipelineEngine, self).backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                part_output = PartitionedTensor.from_meta(
                    meta=self.outputs_extra["_meta"],
                    local_part=self.outputs_extra["_local_data"],
                    group=self.grid.get_slice_parallel_group()
                )
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
                outputs[self.outputs_extra["_grad_key"]] = self.pipe_buffers['output_tensors'][buffer_id]
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs[self.outputs_extra["_grad_key"]] = self.pipe_buffers['output_tensors'][buffer_id]

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            part_grad = PartitionedTensor.from_meta(meta=self.grad_layer[0],
                                                    local_part=self.grad_layer[1],
                                                    group=self.grid.get_slice_parallel_group())
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, dict):
            out_tensors = outputs[self.outputs_extra["_grad_key"]]
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()
            
        batch = self._next_batch() # {"input_ids": torch.Tensor, "labels": torch.Tensor}
        # batch = self._next_batch() # (inputs, labels)

        # if self.is_first_stage():
        #     loaded = {}
        #     for key, tensor in batch[0].items():
        #         assert torch.is_tensor(tensor)
        #         mine = tensor.clone().detach().to(self.device)
        #         mine.requires_grad = mine.is_floating_point()
        #         loaded[key] = mine

        #     self.pipe_buffers["inputs"][buffer_id] = loaded

        # if self.is_last_stage():
        #     loaded = batch[1]
        #     # tensor or dict
        #     if torch.is_tensor(batch[1]):
        #         loaded = batch[1].to(self.device)
        #     elif isinstance(batch[1], dict):
        #         loaded = {}
        #         for key, tensor in batch[1].items():
        #             assert torch.is_tensor(tensor)
        #             tensor = tensor.to(self.device).detach()
        #             loaded[key] = tensor
        #     else:
        #         raise NotImplementedError
        #     self.pipe_buffers['labels'][buffer_id] = loaded
        
        if self.is_first_stage() or self.is_last_stage():
            loaded = {}
            for key, tensor in batch.items():
                assert torch.is_tensor(tensor)
                mine = tensor.clone().detach().to(self.device)
                mine.requires_grad = mine.is_floating_point()
                loaded[key] = mine
            if self.is_first_stage():
                self.pipe_buffers["inputs"][buffer_id] = loaded
            if self.is_last_stage():
                self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, dict):
            type_tensor = torch.LongTensor(data=[3]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for key, tensor in buffer.items():
                self._send_string(key, recv_stage)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(self.device)
                p2p.send(send_dtype, recv_stage)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        else:
            super()._send_tensor_meta(buffer, recv_stage)

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()
        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes_and_dtypes = []
            for idx in range(num_tensors):
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))

            buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers
        
        elif recv_type == 3:
            # dict
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_dict = {}
            for idx in range(num_tensors):
                recv_key = self._recv_string(send_stage)
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shape = recv_shape.tolist()
                buffer = self._allocate_buffer(recv_shape, dtype=recv_dtype, num_buffers=1)[0]
                recv_dict[recv_key] = buffer
            return recv_dict

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        # TODO 如何处理
        # if self.has_attention_mask or self.has_bool_tensors:
        #     outputs = list(outputs)
        #     outputs[-1] = outputs[-1].half()
        #     outputs = tuple(outputs)

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)
            if self.module.training:
                self._send_string(self.outputs_extra["_grad_key"],
                                  self.next_stage)
                # extra
                if self.is_pipe_partitioned:
                    self._send_tensor_meta(self.outputs_extra["_meta"],
                                           self.next_stage)
                    self._send_tensor_meta(self.outputs_extra["_local_data"],
                                           self.next_stage)
        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, dict):
            # extra
            if self.is_pipe_partitioned and self.module.training:
                p2p.send(self.outputs_extra["_meta"], self.next_stage)
                p2p.send(self.outputs_extra["_local_data"], self.next_stage)
            for key, tensor in outputs.items():
                # TODO 是否有必要？
                self._send_string(key, self.next_stage)
                p2p.send(tensor, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        # Restore the boolean tensor
        # TODO 如何处理
        # if self.has_attention_mask or self.has_bool_tensors:
        #     outputs = list(outputs)
        #     outputs[-1] = outputs[-1].bool()
        #     outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()
        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        
        if self.is_grad_partitioned:
            part = None
            _grad_key = None
            for key, value in inputs.items():
                if value.grad is not None:
                    # TODO 在 collie 里我们暂时限定 dict 里只有一个 tensor 是有梯度的
                    assert part is None, "More than one tensors have grad."
                    part = PartitionedTensor(tensor=value.grad, group=self.grid.get_slice_parallel_group())
                    _grad_key = key

            assert part is not None
            # TODO key 是否需要？
            p2p.send(part.to_meta(), self.prev_stage)
            p2p.send(part.data(), self.prev_stage)
        else:
            for key, tensor in inputs.items():
                # Skip tensors that will not produce a grad
                if not tensor.is_floating_point():
                    assert tensor.grad is None
                    continue
                if tensor.grad is None:
                    continue
                p2p.send(tensor.grad, self.prev_stage)

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        # if self.has_attention_mask or self.has_bool_tensors:
        #     bool_keys = []
        #     for key, tensor in inputs.items():
        #         if tensor.dtype == torch.bool:
        #             bool_keys.append(key)
        #     for key in bool_keys:
        #         inputs.pop(key)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recvd = None
        # print(f"Print self.pipe_recv_buf: {self.pipe_recv_buf} Prev stage: {self.prev_stage}")
        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
            if self.module.training:
                self.inputs_extra["_grad_key"] = self._recv_string(self.prev_stage)
                if self.is_pipe_partitioned:
                    self.inputs_extra["_meta"] = self._recv_tensor_meta(self.prev_stage)
                    self.inputs_extra["_local_data"] = self._recv_tensor_meta(self.prev_stage)

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, dict)
            # extra
            recvd = {}
            if self.is_pipe_partitioned and self.module.training:
                if self.meta_buffer is None:
                    self.meta_buffer = torch.zeros(self.inputs_extra["_meta"].size(), dtype=torch.long, device=self.device)
                self.inputs_extra["_meta"] = self.meta_buffer
                p2p.recv(self.inputs_extra["_meta"], self.prev_stage)
                p2p.recv(self.inputs_extra["_local_data"], self.prev_stage)
            for key, tensor in self.pipe_recv_buf.items():
                assert torch.is_tensor(tensor)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and key == "_meta" and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(tensor.size(), dtype=torch.long, device=self.device)
                    buffer = self.meta_buffer
                # TODO 是否有必要？
                recv_key = self._recv_string(self.prev_stage)
                assert key == recv_key, f"{key}, {recv_key}"
                p2p.recv(tensor, self.prev_stage)
                recvd[key] = tensor.clone().detach()

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            # TODO 如何处理
            # if self.has_attention_mask or self.has_bool_tensors:
            #     recvd[-1] = recvd[-1].bool()
            for key, buffer in recvd.items():
                    buffer.requires_grad = self.module.training and \
                        (key == self.inputs_extra["_grad_key"])
        if isinstance(recvd, dict):
            self.pipe_buffers['inputs'][buffer_id] = {k:v for k, v in recvd.items()}
        else:
            self.pipe_buffers['inputs'][buffer_id] = recvd

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            # TODO 这个 if 无法进入，暂时搁置
            part_output = PartitionedTensor.from_meta(meta=outputs[0],
                                                      local_part=outputs[1],
                                                      group=self.grid.get_slice_parallel_group())
            outputs[0].data = part_output.full()
            outputs = (outputs[0], *outputs[2:])
            # save for backward
            if isinstance(outputs, dict):
                self.pipe_buffers['outputs'][buffer_id] = {k:v for k, v in outputs.items()}
            else:
                self.pipe_buffers['outputs'][buffer_id] = outputs

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if self.is_grad_partitioned:
                sizes_and_dtypes = []
                # meta & data
                sizes_and_dtypes.append((list(self.outputs_extra["_meta"].size()), self.outputs_extra["_meta"].dtype))
                # TODO send 的时候只 send 了一个张量的 meta 和 grad，是否有必要 extend
                sizes_and_dtypes.append((list(self.outputs_extra["_local_data"].size()), self.outputs_extra["_local_data"].dtype))
                # sizes_and_dtypes.extend(
                #     [(list(t.size()), t.dtype) for k, t in outputs.items()
                #      if t.is_floating_point() ]
                # )
            else:
                sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs.values() if t.is_floating_point() and t.requires_grad]
            self.grad_layer = self._allocate_buffers(sizes_and_dtypes, num_buffers=1)[0]

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(outputs, dict)
            for idx, buffer in enumerate(self.grad_layer):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                p2p.recv(buffer, self.next_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        return super()._exec_optimizer_step(lr_kwargs)

    def _exec_reduce_grads(self):
        return super()._exec_reduce_grads()
    
    def _exec_reduce_tied_grads(self):
        return super()._exec_reduce_tied_grads()

    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _send_string(self, string, recv_stage):
        encode = list(string.encode())
        len_tensor = torch.LongTensor(data=[len(encode)]).to(self.device)
        p2p.send(len_tensor, recv_stage)
        str_tensor = torch.LongTensor(data=encode).to(self.device)
        p2p.send(str_tensor, recv_stage)

    def _recv_string(self, send_stage):
        len_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(len_tensor, send_stage)
        str_tensor = torch.LongTensor([1] * len_tensor.item()).to(self.device)
        p2p.recv(str_tensor, send_stage)
        recv_str = bytes(str_tensor.tolist()).decode()
        return recv_str

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for k, t in inputs.items():
                if t.grad is not None:
                    t.grad.data.zero_()

    def _scale_loss_by_gas(self, prescaled_loss):
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / self.gradient_accumulation_steps()
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / self.gradient_accumulation_steps())
                else:
                    scaled_loss.append(l)
        elif isinstance(scaled_loss, dict):
            scaled_loss = {}
            for k, l in prescaled_loss.items():
                if isinstance(l, torch.Tensor):
                    scaled_loss[k] = l / self.gradient_accumulation_steps()
                else:
                    scaled_loss[k] = l
        else:
            scaled_loss = prescaled_loss
            if self.warn_unscaled_loss:
                logger.warning(f"DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}")
                self.warn_unscaled_loss = False

        return scaled_loss
    
    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            elif isinstance(outputs[0], dict):
                # list of dict 
                reduced = {k:torch.zeros_like[o] for k, o in outputs[0].items()}
                for idx, out in enumerate(outputs):
                    for key, o in out.items():
                        reduced[key] += o
            else:
                # TODO 这里是源代码。是不是不太对？
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                elif isinstance(reduced, dict):
                    for k, in reduced.keys():
                        dist.all_reduce(reduced[k], group=self.mpu.get_data_parallel_group())
                        reduced[k] /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx], group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')
