from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from megatron.core import parallel_state
import copy

def hack_deepspeed():
    raw_init = copy.deepcopy(DeepSpeedZeroOptimizer.__init__)
    def safe_init(self):
        while True:
            try:
                raw_init(self)
                break
            except RuntimeError as e:
                continue
    DeepSpeedZeroOptimizer.__init__ = safe_init
    raw_initialize_optimizer_states = copy.deepcopy(DeepSpeedZeroOptimizer.initialize_optimizer_states)
    def safe_initialize_optimizer_states(self):
            while True:
                try:
                    raw_initialize_optimizer_states(self)
                    break
                except RuntimeError as e:
                    continue
    DeepSpeedZeroOptimizer.initialize_optimizer_states = safe_initialize_optimizer_states
    
def hack_megatron():
    parallel_state.get_model_parallel_world_size = lambda: parallel_state.get_tensor_model_parallel_world_size()
    parallel_state.get_model_parallel_rank = lambda: parallel_state.get_tensor_model_parallel_rank()