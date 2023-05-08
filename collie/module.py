from megatron.core import tensor_parallel

class ColumnParallelLinearWithoutBias(tensor_parallel.ColumnParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]
    
class RowParallelLinearWithoutBias(tensor_parallel.RowParallelLinear):
    def forward(self, input_):
        return super().forward(input_)[0]