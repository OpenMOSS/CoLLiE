""" **CoLLie** 中的异步数据提供器，为模型生成过程提供更加广泛的数据采集渠道
"""
__all__ = [
    'BaseProvider',
    'GradioProvider',
    '_GenerationStreamer'
]
import time

from torch.multiprocessing import Process, Queue
from transformers.generation.streamers import BaseStreamer
from transformers import PreTrainedTokenizer
import torch

class BaseProvider:
    """ BaseProvider 为异步数据提供器的基类，提供了一些基本的接口
    """
    def __init__(self, 
                 stream: bool = False) -> None:
        self.data = Queue()
        self.feedback = Queue()
        self.stream = stream
        self.provider_started = False
        
    def provider_handler(self):
        """ provider_handler 为异步数据提供器的主要逻辑，需要被子类重写，主要功能为异步地收集数据并放入队列 `self.data` 中
        """
        while True:
            self.data.put('Hello World')
            time.sleep(1)
        
    def start_provider(self):
        """ start_provider 为异步数据提供器的启动函数，会在一个新的进程中启动 `provider_handler` 函数
        """
        process = Process(target=self.provider_handler)
        process.start()
        self.provider_started = True
        
    def get_data(self):
        """ get_data 为异步数据提供器的数据获取函数，会从队列 `self.data` 中获取数据
        """
        if self.data.empty():
            return None
        else:
            return self.data.get()
        
    def get_feedback(self):
        """ get_feedback 为异步数据提供器的反馈获取函数，会从队列 `self.feedback` 中获取反馈，主要指模型生成的结果
        """
        if self.feedback.empty():
            return None
        else:
            return self.feedback.get()
        
    def put_feedback(self, feedback):
        """ put_feedback 为异步数据提供器的反馈放入函数，会将反馈放入队列 `self.feedback` 中，该函数由 **CoLLie** 自动调用，将模型生成的结果放入该队列中
        """
        self.feedback.put(feedback)
        
class GradioProvider(BaseProvider):
    """ 基于 Gradio 的异步数据提供器，会在本地启动一个 Gradio 服务，将用户输入的文本作为模型的输入
    """ 
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 port: int = 7878,
                 stream: bool = False) -> None:
        super().__init__(stream)
        self.tokenizer = tokenizer
        self.port = port
        
    def provider_handler(self):
        import gradio as gr
        output_cache = []
        def submit(text):
            output_cache.clear()
            self.data.put(self.tokenizer(text, return_tensors='pt')["input_ids"])
            while True:
                feedback = self.get_feedback()
                if feedback is not None:
                    if feedback == 'END_OF_STREAM':
                        break
                    output_cache.extend(torch.flatten(feedback).cpu().numpy().tolist())
                    yield self.tokenizer.decode(output_cache)
                    if not self.stream:
                        break
        interface = gr.Interface(fn=submit, inputs="textbox", outputs="text")
        interface.queue()
        interface.launch(server_name="0.0.0.0", server_port=self.port, share=True)

class DashProvider(BaseProvider):
    """ 基于 Dash 的异步数据提供器，会在本地启动一个 Dash 服务，将用户输入的文本作为模型的输入
    """ 
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 port: int = 7878,
                 stream: bool = False) -> None:
        super().__init__(stream)
        self.tokenizer = tokenizer
        self.port = port
    
    def provider_handler(self):
        # from collie.utils.dash_html import app
        from dash import Dash, html, Input, Output, dcc

        app = Dash(__name__)
        app.layout = html.Div([
            html.H6("更改文本框中的值以查看回调操作！"),
            html.Div(["输入：",
                    dcc.Input(id='my-input', value='初始值', type='text')]),
            html.Br(),
            html.Div(id='my-output'),
        ])

        app.run_server(port=self.port, host="0.0.0.0", debug=True)

class _GenerationStreamer(BaseStreamer):
    """ 重写 `transformers` 的 `BaseStreamer` 类以兼容 **CoLLie** 的异步数据提供器
    """
    def __init__(self, server: BaseProvider) -> None:
        self.server = server
        self.stop_signal = 'END_OF_STREAM'
        
    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("_GenerationStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        self.server.put_feedback(value)
        
    def end(self):
        self.server.put_feedback(self.stop_signal)