import os
import asyncio
import multiprocessing as mp
from websockets.server import serve
from websockets.sync.client import connect

import time

from torch.multiprocessing import Process, Queue, set_start_method
from transformers.generation.utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers import PreTrainedTokenizer
from pydantic import BaseModel
from fastapi import FastAPI
import torch
import queue

class BaseServer:
    def __init__(self, 
                 stream: bool = False) -> None:
        self.data = Queue()
        self.feedback = Queue()
        self.stream = stream
        
    def provider_handler(self):
        while True:
            self.data.put('Hello World')
            time.sleep(1)
        
    def start_provider(self):
        process = Process(target=self.provider_handler)
        process.start()
        
    def get_data(self):
        if self.data.empty():
            return None
        else:
            return self.data.get()
        
    def get_feedback(self):
        if self.feedback.empty():
            return None
        else:
            return self.feedback.get()
        
    def put_feedback(self, feedback):
        self.feedback.put(feedback)
        
class GradioServer(BaseServer):
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
        interface.launch(server_name="0.0.0.0", server_port=self.port)
        
class GenerationStreamer(BaseStreamer):
    def __init__(self, server: BaseServer) -> None:
        self.server = server
        self.stop_signal = 'END_OF_STREAM'
        
    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("GenerationStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        self.server.put_feedback(value)
        
    def end(self):
        self.server.put_feedback(self.stop_signal)