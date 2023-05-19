import os
import time
from multiprocessing import Process
import threading
from typing import Optional
import socket
from fastapi import FastAPI
import uvicorn
import queue
from transformers import PreTrainedTokenizer

class HttpTokenizerServer:
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 master_addr: Optional[str]=None, 
                 master_port: Optional[int]=None,
                 max_retry: int = 5) -> None:
        if master_addr is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
        if master_port is None:
            master_port = int(os.environ.get("MASTER_PORT", "29500")) + 1
        self.master_addr = master_addr
        self.master_port = master_port
        self.max_retry = max_retry
        self.tokenizer = tokenizer
        self.queue = queue.Queue()
        
    @staticmethod
    def worker(master_addr, master_port, max_retry=5):
        app = FastAPI()
        @app.post("/")
        async def post_data(buffer: str):
            retry_count = max_retry
            while retry_count > 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as worker_socket:
                    try:
                        worker_socket.connect((master_addr, master_port))
                        worker_socket.send(buffer.encode())
                        break
                    except socket.error as e:
                        retry_count -= 1
                        print(f"Retrying connection. Attempt number: {max_retry - retry_count}")
                        time.sleep(1)
                        continue
            return {"message": "Success"}
        uvicorn.run(app, host=master_addr, port=master_port + 1)
            
    def start_worker(self):
        self.worker_process = Process(target=self.worker, args=(self.master_addr, self.master_port, self.max_retry))
        self.worker_process.start()
    
    @staticmethod
    def listener(master_addr, master_port, queue):
        listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener_socket.bind((master_addr, master_port))
        listener_socket.listen(1)
        while True:
            conn, addr = listener_socket.accept()
            buffer = []
            while True:
                data = conn.recv(1024).decode()
                if not data:
                    break
                buffer.append(data)
            conn.close()
            queue.put("".join(buffer))

    def start_listener(self):
        self.listener_thread = threading.Thread(target=self.listener, args=(self.master_addr, self.master_port, self.queue))
        self.listener_thread.start()
    
    def get(self):
        item = None
        try:
            item = self.queue.get(block=False)
            item = f"{item}"
            return self.tokenizer(item, return_tensors="pt").unsqueeze(0).cuda()
        except queue.Empty:
            pass
        return item
    