""" **CoLLie** 中的异步数据提供器，为模型生成过程提供更加广泛的数据采集渠道
"""
__all__ = [
    'BaseProvider',
    'GradioProvider',
    '_GenerationStreamer'
]
import os
import shutil
import time

import pandas as pd
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
        import dash 
        from dash import Dash, html, Input, Output, State, dcc, dash_table
        from dash.long_callback import DiskcacheLongCallbackManager
        import dash_bootstrap_components as dbc
        ## Diskcache
        import diskcache
        # 文件上传
        import dash_uploader as du
        


        CACHE_PATH = "./.cache"
        if os.path.exists(CACHE_PATH):
            shutil.rmtree(CACHE_PATH)
            os.mkdir(CACHE_PATH)

        disk_cache = diskcache.Cache(os.path.join(CACHE_PATH, "cache"))
        long_callback_manager = DiskcacheLongCallbackManager(disk_cache)

        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], long_callback_manager=long_callback_manager, suppress_callback_exceptions=True)

        # 配置上传文件夹
        du.configure_upload(app, folder=os.path.join(CACHE_PATH, "TEMP"))
        # 链接的上传
        link_upload = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Input(placeholder="please enter the absolute file link...", className="mb-3", id="link_uploader"),
                    # html.P("该文件路径不存在")
                    dbc.Alert("该文件路径不存在", color="warning", is_open=False, dismissable=True, duration=2000, id="link_uploader_warn"),
                ]
            ),
        )
        # 文件上传的组件
        file_upload = dbc.Card(
            dbc.CardBody(
                [
                    du.Upload(
                        id='upload_file',
                        text='点击或拖动文件到此进行上传！',
                        text_completed='已完成上传文件：',
                        cancel_button=True,
                        pause_button=True,
                        filetypes=["txt"],
                        default_style={
                            'background-color': '#fafafa',
                            'font-weight': 'bold'
                        },
                        upload_id='myupload'
                    )
                ]
            ),
        )
        # 文件或者链接上传并且生成
        upload_app = dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Upload", style={"width": "100%", "text-align": "center"})),
                    dbc.ModalBody(
                        dbc.Tabs([
                            dbc.Tab(file_upload, label="文件上传", tab_id="tab1"),
                            dbc.Tab(link_upload, label="链接上传", tab_id="tab2"),
                        ])
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "submit", id="submit_file_link", className="ms-4", n_clicks=0, style={"margin-right": "10px"}),
                            dbc.Button(
                                "Close", id="close", n_clicks=0)
                        ],
                        style={"display": "flex", "justify-content": "center"}
                    ),
                ],
                id="modal",
                is_open=False,
                backdrop="static",
                size="lg"
            )
        # 顶栏组件
        navbar = dbc.Navbar(
                dbc.Container(
                    [
                        html.A(
                            # Use row and col to control vertical alignment of logo / brand
                            dbc.Row(
                                [
                                    dbc.Col(dbc.NavbarBrand("Collie for Generation", className="ms-0"), 
                                            ),
                                ],
                                align="center",
                            ),
                            # href="https://plotly.com",
                            style={"textDecoration": "none", },
                        ),
                    ],
                    fluid=True
                ),
                color="dark",
                dark=True,
            )
        # Text Inpt 卡片
        top_card = dbc.Card(
            [
                dbc.CardHeader(html.H4("Text Input", style={"width": "100%", "text-align": "center"})),
                dbc.CardBody(
                    dbc.Textarea(
                    size="lg", placeholder="This is input text, please edit there", id="text_input", style={"width": "100%", "height": "150px", "resize": "none"})
                ),
                dbc.CardFooter([
                    dbc.Button(
                    "Upload", id="upload-button", className="me-3", n_clicks=0, color="dark", outline=True, style={"margin-right": "20px"}, size="lg"
                ),
                    dbc.Button(
                    "Clear-all", id="clear-button", className="me-3", n_clicks=0, color="primary", outline=True, style={"margin-right": "20px"}, size="lg"
                ),dbc.Button(
                    "Submit", id="submit-button", className="me-2", n_clicks=0, color="success", outline=True, size="lg"
                )
                ],style={"display": "flex", "justify-content": "center"}),
            ],
            style={"width": "auto", "min-height": "250px","height":"100%"},
            color="secondary",
            outline=True
        )
        # Generated Output 卡片
        bottom_card = dbc.Card(
            [
                html.Div(id="hidden-div", style={"display":"none"}),
                dbc.Toast(
                    "正在生成中，请不要重复提交",
                    id="positioned-toast",
                    header="Tips",
                    is_open=False,
                    dismissable=True,
                    icon="danger",
                    duration=4000,
                    # top: 66 positions the toast below the navbar
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                ),
                dbc.Toast(
                    "生成结束",
                    id="tip-end",
                    header="Tips",
                    is_open=False,
                    dismissable=True,
                    icon="danger",
                    duration=4000,
                    # top: 66 positions the toast below the navbar
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                ),
                upload_app,
                
                # 刷新时候清除
                dcc.Store(id='memory'),
                
                dbc.CardHeader(html.H4("Generated Output", style={"width": "100%", "text-align": "center"})),
                dbc.CardBody(dbc.Textarea(
                    size="lg", placeholder="This is generated text", style={"width": "100%", "height": "150px", "resize": "none"}, id="gen_output")
                ),
                dbc.CardFooter(
                    dbc.Button(
                    "Download", id="download-button", className="me-2", n_clicks=0, outline=True, color="danger", size="lg"
                ),style={"display": "flex", "justify-content": "center"}),
                dcc.Download(id="download-dataframe-csv")
            ],
            style={"width": "auto", "min-height": "250px"},
            color="success", 
            outline=True
        )
        # 展示生成内容的卡片
        history_generation = dbc.Card(
            [
                # 不展示， 用来做中介以便更新表格参数
                dbc.Button("test_button", id="clock_assistance", n_clicks=0, style={"display":"none"}),
                
                dbc.CardHeader(html.H4("History Records", style={"width": "100%", "text-align": "center"})),
                dbc.CardBody(          
                    dash_table.DataTable(id='live-update-table',
                                        style_header={'backgroundColor':'#305D91','padding':'10px','color':'#FFFFFF'},
                                        style_table={'overflowX':'auto', 'overflowY': 'auto', 'height': "150px"},
                                        style_cell_conditional=[{'if': {'column_id': 'input_text'}, 'width': '35%'}, {'if': {'column_id': 'gen_text'}, 'width': '65%'}],
                                        style_cell={'overflow':'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'normal', 'textAlign':'center'},
                                        data=[{}],
                                        columns=[{"name": i, "id": i} for i in ['input_text', 'gen_text']])
                ),
                dbc.CardFooter(
                    dbc.Button(
                    "FreeRec", id="freerec-button", className="me-2", n_clicks=0, outline=True, color="warning", size="lg"
                ),style={"display": "flex", "justify-content": "center"}),
            ],
            style={"width": "auto", "min-height": "250px"},
            color="success", 
            outline=True
        )
        
        # 展示进度条的卡片
        progress_card = dbc.Card(
            [
                # 刷新时候清除
                dcc.Store(id='memory_file_data'),
                
                dbc.CardHeader(html.H4("Prcocess Time", style={"width": "100%", "text-align": "center"})),
                dbc.CardBody(          
                    [dbc.Progress(value=0, id="animated-progress", animated=True, striped=True, style={"margin-bottom": "20px", "margin-top": "10px"}),
                    dcc.Interval(id='timer_progress', interval=1000),
                    dash_table.DataTable(id='process-table',
                                        style_header={'backgroundColor':'#305D91','padding':'10px','color':'#FFFFFF'},
                                        style_table={'overflowX':'auto', 'overflowY': 'auto', 'height': "100px"},
                                        style_cell_conditional=[{'if': {'column_id': 'CurProcess'}, 'width': '25%'},
                                                                {'if': {'column_id': 'CurTime'}, 'width': '25%'},
                                                                {'if': {'column_id': 'TotalProcess'}, 'width': '25%'},
                                                                {'if': {'column_id': 'TotalTime'}, 'width': '25%'}],
                                        style_cell={'overflow':'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'normal', 'textAlign':'center'},
                                        data=[{'CurProcess': 0, 'CurTime': 0,  'TotalProcess':0 ,'TotalTime':0}],
                                        columns=[{"name": i, "id": i} for i in ['CurProcess', 'CurTime', 'TotalProcess' ,'TotalTime']])
                    ],
                    style={"width": "100%", "height": "180px", "resize": "none"}
                ),
                dbc.CardFooter(
                    dbc.Button(
                        "Reset", id="reset-button", className="me-2", n_clicks=0, outline=True, color="info", size="lg"
                ),style={"display": "flex", "justify-content": "center"}),
            ],
            style={"width": "auto", "min-height": "250px"},
            color="success", 
            outline=True
        )
        
        app.layout = dbc.Container([
            dbc.Row([
                navbar
            ]),
            dbc.Row([
                dbc.Col(top_card, width=6),
                dbc.Col(bottom_card, width=6),
            ], style={"margin-top": "20px", }),
            dbc.Row([
                dbc.Col(history_generation, width=6),
                dbc.Col(progress_card, width=6)
            ], style={"margin-top": "20px", })
        ], fluid=True)

        @app.long_callback(
            output=[Output("memory", 'data', allow_duplicate=True), Output("gen_output", "value", allow_duplicate=True),
                    Output("submit-button", "n_clicks", allow_duplicate=True), Output("clock_assistance", "n_clicks", allow_duplicate=True),
                    Output("tip-end", "is_open", allow_duplicate=True), Output("process-table", "data", allow_duplicate=True)], 
            inputs=[State("memory", 'data'), State("text_input", "value"), State("process-table", "data"),
                    Input("submit-button", "n_clicks")],
            running=[
                    (Output("submit-button", "disabled"), True, False),
                    (Output("positioned-toast", "is_open"), True, False)
            ],
            prevent_initial_call=True
        )
        def submit_click(session_data, text_input, process_data,n):
            session_data = session_data or []
            if n == 1 and text_input is not None:
                # 进度条文件的写入
                progress_file = os.path.join(CACHE_PATH, 'progress.txt')
                progress_pt = open(progress_file, 'w')
                precent = round(0)
                progress_pt.write(f"{precent}%\n")
                progress_pt.close()
                self.data.put(self.tokenizer(text_input, return_tensors='pt')["input_ids"])
                n = 0
                output_cache = []
                res = []
                start = time.time()
                while True:
                    feedback = self.get_feedback()
                    if feedback is not None:
                        if feedback == 'END_OF_STREAM':
                            break
                        output_cache.extend(torch.flatten(feedback).cpu().numpy().tolist())
                        res.append(self.tokenizer.decode(output_cache))
                        if not self.stream:
                            break
                        progress_pt = open(progress_file, 'w')
                        precent = precent + round((100-precent)/2)
                        progress_pt.write(f"{precent}%\n")
                        progress_pt.close()
                end = time.time()
                process_data[0]['CurTime'] = round((end - start) / 60, 2)
                process_data[0]['TotalTime'] += round((end - start) / 60, 2)
                process_data[0]['TotalProcess'] += 1
                process_data[0]['CurProcess'] = 1
                progress_pt = open(progress_file, 'w')
                precent = 100
                progress_pt.write(f"{precent}%\n")
                progress_pt.close()        
                n = 0
                session_data.append({'input_text': text_input, 'gen_text': res[-1]})
                return session_data, res[-1], n, 1, True, process_data
            else:
                n = 0
                return session_data, "", n, 1, False, process_data          

        @app.callback(
            [Output("text_input", "value"), Output("gen_output", "value"), Output("clear-button", "n_clicks")], 
            [State("text_input", "value"), State("gen_output", "value"), Input("clear-button", "n_clicks")],
            prevent_initial_call=True
        )
        def input_text(value, value1, n_clicks):
            if n_clicks > 0:
                value = ''
                value1 = ''
                n_clicks = 0
            return value, value1, n_clicks
        
        # 打开 upload 的回调函数
        @app.callback(
            Output("modal", "is_open", allow_duplicate=True),
            [Input("upload-button", "n_clicks"), Input("close", "n_clicks")],
            [State("modal", "is_open")],
            prevent_initial_call=True
        )
        def upload_open(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open
            
        
        # 文件上传的回调函数 upload_file
        @app.callback(
            Output('hidden-div', 'children'),
            Input('upload_file', 'isCompleted'),
            State('upload_file', 'fileNames')
        )
        def upload_file_fn(isCompleted, fileNames):
            if isCompleted:
                fileTemp = os.path.join(CACHE_PATH, "TEMP", "myupload")
                all_files = os.listdir(fileTemp)
                print(fileNames, all_files)
                all_files.remove(fileNames[0])
                for file_name in all_files:
                    os.remove(os.path.join(fileTemp, file_name))
                    
            return dash.no_update

        # 链接上传的回调函数 link_uploader
        @app.callback(
            [Output('link_uploader_warn', 'is_open'), Output('link_uploader', 'valid')],
            Input('link_uploader', 'value'),
            prevent_initial_call=True
        )
        def link_uploader_fn(link):
            if len(link) == 0:
                return False, False
            if os.path.isfile(link):            
                return False, True
            else:
                return True, False
        
        # 提交上传文件或者链接后关闭页面然后上传
        @app.callback(
            [Output("modal", "is_open"), Output("memory_file_data", "data")],
            [Input("submit_file_link", "n_clicks"), State("link_uploader", "value")],
            prevent_initial_call=True 
        )
        def submit_file_link(n1, value):
            if n1 > 0:
                filenames = os.listdir(os.path.join(CACHE_PATH, "TEMP", "myupload"))
                if len(filenames) == 1:
                    filepath = os.path.join(CACHE_PATH, "TEMP", "myupload", filenames[0])
                else:
                    # 先检测链接中是绝对路径，还是为相对路径
                    curPath = os.getcwd()
                    if not os.path.isabs(value):
                        value = os.path.join(curPath, value)
                    if not os.path.isfile(value):
                        return True
                    filepath = value
                
                data_list = []
                with open(filepath, "r") as fp:
                     for line in fp:
                         line = line.strip("\n").strip()
                         if len(line) > 0:
                            data_list.append(line)
                return False, data_list
            return True, []
        
        # 更新表格参数
        @app.callback([Output("live-update-table", "data"), Output("live-update-table", "columns")],
                    [Input("clock_assistance", "n_clicks"), State("memory", "data")],
                    
        )
        def update_table(n, memery_data):
            df = pd.DataFrame(memery_data)
            return df.to_dict('records'), [{"name": i, "id": i} for i in ['input_text', 'gen_text']]
        
        # 清除记录数据
        @app.callback(
            [Output("live-update-table", "data", allow_duplicate=True), Output("live-update-table", "columns", allow_duplicate=True), 
             Output("memory", "data", allow_duplicate=True), Output("freerec-button", "n_clicks")],
            Input("freerec-button", "n_clicks"),
            prevent_initial_call=True
        )
        def freeRecord(n):
            return [{}], [{"name": i, "id": i} for i in ['input_text', 'gen_text']], [], 0
        
        # 文件批量生成
        @app.long_callback(
            output=[Output("tip-end", "is_open"), Output("memory", 'data'), Output("clock_assistance", "n_clicks"),
                    Output("process-table", "data")],
            inputs=[Input("memory_file_data", "data"), State("memory", 'data'), State("process-table", "data")],
            running=[
                    (Output("submit-button", "disabled"), True, False),
                    (Output("positioned-toast", "is_open"), True, False)
            ],
            prevent_initial_call=True
        )
        def file_generate(data_list, mem_data, process_data):
            all_pair_gen = []
            progress_file = os.path.join(CACHE_PATH, 'progress.txt')
            mem_data = mem_data or []
            
            for idx, text_input in enumerate(data_list):
                self.data.put(self.tokenizer(text_input, return_tensors='pt')["input_ids"])
                output_cache = []
                res = []
                start = time.time()
                while True:
                    feedback = self.get_feedback()
                    if feedback is not None:
                        if feedback == 'END_OF_STREAM':
                            break
                        output_cache.extend(torch.flatten(feedback).cpu().numpy().tolist())
                        res.append(self.tokenizer.decode(output_cache))
                        if not self.stream:
                            break
                end = time.time()
                progress_pt = open(progress_file, 'w')
                precent = round((idx+1)/len(data_list) * 100)
                progress_pt.write(f"{precent}%\n")
                progress_pt.close()
                all_pair_gen.append({'input_text': text_input, 'gen_text': res[-1]})
                mem_data.append({'input_text': text_input, 'gen_text': res[-1]})
                
                process_data[0]['CurTime'] += round((end - start) / 60, 2)
                process_data[0]['TotalTime'] += round((end - start) / 60, 2)
                process_data[0]['TotalProcess'] += 1
            process_data[0]['CurProcess'] = len(data_list)
                
            return True, mem_data, 1, process_data
            
        # 进度条更新展示
        @app.callback(
            output=[Output('animated-progress', 'value'), Output('animated-progress', 'label')],
            inputs=Input('timer_progress', "n_intervals"),
            # progress_default=0,
            prevent_initial_call=True,
        )
        def progress_callback(n_intervals):
            try:
                with open(os.path.join(CACHE_PATH, 'progress.txt'), 'r') as file:
                    str_raw = file.read()
                last_line = list(filter(None, str_raw.split('\n')))[-1]
                percent = float(last_line.split('%')[0])
            except:
                percent = 0
            finally:
                text = f'{percent:.0f}%'
                return percent, text
        
        # 清除 process的数据
        @app.callback(
            output=[Output('animated-progress', 'value', allow_duplicate=True), Output('animated-progress', 'label', allow_duplicate=True),
                    Output("process-table", "data", allow_duplicate=True), Output("reset-button", "n_clicks")],
            inputs=[Input("reset-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_process_data(n_clicks):
            if n_clicks > 0:
                progress_file = os.path.join(CACHE_PATH, 'progress.txt')
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                return 0, "0%", [{'CurProcess': 0, 'CurTime': 0,  'TotalProcess':0 ,'TotalTime':0}], 0
        
        # 下载数据
        @app.callback(
            Output("download-dataframe-csv", "data"),
            Input("download-button", "n_clicks"), 
            State("memory", "data"),
            prevent_initial_call=True
        )
        def download_json_file(n, data):
            data = data or []
            return dcc.send_data_frame(pd.DataFrame(data).to_csv, "data.csv")
        
        app.run_server(port=self.port, host="0.0.0.0")

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