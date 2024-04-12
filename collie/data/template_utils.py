import json
from typing import Callable


def prepare_chatml_messages(messages, special_tokens_map, text_field, add_generation_prompt=False):
    """
    准备ChatML格式的多轮对话，其格式如下：

        {{ bos_token }}
        {% for message in messages %}
            {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
        {% endfor %}
        {% if add_generation_prompt %}
            {{ '<|im_start|>assistant\n' }}
        {% endif %}

    :param messages: 包含对话消息的字典，每个消息包含角色和内容字段。
    :param special_tokens_map: 特殊token映射，包含'bos_token'等特殊token。
    :param text_field: 对话数据中文本字段的名称。
    :param add_generation_prompt: 是否添加用于生成的提示，默认为False。

    :return: 返回一个列表，其中每个元素是一个字典，包含'content'和'require_loss'两个字段。'content'字段表示消息内容，
             'require_loss'字段表示是否需要计算该消息的损失。
    """
    prepared_messages = []
    prepared_messages += [{"content": special_tokens_map['bos_token'], "require_loss": False}]
    for message in messages[text_field]:
        if message['role'] == "assistant":
            prepared_messages += [{"content": '<|im_start|>assistant\n', "require_loss": False}]
            prepared_messages += [{"content": message['content'] + '<|im_end|>', "require_loss": True}]
            prepared_messages += [{"content": '\n', "require_loss": False}]
        else:
            prepared_messages += [
                {"content": f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n", "require_loss": False}
            ]
    if add_generation_prompt:
        prepared_messages += [{"content": '<|im_start|>assistant\n', "require_loss": False}]
    return prepared_messages


def prepare_moss_messages(messages, special_tokens_map, text_field, add_generation_prompt=False):
    """
    准备MOSS格式的多轮对话，其格式如下：

        {{ bos_token }}
        {% for message in messages %}


    """
    prepared_messages = []
    prepared_messages += [{"content": special_tokens_map['bos_token'], "require_loss": False}]
    end_token_dict = {
        "user": "<|end_of_user|>",
        "system": "<|end_of_system|>",
        "assistant": "<|end_of_assistant|>",
        "func_call": "<|end_of_func_call|>",
        "func_ret": "<|end_of_func_ret|>",
        "thought": "<|end_of_thought|>",  # 暂时没用
        "image": "<|end_of_image|>",  # 暂时没用
        "audio": "<|end_of_audio|>",  # 暂时没用
        "video": "<|end_of_video|>",  # 暂时没用
        "moss": "<|end_of_moss|>",  # 每轮对话的最终回复
    }
    first_user_message = True  # 用于决定是否添加"<|end_of_moss|>"
    first_response_this_turn = True  # 用于决定是否将"<|im_start|>"设置为需要学习

    for message in messages[text_field]:
        role = message['role']
        if role == "user":
            message_content = f"<|im_start|>user\n{message['content']}{end_token_dict['user']}\n"
            if first_user_message:
                prepared_messages.append({"content": message_content, "require_loss": False})
                first_user_message = False
            else:
                # 上一轮 moss 完成了回复，预测的最后一个 token 是 <|end_of_moss|>
                prepared_messages.append({"content": end_token_dict['moss'], "require_loss": True})
                prepared_messages.append({"content": f"\n{message_content}", "require_loss": False})
                first_response_this_turn = True
        elif role == "assistant":
            if "content" in message:
                if first_response_this_turn:
                    prepared_messages.append({"content": '<|im_start|>', "require_loss": False})
                    first_response_this_turn = False
                else:
                    prepared_messages.append({"content": '<|im_start|>', "require_loss": True})
                prepared_messages.append(
                    {"content": f"assistant\n{message['content']}{end_token_dict['assistant']}", "require_loss": True})
                prepared_messages.append({"content": '\n', "require_loss": False})
            elif "func_call" in message:
                func_call_content = json.dumps(message["func_call"])
                if first_response_this_turn:
                    prepared_messages.append({"content": '<|im_start|>', "require_loss": False})
                    prepared_messages.append(
                        {"content": f"func_call\n{func_call_content}{end_token_dict['func_call']}\n",
                         "require_loss": True})
                    first_response_this_turn = False
                else:
                    prepared_messages.append(
                        {
                            "content": f"<|im_start|>func_call\n{func_call_content}{end_token_dict['func_call']}",
                            "require_loss": True
                        }
                    )
                    prepared_messages.append({"content": '\n', "require_loss": False})
            elif "func_ret" in message:
                func_ret_content = json.dumps(message["func_ret"])
                prepared_messages.append(
                    {
                        "content": f"<|im_start|>func_ret\n{func_ret_content}{end_token_dict['func_ret']}\n",
                        "require_loss": False
                    }
                )
            else:
                raise ValueError(f"Unknown assistant message: {message}")
        else:
            prepared_messages.append(
                {"content": f"<|im_start|>{role}\n{message['content']}{end_token_dict[role]}\n", "require_loss": False}
            )

    prepared_messages.append({"content": end_token_dict['moss'] + "\n", "require_loss": True})
    if add_generation_prompt:
        prepared_messages += [{"content": '<|im_start|>', "require_loss": False}]
    return prepared_messages


TOKENIZER_PREPARE_TEMPLATE_FN_MAPPING = {
    "Qwen2Tokenizer": prepare_chatml_messages,
    "Qwen2TokenizerFast": prepare_chatml_messages,
}


def tokenize_conversation(conversation, tokenizer, text_field='history', prepare_template_fn: Callable or None = None,
                          add_generation_prompt=False):
    if prepare_template_fn is None:
        if type(tokenizer).__name__ not in TOKENIZER_PREPARE_TEMPLATE_FN_MAPPING:
            raise ValueError(f"Tokenizer {type(tokenizer).__name__} has no preset template. Please provide one.")
        else:
            prepare_template_fn = TOKENIZER_PREPARE_TEMPLATE_FN_MAPPING[type(tokenizer).__name__]
    prepared_messages = prepare_template_fn(
        messages=conversation,
        special_tokens_map=tokenizer.special_tokens_map,
        text_field=text_field,
        add_generation_prompt=add_generation_prompt
    )

    input_ids = []
    labels = []
    attention_mask = []
    for m in prepared_messages:
        output = tokenizer(m['content'], add_special_tokens=False)
        if m['require_loss']:
            label = output['input_ids']
        else:
            label = [-100] * len(output['input_ids'])
        input_ids += output['input_ids']
        labels += label
        attention_mask += output['attention_mask']

    return input_ids, labels, attention_mask


if __name__ == "__main__":
    T = {
        "chat": [
            {
                "role": "system",
                "content": "I have nothing to tell you, just be yourself."
            }, {
                "role": "user",
                "content": "Give me the current weather (use Celcius) for Toronto, Canada."
            },
            {
                "role": "assistant",
                # "hidden": True,
                "content": "I need to use the function Weather.get_weather to get the current weather for Toronto, Canada."
            },
            {
                "role": "assistant",
                # "hidden": True,
                "func_call": {
                    "name": "Weather.get_weather",
                    "parameters": {
                        "city": "Toronto, Canada"
                    }
                }
            },
            {
                "role": "assistant",
                # "hidden": True,
                "func_ret": {"min_temperature": 22, "max_temperature": 25, "weather": "Sunny"}
            },
            {
                "role": "assistant",
                "content": "The current weather in Toronto is 22C."
            }, {
                "role": "user",
                "content": "Draw a picture of Toronto."
            },
            {
                "role": "assistant",
                "content": "Ok, I will draw a picture for you."
            },
            {
                "role": "assistant",
                "func_call": {
                    "name": "text2img",
                    "parameters": {
                        "query": "Draw a picture of Toronto"
                    }
                }
            },
            {
                "role": "assistant",
                "func_ret": {
                    "image": "xxxxx"
                }
            },
            {
                "role": "assistant",
                "content": "这幅图展示了xxxx"
            }
        ]
    }
    T['history'] = [t for t in T['chat'] if "content" in t]


    def _colored_string(string: str, color: str or int) -> str:
        """在终端中显示一串有颜色的文字

        :param string: 在终端中显示的文字
        :param color: 文字的颜色
        :return:
        """
        if isinstance(color, str):
            color = {
                "black": 30,
                "red": 31,
                "green": 32,
                "yellow": 33,
                "blue": 34,
                "purple": 35,
                "cyan": 36,
                "white": 37
            }[color]
        return "\033[%dm%s\033[0m" % (color, string)


    result = prepare_moss_messages(T, special_tokens_map={'bos_token': '<BOS>'}, text_field='chat')

    print("\n\n====moss2====\n")
    for t in result:
        print(_colored_string(t['content'], 'green' if t["require_loss"] else 'red'), end="")

    result = prepare_chatml_messages(T, special_tokens_map={'bos_token': '<BOS>'}, text_field='history')

    print("\n\n====chatml====\n")
    for t in result:
        print(_colored_string(t['content'], 'green' if t["require_loss"] else 'red'), end="")

    T['history'] = [
        {
            "role": "user",
            "content": "Give me the current weather (use Celcius) for Toronto, Canada."
        },
        {
            "role": "assistant",
            "func_call": {
                "name": "Weather.get_weather",
                "parameters": {
                    "city": "Toronto, Canada"
                }
            }
        },
        {
            "role": "assistant",
            "func_ret": {"min_temperature": 22, "max_temperature": 25, "weather": "Sunny"}
        },
        {
            "role": "assistant",
            "content": "Ok, I will draw a picture for you."
        },
        {
            "role": "user",
            "content": "Draw a picture of Toronto."
        },
        {
            "role": "assistant",
            "content": "Ok, I will draw a picture for you."
        },
        {
            "role": "assistant",
            "func_call": {
                "name": "text2img",
                "parameters": {
                    "query": "Draw a picture of Toronto"
                }
            }
        },
        {
            "role": "assistant",
            "func_ret": {
                "image": "xxxxx"
            }
        },
        # {
        #     "role": "assistant",
        #     "content": "这幅图展示了xxxx"
        # }
    ]

    result = prepare_moss_messages(T, special_tokens_map={'bos_token': '<BOS>'}, text_field='history')

    print("\n\n====moss2====\n")
    for t in result:
        print(_colored_string(t['content'], 'green' if t["require_loss"] else 'red'), end="")
