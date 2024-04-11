from typing import Callable


def prepare_chatml_messages(messages, special_tokens_map, add_generation_prompt=False):
    """
    ChatML Template

    {{ bos_token }}
    {% for message in messages %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {% endfor %}
    {% if add_generation_prompt %}
        {{ '<|im_start|>assistant\n' }}
    {% endif %}
    """
    prepared_messages = []
    prepared_messages += [{"content": special_tokens_map['bos_token'], "require_loss": False}]
    for message in messages['history']:
        if message['role'] == "assitant":
            prepared_messages += [{"content": '<|im_start|>' + message['role'] + '\n', "require_loss": False}]
            prepared_messages += [{"content": message['content'], "require_loss": True}]
            prepared_messages += [{"content": '<|im_end|>' + '\n', "require_loss": False}]
        else:
            prepared_messages += [
                {"content": f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n", "require_loss": False}]
    if add_generation_prompt:
        prepared_messages += [{"content": '<|im_start|>assistant\n', "require_loss": False}]
    return prepared_messages


def prepare_moss_messages(messages, special_tokens_map, add_generation_prompt=False):
    pass


TOKENIZER_PREPARE_TEMPLATE_FN_MAPPING = {
    "Qwen2Tokenizer": prepare_chatml_messages,
    "Qwen2TokenizerFast": prepare_chatml_messages,
}


def tokenize_conversation(conversation, tokenizer, prepare_template_fn: Callable or None = None,
                          add_generation_prompt=False):
    if prepare_template_fn is None:
        if type(tokenizer).__name__ not in TOKENIZER_PREPARE_TEMPLATE_FN_MAPPING:
            raise ValueError(f"Tokenizer {type(tokenizer).__name__} has no preset template. Please provide one.")
        else:
            prepare_template_fn = TOKENIZER_PREPARE_TEMPLATE_FN_MAPPING[type(tokenizer).__name__]
    prepared_messages = prepare_template_fn(messages=conversation, special_tokens_map=tokenizer.special_tokens_map,
                                            add_generation_prompt=add_generation_prompt)

    input_ids = []
    labels = []
    attention_mask = []
    for m in prepared_messages:
        # add_special_token=False
        pass

    return input_ids, labels, attention_mask
