import argparse

from transformers import AutoModelForCausalLM

from peft import PeftModel


def main(args):
    merged_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.base, trust_remote_code=True, device_map="auto"
        ),
        args.adapter,
    ).merge_and_unload()
    merged_model.save_pretrained(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", "-m", type=str)
    parser.add_argument("--adapter", "-a", type=str)
    parser.add_argument("--save_path", "-d", type=str)
    args = parser.parse_args()
    main(args)

# python -u merge_weights.py -m internlm/internlm-7b -a ./lora/default/last -d ./lora/default/merged
