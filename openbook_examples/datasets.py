import copy
from tqdm import tqdm

from datasets import load_dataset
from dataclasses import dataclass
from fastNLP import print
from torch.utils.data import Dataset
from transformers.trainer import *

from prompts import QuestionPart, Exemplar

IGNORE_INDEX = -100


class MyDataset(Dataset):
    def __init__(self, data_args, tokenizer, dataset_info, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            dataset = load_dataset(dataset_info.path, name=dataset_info.name, split=split)
            self.data = self.process(dataset_info.extractor, dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)
        print('Data format:', self.data[0])

    def process(self, extractor, dataset, save_file):
        data = []
        for instance in tqdm(dataset):
            exemplar = Exemplar(**extractor(instance))
            if self.data_args.prompt_type == 'natural':
                prompt = exemplar.get_natural_prompt()
            else:
                prompt = exemplar.get_brown_prompt()
            source = prompt['source']
            targets = []

            def _tokenize_fn(source, target):
                targets.append(target)
                example = f"{source}{target}"
                example_tokenized = self.tokenizer.tokenizer.encode(example, bos=True, eos=True)
                source_tokenized = self.tokenizer.tokenizer.encode(source, bos=True, eos=False)

                input_ids = example_tokenized
                labels = copy.deepcopy(input_ids)
                if not self.data_args.train_on_inputs:
                    labels = np.array(labels)
                    labels[:len(source_tokenized) - 1] = IGNORE_INDEX
                return input_ids, labels

            if self.split == 'train':
                input_ids, labels = _tokenize_fn(source, prompt['target'])
            else:
                input_ids = []
                labels = []
                for choice in prompt['choices']:
                    op_input_ids, op_labels = _tokenize_fn(source, choice)
                    input_ids.append(op_input_ids)
                    labels.append(op_labels)

            data.append({'input_ids': input_ids,
                         'labels': labels,
                         'source': source,
                         'target': targets,
                         'answer': exemplar.answer_idx})

        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }


def get_train_sampler(train_dataset, args) -> Optional[torch.utils.data.Sampler]:
    if train_dataset is None or not has_length(train_dataset):
        return None

    generator = torch.Generator(device='cuda')
    # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
    # `args.seed`) if data_seed isn't provided.
    # Further on in this method, we default to `args.seed` instead.
    seed = args.data_seed if args.data_seed is not None else args.seed
    generator.manual_seed(seed)

    if args.group_by_length:
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            lengths = (
                train_dataset[args.length_column_name]
                if args.length_column_name in train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = "input_ids"
        return LengthGroupedSampler(
            args.train_batch_size * args.gradient_accumulation_steps,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
            generator=generator,
        )
    else:
        return RandomSampler(train_dataset, generator=generator)


def get_train_dataloader(train_dataset, data_collator, args) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].
    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.
    Subclass and override this method if you want to inject some custom behavior.
    """
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = train_dataset
    data_collator = data_collator

    train_sampler = get_train_sampler(train_dataset, args)

    return DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=args.dataloader_drop_last,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
    )


def get_eval_sampler(eval_dataset: Dataset, args) -> Optional[torch.utils.data.Sampler]:
    return SequentialSampler(eval_dataset)


def get_eval_dataloader(eval_dataset: Optional[Dataset], data_collator, args) -> DataLoader:
    """
    Returns the evaluation [`~torch.utils.data.DataLoader`].
    Subclass and override this method if you want to inject some custom behavior.
    Args:
        eval_dataset (`torch.utils.data.Dataset`, *optional*):
            If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
            by the `model.forward()` method are automatically removed. It must implement `__len__`.
    """
    if eval_dataset is None and eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    eval_dataset = eval_dataset if eval_dataset is not None else eval_dataset
    data_collator = data_collator

    eval_sampler = get_eval_sampler(eval_dataset, args)

    return DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        drop_last=args.dataloader_drop_last,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )


@dataclass
class DatasetInfo:
    path: str = None
    exemplar_split: str = None
    eval_split: str = None
    test_split: str = None
    extractor: Callable = None
    name: str = None
    data_dir: str = None


def get_dataset_info(dataset_name):
    if dataset_name == 'hellaswag':
        return DatasetInfo(
            path="hellaswag",
            exemplar_split="train",
            eval_split="validation",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        (
                            f"({row['activity_label']}) " if
                            row["source_id"].startswith("activity")
                            else ""
                        ) + row["ctx_a"],
                        tag="Passage"
                    ),
                    QuestionPart(
                        "Which choice best continues the passage?",
                        tag="Question"
                    )
                ],
                "choices": [
                    f"{row['ctx_b']}{' ' if len(row['ctx_b']) else ''}{e}"
                    for e in row["endings"]
                ],
                "answer_idx": int(row["label"]) if len(row["label"]) else None
            }
        )
    elif dataset_name == 'openbookqa':
        return DatasetInfo(
            path="openbookqa",
            name="main",
            exemplar_split="train",
            eval_split="validation",
            test_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question_stem"], tag="Question")
                ],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
            }
        )
    elif dataset_name == 'ARC-Easy':
        return DatasetInfo(
            path="ai2_arc",
            name="ARC-Easy",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
            }
        )
    elif dataset_name == 'ARC-Challenge':
        return DatasetInfo(
            path="ai2_arc",
            name="ARC-Challenge",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
            }
        )
    elif dataset_name == 'winogrande':
        return DatasetInfo(
            path="winogrande",
            name="winogrande_xl",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(row["sentence"], tag="Question")
                ],
                "choices": [row[f"option{i + 1}"] for i in range(2)],
                "answer_idx": (
                    None if row["answer"] == "" else
                    int(row["answer"]) - 1
                )
            }
        )
    else:
        raise NotImplementedError
