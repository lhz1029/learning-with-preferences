# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from trl import SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM

from data_processing import create_assistant_dataset, create_editor_dataset, create_judge_dataset


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="OpenAssistant/oasst1", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default=None, metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    roles: Optional[str] = field(default="assistant", metadata={"help": "if multiple roles, separate by comma"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

set_seed(training_args.seed)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """
    Prepare the text from a sample of the dataset.
    Overrides the chat template from the model.
    (We start with non-chat models anyway though, so this shouldn't matter)

    """
    text = f"###Instruction: {example['instruction']}\n\n###Response: {example['response']}"
    return text


def create_datasets(tokenizer, args, seed=None):
    """
    Returns dataset for training (some combination of roles) and dict of datasets for evaluation.
    No matter the roles, the evaluation datasets are always the same.

    """
    datasets = {}
    datasets["assistant"] = create_assistant_dataset(args)
    datasets["editor"] = create_editor_dataset(args)
    datasets["judge"] = create_judge_dataset(args)
    training_roles = []
    eval_datasets = {}
    for role in ["assistant", "editor", "judge"]:
        dataset = datasets[role]
        if args.streaming:
            print("Loading the dataset in streaming mode")
            valid_data = dataset.take(args.size_valid_set)
            train_data = dataset.skip(args.size_valid_set)
            train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
        else:
            dataset = dataset.train_test_split(test_size=0.005, seed=seed)
            train_data = dataset["train"]
            valid_data = dataset["test"]
            print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

        chars_per_token = chars_token_ratio(train_data, tokenizer)
        print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        # create the mixture of roles for training
        roles = args.roles.split(",")
        if role in roles:
            training_roles.append(train_data)

        # do not add a dataset for evaluation if we're just training on a single role
        # (to avoid repeat evaluation of the same validation data)
        if len(roles) > 1 or role not in roles:
            eval_datasets[role] = ConstantLengthDataset(
                tokenizer,
                valid_data,
                formatting_func=prepare_sample_text,
                infinite=False,
                seq_length=args.seq_length,
                chars_per_token=chars_per_token,
            )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        concatenate_datasets(training_roles),
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_datasets = {
        "valid": valid_dataset
    }
    for k in eval_datasets:
        valid_datasets[k] = eval_datasets[k]
    return train_dataset, valid_datasets

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

train_dataset, eval_datasets = create_datasets(tokenizer, script_args, seed=training_args.seed)

if script_args.packing:
    data_collator = None
else:
    raise ValueError("Use with DataCollatorForCompletionOnlyLM not fully implemented.")
    # TODO maybe we want CompletionOnly data collator at some point
    # Current error:
    # ValueError: You passed `packing=False` to the SFTTrainer, but you didn't pass a `dataset_text_field` or `formatting_func` argument.
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="###Response: ",
        instruction_template="###Instruction: ",
        tokenizer=tokenizer,
        mlm=False,
    )

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
elif is_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
