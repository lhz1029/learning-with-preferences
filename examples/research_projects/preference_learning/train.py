# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from peft import AutoPeftModelForCausalLM, LoraConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback

from trl import SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM

from data_processing import (
    load_original_data, process_oasst1_rank,
    create_assistant_dataset, create_editor_dataset, create_judge_dataset
)


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
    
    project_name: Optional[str] = field(default="huggingface", metadata={"help": "the project name"})


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
    text = f"###Instruction: {example['instruction']} ###Response: {example['response']}"
    return text

def formatting_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"###Instruction: {example['instruction'][i]} ###Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def create_datasets(tokenizer, args, seed=None):
    """
    Returns dataset for training (some combination of roles) and dict of datasets for evaluation.
    No matter the roles, the evaluation datasets are always the same.

    """
    if args.streaming:
        raise NotImplementedError("Streaming not yet supported with role creation")
    df = load_original_data(args)
    df = process_oasst1_rank(df)
    # make sure no instruction is repeated in the training and validation set
    # even though there could be duplicate instructions for different message_ids
    df.reset_index(inplace=True)
    assert (df.index.values == df["index"].values).all()
    grouped_indices = df.groupby("instruction")["index"].agg(list)
    train_idx_grouped, valid_idx_grouped = train_test_split(grouped_indices.index, test_size=0.005, random_state=seed)
    train_idx = grouped_indices[train_idx_grouped].explode().values
    valid_idx = grouped_indices[valid_idx_grouped].explode().values
    train_df = df.loc[train_idx]
    valid_df = df.loc[valid_idx]

    fns = {}
    fns["assistant"] = create_assistant_dataset
    fns["editor"] = create_editor_dataset
    fns["judge"] = create_judge_dataset
    training_roles = []
    validation_roles = []
    eval_datasets = {}
    for role in ["assistant", "editor", "judge"]:
        # create the mixture of roles for training
        roles = args.roles.split(",")
        if role in roles:
            train_data = fns[role](train_df)
            chars_per_token = chars_token_ratio(train_data, tokenizer)
            print(f"The character to token ratio of the {role} train dataset is: {chars_per_token:.2f}")
            training_roles.append(train_data)
        # create the evaluation datasets
        valid_data = fns[role](valid_df)
        chars_per_token = chars_token_ratio(valid_data, tokenizer)
        if args.packing:
            eval_datasets[role] = ConstantLengthDataset(
                tokenizer,
                valid_data,
                formatting_func=prepare_sample_text,
                infinite=False,
                seq_length=args.seq_length,
                chars_per_token=chars_per_token,
            )
        else:
            eval_datasets[role] = valid_data
        # create validation data
        if role in roles:
            validation_roles.append(valid_data)

    if args.packing:
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
            concatenate_datasets(validation_roles),
            formatting_func=prepare_sample_text,
            infinite=False,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
        )
    else:
        train_dataset = concatenate_datasets(training_roles)
        valid_dataset = concatenate_datasets(validation_roles)
    valid_datasets = {
        "valid": valid_dataset
    }
    for k in eval_datasets:
        valid_datasets[k] = eval_datasets[k]
    return train_dataset, valid_datasets

def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    prediction_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1))
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Control the frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq


    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        if state.epoch % self.freq == 0:
          # generate predictions
          predictions = self.trainer.predict(self.sample_dataset)
          # decode predictions and labels
          predictions = decode_predictions(self.tokenizer, predictions)
          # add predictions to a wandb.Table
          predictions_df = pd.DataFrame(predictions)
          predictions_df["epoch"] = state.epoch
          records_table = self._wandb.Table(dataframe=predictions_df)
          # log the table to wandb
          self._wandb.log({"sample_predictions": records_table})


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = script_args.project_name
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
    print(tokenizer.padding_side)
    # tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataset, eval_datasets = create_datasets(tokenizer, script_args, seed=training_args.seed)

    if script_args.packing:
        data_collator = None
    else:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=[835, 5103, 29901], # " ###Response: ",
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
        formatting_func=None if script_args.packing else formatting_func,
    )
    if not script_args.packing:
        # TODO: enable logging with packing=True by resolving 'ConstantLengthDataset' object has no attribute 'select'
        progress_callback = WandbPredictionProgressCallback(trainer, tokenizer, eval_datasets["valid"], 3)# 15)
        trainer.add_callback(progress_callback)
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
