import os
import json
import pandas as pd
from accelerate import Accelerator
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    pipeline,
    HfArgumentParser,
    set_seed,
    GenerationConfig,
)
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from transformers.pipelines.pt_utils import KeyDataset

from data_processing import (
    load_assistant_dataset, load_editor_dataset, load_judge_dataset,
    create_assistant_dataset, create_editor_dataset, create_judge_dataset
)
# from accelerate import PartialState
# from diffusers import DiffusionPipeline

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="OpenAssistant/oasst1", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default=None, metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="validation", metadata={"help": "the split to use"})
    split_subset: Optional[str] = field(default="validation", metadata={"help": "the subset split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    role: Optional[str] = field(default="assistant", metadata={"help": "if multiple role, separate by comma"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    
    project_name: Optional[str] = field(default="huggingface", metadata={"help": "the project name"})
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "the project name"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    
    output_dir: Optional[str] = field(default="generations", metadata={"help": "the output directory"})
    output_filename: Optional[str] = field(default="generations.jsonl", metadata={"help": "the output filename"})
    num_prompts: Optional[int] = field(default=None, metadata={"help": "the number of prompts"})
    
    dataset_filename: Optional[str] = field(default=None, metadata={"help": "jsonl filename to initialize the dataset"})

def process_incoming(df, incoming_format):
    """
    Note that the ranking is flipped so that the dispreferred has rank 1.
    This way when we create the editor dataset, the dispreferred will be in the input,
    and the preferred will be thrown away.
    (The editor dataset is the only place the preferred vs. dispreferred distinction is important.)
    
    """
    if incoming_format == "assistant":
        df["instruction"] = df["input"].apply(lambda s: s.split("\n\nTask Prompt:")[-1])
        df["text"] = df["generation"].apply(lambda s: s.split(" ###Response:", 1)[1])
        df["text"] = df.text.apply(lambda s: [s, ""])
        df["rank"] = [[2, 1]]*len(df)
        return df[["instruction", "text", "rank"]]
    elif incoming_format == "editor":
        df["instruction"] = df["input"].apply(lambda s: s.split("\n\nTask Prompt:")[-1].split("\n\nCandidate answer:", 1)[0])
        df["preferred"] = df["generation"].apply(lambda s: s.split(" ###Response:", 1)[1])
        print(df[~df.input.str.contains("\n\nCandidate answer:")].input.values)
        print(df[~df.input.str.contains("\n\nCandidate answer:")].index)
        df["dispreferred"] = df["input"].apply(lambda s: s.split("\n\nCandidate answer:", 1)[1])
        df["text"] = df.apply(lambda row: [row["preferred"], row["dispreferred"]], axis=1)
        df["rank"] = [[2, 1]]*len(df)
        print(df)
        return df
    elif incoming_format == "judge":
        df["instruction"] = df["input"].apply(lambda s: s.split("\n\nTask Prompt:")[-1].split("\n\nCandidate answer:", 1)[0])
        df["textA"] = df["input"].apply(lambda s: s.split("\n\nCandidate answer A:", 1)[1].split("\n\nCandidate answer B:", 1)[0])
        df["textB"] = df["input"].apply(lambda s: s.split("\n\nCandidate answer B:", 1)[1])
        # Note: this will prioritize B predictions in the case that the judge outputs something that is neither A nor B
        df["rank"] = df["generation"].apply(lambda s: [2, 1] if s.endswith("A") else [1, 2])
        return df

def load_dataset_from_json(args):
    """
    """
    with open(args.dataset_filename, "r") as f:
        data = f.readlines()
    df = pd.DataFrame([json.loads(d) for d in data])
    # infer incoming format from the first row
    if df["input"][0].startswith("Review the task prompt and candidate answer and rewrite the answer to be higher-quality."):
        incoming_format = "editor"
    elif df["input"][0].startswith("Review the task prompt and candidate answers and choose the answer (A or B) that is higher-quality."):
        incoming_format = "judge"
    else:
        incoming_format = "assistant"
    df = process_incoming(df, incoming_format)
    # get dataset in format instruction, text, rank
    fn = {}
    fn["assistant"] = create_assistant_dataset
    fn["editor"] = create_editor_dataset
    fn["judge"] = create_judge_dataset
    return fn[script_args.role](df)

def format_prompt(instruction):
    """
    Format the prompt to match what was seen in the training data.

    """
    text = f"###Instruction: {instruction} ###Response: "
    return text

def format_prompt_mistral(instruction):
    """
    Format the prompt to match the mistral chat template.
    (Also matches the llama2 chat template)

    """
    text = f"<s>[INST] {instruction} [/INST] "
    return text

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    os.environ["WANDB_PROJECT"] = script_args.project_name
    set_seed(script_args.seed)

    print(f"Starting to load the model {script_args.model_name} into memory")
    
    # adapters_name = "lucas0/empath-llama-7b"
    # m = PeftModel.from_pretrained(m, adapters_name)
    # m = m.merge_and_unload()
    # tok = LlamaTokenizer.from_pretrained(model_name)
    # tok.bos_token_id = 1
    if script_args.ckpt_path:
        model = AutoPeftModelForCausalLM.from_pretrained(script_args.ckpt_path, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
            
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=bnb_config,
            device_map={"": Accelerator().local_process_index},
            trust_remote_code=True,
            use_auth_token=True,
        )
        

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    print("tokenizer.pad_token_id", tokenizer.pad_token_id)
    
    if script_args.dataset_filename:
        dataset = load_dataset_from_json(script_args)
    else:
        fn = {}
        fn["assistant"] = load_assistant_dataset
        fn["editor"] = load_editor_dataset
        fn["judge"] = load_judge_dataset
        dataset = fn[script_args.role](script_args)
        
    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        script_args.model_name, do_sample=True, return_unused_kwargs=True
    )
    print(generation_config)

    output_filename = os.path.join(script_args.output_dir, script_args.output_filename)
    parent_path = os.path.dirname(output_filename)
    os.makedirs(parent_path, exist_ok=True)
    dataset = dataset.select(range(script_args.num_prompts)) if script_args.num_prompts else dataset
    for element in dataset:
        # maybe want to use a chat template at some point
        # response_tokens = [835, 5103, 29901, 29871]
        # print("tokenizer.pad_token_id", tokenizer.pad_token_id)
        # print("tokenizer.pad_token", tokenizer.pad_token)
        # print("tokenizer.chat_template", tokenizer.chat_template)
        # print("tokenizer.default_chat_template", tokenizer.chat_template)
        if script_args.ckpt_path:
            model_inputs = tokenizer([format_prompt(element["instruction"])], return_tensors="pt").to("cuda")
        else:
            model_inputs = tokenizer([format_prompt_mistral(element["instruction"])], return_tensors="pt").to("cuda")
            print(model_inputs)

        # print(model_inputs["input_ids"].shape)
        generated_ids = model.generate(**model_inputs, max_new_tokens=4096 - model_inputs["input_ids"].shape[1])
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = {"input": element["instruction"],
                  "output": output,
                  "generation": output.split(element["instruction"], 1)[1],
                  "reference": element["response"]}
        with open(output_filename, "a") as f:
            f.write(json.dumps(result) + "\n")

    # batching seems considerably slower for some reason
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=2, truncation=True, return_full_text=False)
    # for outputs in generator(KeyDataset(dataset.select(range(script_args.num_prompts)), "instruction")):
    #     outputs = generator(inputs)
    #     print(outputs)
    #     results = [text.split(" ###Response", 1) for text in outputs]
    #     results = {text[0]: text[1] if len(text) == 2 else "" for text in results}
    #     with open(output_filename, "a") as f:
    #         f.write(json.dumps("\n".join(results) + "\n"))
   
    # distributed_state = PartialState()
    # pipeline.to(distributed_state.device)

    # with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    #     result = pipeline(prompt).images[0]
    #     result.save(f"result_{distributed_state.process_index}.png")
        