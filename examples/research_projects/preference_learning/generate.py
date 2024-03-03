import os
import json
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
    load_original_data, process_oasst1_rank,
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

    roles: Optional[str] = field(default="assistant", metadata={"help": "if multiple roles, separate by comma"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    
    project_name: Optional[str] = field(default="huggingface", metadata={"help": "the project name"})
    ckpt_path: Optional[str] = field(default="checkpoints/sft_oasst", metadata={"help": "the project name"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    
    output_dir: Optional[str] = field(default="generations", metadata={"help": "the output directory"})
    output_filename: Optional[str] = field(default="generations.jsonl", metadata={"help": "the output filename"})
    num_prompts: Optional[int] = field(default=10, metadata={"help": "the number of prompts"})

def load_assistant_dataset(args):
    df = load_original_data(args)
    df = process_oasst1_rank(df)
    return create_assistant_dataset(df)

def load_editor_dataset(args):
    df = load_original_data(args)
    df = process_oasst1_rank(df)
    return create_editor_dataset(df)

def load_judge_dataset(args):
    df = load_original_data(args)
    df = process_oasst1_rank(df)
    return create_judge_dataset(df)

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    os.environ["WANDB_PROJECT"] = script_args.project_name
    set_seed(script_args.seed)

    print(f"Starting to load the model {script_args.model_name} into memory")

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
        
    # m = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name,
    #     quantization_config=bnb_config,
    #     device_map={"": Accelerator().local_process_index},
    #     trust_remote_code=True,
    #     use_auth_token=True,
    # )
    
    # adapters_name = "lucas0/empath-llama-7b"
    # m = PeftModel.from_pretrained(m, adapters_name)
    # m = m.merge_and_unload()
    # tok = LlamaTokenizer.from_pretrained(model_name)
    # tok.bos_token_id = 1

    model = AutoPeftModelForCausalLM.from_pretrained(script_args.ckpt_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    print(tokenizer.pad_token_id)
    
    fn = {}
    fn["assistant"] = load_assistant_dataset
    fn["editor"] = load_editor_dataset
    fn["judge"] = load_judge_dataset
    dataset = fn[script_args.roles](script_args)
    
    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        script_args.model_name, do_sample=True, return_unused_kwargs=True
    )
    print(generation_config)

    output_filename = os.path.join(script_args.output_dir, script_args.output_filename)
    parent_path = os.path.dirname(output_filename)
    os.makedirs(parent_path, exist_ok=True)
    for element in dataset.select(range(script_args.num_prompts)):
        model_inputs = tokenizer([element["instruction"]], return_tensors="pt").to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = {"input": element["instruction"],
                  "generation": output.split(element["instruction"], 1)[1],
                  "reference": element["response"]}
        with open(output_filename, "a") as f:
            f.write(json.dumps(result) + "\n")

    # batching seems considerably slower for some reason
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=2, truncation=True, return_full_text=False)
    # for outputs in generator(KeyDataset(dataset.select(range(script_args.num_prompts)), "instruction")):
    #     outputs = generator(inputs)
    #     print(outputs)
    #     results = [text.split("\n\n###Response", 1) for text in outputs]
    #     results = {text[0]: text[1] if len(text) == 2 else "" for text in results}
    #     with open(output_filename, "a") as f:
    #         f.write(json.dumps("\n".join(results) + "\n"))
   
    # distributed_state = PartialState()
    # pipeline.to(distributed_state.device)

    # with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    #     result = pipeline(prompt).images[0]
    #     result.save(f"result_{distributed_state.process_index}.png")
        