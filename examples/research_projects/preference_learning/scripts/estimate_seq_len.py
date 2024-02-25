from collections import namedtuple
from transformers import AutoTokenizer
from data_processing import (
    create_assistant_dataset, create_editor_dataset, create_judge_dataset,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_sample_text(example):
    """
    Prepare the text from a sample of the dataset.
    Overrides the chat template from the model.
    (We start with non-chat models anyway though, so this shouldn't matter)
    
    """
    text = f"###Instruction: {example['instruction']}\n\n###Response: {example['response']}"
    return text


if __name__ == "__main__":
    args = namedtuple("Args", ["dataset_name", "subset", "split", "num_workers", "streaming"])
    args.dataset_name = "OpenAssistant/oasst1"
    args.subset = None
    args.split = "validation"
    args.num_workers = 1
    args.streaming = False

    fn_dict = {
        "assistant": create_assistant_dataset,
        "editor": create_editor_dataset,
        "judge": create_judge_dataset,
    }
    for role in ["assistant", "editor", "judge"]:
        print(role)
        dataset = fn_dict[role](args)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        nb_examples = 100
        token_counts = []
        for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
            text = prepare_sample_text(example)
            if tokenizer.is_fast:
                token_counts.append(len(tokenizer(text).tokens()))
            else:
                token_counts.append(len(tokenizer.tokenize(text)))

        print("Max: ", max(token_counts))
        print("Min: ", min(token_counts))
        print("Mean: ", sum(token_counts) / len(token_counts))
        plt.hist(token_counts)
        plt.xlabel("Sequence length")
        plt.ylabel("Number of sequences")
        plt.title("Distribution of sequence lengths")
        plt.savefig(f"sequence_length_distribution_{role}.png")