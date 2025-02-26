import unittest
import argparse
from absl.testing import parameterized
from transformers import AutoTokenizer

from train import create_datasets

def create_original_df(df, role):
    if role == "assistant":
        df["preferred"] = df["response"]
        return df
    elif role == "editor":
        ins = df["instruction"].apply(
            lambda s: s.split("\n\nTask Prompt: ", 1)[1].split("\n\nCandidate answer: ")[0])
        dispreferred = df["instruction"].apply(
            lambda s: s.split("\n\nTask Prompt: ", 1)[1].split("\n\nCandidate answer: ")[1])
        df["instruction"] = ins
        df["dispreferred"] = dispreferred
        df["preferred"] = df["response"]
        return df
    elif role == "judge":
        ins = df["instruction"].apply(
            lambda s: s.split("\n\nTask Prompt: ", 1)[1].split("\n\nCandidate answer A: ")[0])
        df["first"] = df["instruction"].apply(
            lambda s: s.split("\n\nTask Prompt: ", 1)[1]
                        .split("\n\nCandidate answer A: ")[1]
                        .split("\n\nCandidate answer B: ")[0])
        df["second"] = df["instruction"].apply(
            lambda s: s.split("\n\nTask Prompt: ", 1)[1]
                        .split("\n\nCandidate answer A: ")[1]
                        .split("\n\nCandidate answer B: ")[1])
        df["instruction"] = ins
        df["preferred"] = df.apply(lambda s: s["first"] if s["response"] == "A" else s["second"], axis=1)
        df["dispreferred"] = df.apply(lambda s: s["first"] if s["response"] == "B" else s["second"], axis=1)
        return df[["instruction", "preferred", "dispreferred"]]

class TestTrain(parameterized.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = argparse.Namespace(
            size_valid_set=1000,
            shuffle_buffer=1000,
            streaming=False,
            seq_length=1024,
            dataset_name="OpenAssistant/oasst1",
            subset=None,
            split="train",
            num_workers=4,
        )
        self.seed = 42
    
    @parameterized.named_parameters(
        {"testcase_name": "assistant", "roles": "assistant"},
        {"testcase_name": "editor", "roles": "editor"},
        {"testcase_name": "judge", "roles": "judge"},
    )
    def test_create_datasets(self, roles):
        # add roles to argparse Namespace
        self.args.roles = roles
        train_role = roles
        train_dataset, eval_datasets = create_datasets(self.tokenizer, self.args, self.seed)
        
        # assert that there is no overlap between the train and eval datasets
        def no_dataset_overlap(train_dataset, eval_dataset, train_role, eval_role):
            train_df = train_dataset.dataset.to_pandas()
            eval_df = eval_dataset.dataset.to_pandas()
            train_orig = create_original_df(train_df, train_role)
            if eval_role == "valid":
                eval_role = train_role
            eval_orig = create_original_df(eval_df, eval_role)
            if train_role == "assistant" or eval_role == "assistant":
                return train_orig.merge(eval_orig, on=["instruction", "preferred"], how="inner").empty
            else:
                return train_orig.merge(eval_orig, on=["instruction", "preferred", "dispreferred"], how="inner").empty
        
        for eval_role, eval_dataset in eval_datasets.items():
            self.assertTrue(no_dataset_overlap(train_dataset, eval_dataset, train_role, eval_role))