from itertools import combinations
import pandas as pd

from datasets import load_dataset, Dataset


def load_original_data(args):
    """
    English-only.
    Only the first interaction of each conversation.
    All pairwise combinations from the rankings.
    
    """
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    df = dataset.to_pandas()
    return df

def process_oasst1_rank(df):
    """
    English-only.
    Only the first interaction of each conversation.
    All pairwise combinations from the rankings.
    This also means that we lose conversations with only one message.
    
    """
    # English only
    df = df[df.lang == "en"]
    # Only first interaction in each conversation
    subset = df[df["parent_id"].isnull()]
    subset = subset[['message_id', 'text']]
    subset.rename(columns={"text": "instruction"}, inplace=True)
    subset_merged = subset.merge(df[["parent_id", "text", "rank"]], left_on="message_id", right_on="parent_id", how="left")
    assert len(subset_merged.groupby("message_id").text.agg(list)) == len(subset_merged.groupby(["message_id", "instruction"]).text.agg(list))
    subset_merged = subset_merged.groupby(["message_id", "instruction"])[["text", "rank"]].agg(list)
    subset_merged.reset_index(inplace=True)
    return subset_merged[['instruction', 'text', 'rank']]

def rank_to_pairwise(df):
    """
    Accepts a dataframe with columns instruction, text, and rank.
    Returns a dataframe with columns instruction, preferred, and dispreferred.
    
    """
    # turn into pairwise combinations
    rows = []
    for _, row in df.iterrows():
        text_pairs = combinations(row["text"], 2)
        rank_pairs = combinations(row["rank"], 2)
        for text_pair, rank_pair in zip(text_pairs, rank_pairs):
            if rank_pair[0] < rank_pair[1]:
                preferred = text_pair[0]
                dispreferred = text_pair[1]
            elif rank_pair[0] > rank_pair[1]:
                preferred = text_pair[1]
                dispreferred = text_pair[0]
            else:  # if the ranks are the same, we skip (shouldn't happen in this dataset)
                continue
            rows.append([row["instruction"], preferred, dispreferred])
    return pd.DataFrame(rows, columns=["instruction", "preferred", "dispreferred"])

def create_assistant_dataset(df):
    """ 
    Assumes data is already processed as instruction, text, rank.
    Returns a Dataset. 
    Keys are instruction and responses.
    Only returns the best response of the n choices.
    
    """
    df["response"] = df.apply(lambda s: s["text"][s["rank"].index(min(s["rank"]))], axis=1)
    df = df[["instruction", "response"]]
    return Dataset.from_pandas(df)

def create_editor_dataset(df):
    """ 
    Assumes data is already processed as instruction, text, rank.
    Returns a Dataset. 
    Keys are instruction and responses.
    
    """
    df = rank_to_pairwise(df)
    df["instruction"] = df.apply(
        lambda s: "Review the task prompt and candidate answer and rewrite the answer to be higher-quality.\n\nTask Prompt: " + 
        s["instruction"] + "\n\nCandidate answer: " + s["dispreferred"], axis=1)
    df["response"] = df["preferred"]
    df = df[["instruction", "response"]]
    return Dataset.from_pandas(df)

def create_judge_dataset(df):
    """ 
    Assumes data is already processed as instruction, text, rank.
    Returns a Dataset. 
    Keys are instruction and responses.
    
    """
    df = rank_to_pairwise(df)
    # preferred comes first half the time
    df1, df2 = df[::2], df[1::2]
    df1["instruction"] = df1.apply(
        lambda s: "Review the task prompt and candidate answers and choose the answer (A or B) that is higher-quality.\n\nTask Prompt: " + 
        s["instruction"] + "\n\nCandidate answer A: " + s["preferred"] + "\n\nCandidate answer B: " + s["dispreferred"], axis=1)
    df1["response"] = "A"
    # preferred comes second half the time
    df2["instruction"] = df2.apply(
        lambda s: "Review the task prompt and candidate answers and choose the answer (A or B) that is higher-quality.\n\nTask Prompt: " + 
        s["instruction"] + "\n\nCandidate answer A: " + s["dispreferred"] + "\n\nCandidate answer B: " + s["preferred"], axis=1)
    df2["response"] = "B"
    df = pd.concat([df1[["instruction", "response"]], df2[["instruction", "response"]]], axis=0)
    return Dataset.from_pandas(df, preserve_index=False)

        