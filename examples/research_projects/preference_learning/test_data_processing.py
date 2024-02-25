import unittest
import pandas as pd
from collections import namedtuple

from data_processing import (
    load_original_data, process_oasst1_rank, rank_to_pairwise,
    create_assistant_dataset, create_editor_dataset, create_judge_dataset,
)

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.args = namedtuple("Args", ["dataset_name", "subset", "split", "num_workers", "streaming"])
        self.args.dataset_name = "OpenAssistant/oasst1"
        self.args.subset = None
        self.args.split = "validation"
        # train is 3482, 11602, 11602
        self.args.num_workers = 1
        self.args.streaming = False
        
    def test_process_oasst1_rank(self):
        df = pd.DataFrame({
            "lang": ["en", "en", "en", "en", "en"],
            "message_id": [1, 2, 3, 4, 5],
            "parent_id": [None, 1, 1, None, 4],
            "text": ["Hello", "Hi", "Hey", "Bye", "Goodbye"],
            "rank": [1, 2, 3, 4, 5]
        })
        expected_output = pd.DataFrame({
            "instruction": ["Hello", "Bye"],
            "text": [["Hi", "Hey"], ["Goodbye"]],
            "rank": [[2, 3], [5]]
        })
        output = process_oasst1_rank(df)
        pd.testing.assert_frame_equal(output, expected_output)
        
    
    def test_process_oasst1(self):
        df = pd.DataFrame({
            "lang": ["en", "en", "en", "en", "en"],
            "message_id": [1, 2, 3, 4, 5],
            "parent_id": [None, 1, 1, None, 4],
            "text": ["Hello", "Hi", "Hey", "Bye", "Goodbye"],
            "rank": [1, 2, 3, 4, 5]
        })
        expected_output = pd.DataFrame({
            "instruction": ["Hello"],
            "preferred": ["Hi"],
            "dispreferred": ["Hey"]
        })
        output = rank_to_pairwise(process_oasst1_rank(df))
        pd.testing.assert_frame_equal(output, expected_output)
    
    def test_create_assistant_dataset(self):
        output = create_assistant_dataset(self.args)
        assert output.column_names == ["instruction", "response"]
        print(output.shape[0])
        assert output.shape[0] == 188
    
    def test_create_editor_dataset(self):
        output = create_editor_dataset(self.args)
        assert output.column_names == ["instruction", "response"]
        print(output.shape[0])
        assert output.shape[0] == 531
    
    def test_create_judge_dataset(self):
        output = create_judge_dataset(self.args)
        assert output.column_names == ["instruction", "response"]
        print(output.shape[0])
        assert output.shape[0] == 531
        
        