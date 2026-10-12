import unittest
from dataclasses import asdict

import torch

from project1.data import SymbolTaskConfig
from project1.extended import oracle_split
from project1.final_evaluation import final_split


class FinalEvaluationTests(unittest.TestCase):
    def test_final_split_is_deterministic_and_excludes_all_study_prompts(self):
        task = SymbolTaskConfig()
        splits = {name: oracle_split(task, 16, 400 + index, name)
                  for index, name in enumerate(("train", "validation", "test", "diagnostic"))}
        study = {"clean": {"protocol": {"task": asdict(task)}, "splits": splits}}

        first, metadata = final_split(study, 7, 32)
        repeated, repeated_metadata = final_split(study, 7, 32)
        excluded = {tuple(prompt) for split in splits.values()
                    for prompt in split["input_ids"].tolist()}
        self.assertTrue(torch.equal(first["input_ids"], repeated["input_ids"]))
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(metadata["excluded_prompt_count"], len(excluded))
        self.assertFalse(excluded & set(map(tuple, first["input_ids"].tolist())))
