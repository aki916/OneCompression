import json
import tempfile
import unittest
from pathlib import Path

from onecomp.reasoning.data import extract_answer, grade_answer, load_samples, prepare_dataset


class DataTests(unittest.TestCase):
    def test_dataset_revision_must_be_immutable(self):
        for revision in ("main", "v1", "abc", ""):
            with self.assertRaises(ValueError):
                prepare_dataset("math500", "/tmp/unused-reasoning-data.jsonl", revision=revision)

    def test_balanced_box_and_numeric_equivalence(self):
        self.assertEqual(extract_answer(r"x \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertTrue(grade_answer(r"\boxed{\frac{1}{2}}", ".5"))
        self.assertTrue(grade_answer(r"\boxed{042}", "42"))
        self.assertFalse(grade_answer("I tried 42\nbut still thinking 73", "73"))
        self.assertFalse(grade_answer("", ""))
        self.assertTrue(grade_answer("Answer: B.", "B"))

    def test_duplicate_ids_and_empty_split_fail(self):
        row = dict(
            sample_id="a",
            dataset="test",
            question="q",
            answer="a",
            split="development",
            dataset_revision="revision",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_samples(path, "development")
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_samples(path, "evaluation")


if __name__ == "__main__":
    unittest.main()
