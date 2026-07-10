import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

driftguard_spec = importlib.util.spec_from_file_location(
    "driftguard_helpers", PROJECT_ROOT / "models" / "driftguard.py"
)
driftguard_helpers = importlib.util.module_from_spec(driftguard_spec)
driftguard_spec.loader.exec_module(driftguard_helpers)
box_motion_consistency = driftguard_helpers.box_motion_consistency
select_evidence_mask = driftguard_helpers.select_evidence_mask
from evaluate_rmot import evaluate  # noqa: E402


class EvidenceSelectionTest(unittest.TestCase):
    def test_threshold_and_budget_are_independent(self):
        scores = torch.tensor([0.10, 0.30, 0.40, 0.80])
        valid = torch.ones_like(scores, dtype=torch.bool)

        keep = select_evidence_mask(scores, valid, topk=2, threshold=0.35)
        self.assertEqual(keep.tolist(), [False, False, True, True])

        strict = select_evidence_mask(scores, valid, topk=2, threshold=0.60)
        self.assertEqual(strict.tolist(), [False, False, False, True])

        capped = select_evidence_mask(scores, valid, topk=1, threshold=0.35)
        self.assertEqual(capped.tolist(), [False, False, False, True])

    def test_selection_falls_back_to_best_valid_token(self):
        scores = torch.tensor([0.10, 0.20, 0.90])
        valid = torch.tensor([True, True, False])
        keep = select_evidence_mask(scores, valid, topk=2, threshold=0.95)
        self.assertEqual(keep.tolist(), [False, True, False])

    def test_motion_consistency_penalizes_large_displacement(self):
        previous = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        near = torch.tensor([[0.51, 0.50, 0.2, 0.2]])
        far = torch.tensor([[0.90, 0.90, 0.2, 0.2]])
        valid = torch.tensor([True])
        self.assertGreater(
            box_motion_consistency(near, previous, valid).item(),
            box_motion_consistency(far, previous, valid).item(),
        )


class RmotEvaluationTest(unittest.TestCase):
    @staticmethod
    def write_case(root, prediction_rows):
        case = root / "0001" / "expression"
        case.mkdir(parents=True)
        (case / "gt.txt").write_text(
            "1,1,10,10,20,20,1,1,1\n"
            "2,1,10,10,20,20,1,1,1\n"
        )
        (case / "predict.txt").write_text(prediction_rows)

    def test_perfect_tracks_score_one_hundred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_case(
                root,
                "1,7,10,10,20,20,1,1,1\n"
                "2,7,10,10,20,20,1,1,1\n",
            )
            result = evaluate(root)
            self.assertAlmostEqual(result["HOTA"], 100.0)
            self.assertAlmostEqual(result["AssA"], 100.0)
            self.assertEqual(result["IDSW"], 0)

    def test_clear_reports_identity_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_case(
                root,
                "1,7,10,10,20,20,1,1,1\n"
                "2,8,10,10,20,20,1,1,1\n",
            )
            result = evaluate(root)
            self.assertEqual(result["IDSW"], 1)


if __name__ == "__main__":
    unittest.main()
