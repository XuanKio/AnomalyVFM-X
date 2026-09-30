import unittest
import tempfile
from pathlib import Path

from pilot_quality_gate import evaluate, write_gate


class PilotQualityGateTest(unittest.TestCase):
    def test_rejects_healthy_false_positive_even_when_bad_is_detected(self):
        report = {
            "threshold": 0.5,
            "weak_train_pairs": ["cable"],
            "heldout_rejected_pairs": ["metal"],
            "samples": [
                {"image": "cable_good", "variant": "baseline", "mask_pixels_at_0_5": 0},
                {"image": "cable_good", "variant": "adapted", "mask_pixels_at_0_5": 7882},
                {"image": "cable_bad", "variant": "adapted", "mask_pixels_at_0_5": 10414,
                 "weak_label_recall": .9},
                {"image": "metal_bad", "variant": "adapted", "mask_pixels_at_0_5": 20},
            ],
        }
        result = evaluate(report)
        self.assertFalse(result["local_checks_passed"])
        self.assertTrue(any("cable_good" in issue for issue in result["problems"]))
        self.assertFalse(result["eligible_for_release"])

    def test_experiment_page_displays_failure(self):
        report = {"threshold": 0.5, "weak_train_pairs": [],
                  "heldout_rejected_pairs": [], "samples": [
                      {"image": "cable_good", "variant": "baseline", "mask_pixels_at_0_5": 0},
                      {"image": "cable_good", "variant": "adapted", "mask_pixels_at_0_5": 10},
                  ]}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "comparison.html").write_text("<html><h1>Comparison</h1></html>", encoding="utf-8")
            write_gate(folder, report)
            page = (folder / "comparison.html").read_text(encoding="utf-8")
            self.assertIn("KHÔNG ĐẠT", page)
            self.assertIn("cable_good", page)
            self.assertTrue((folder / "gate.json").exists())


if __name__ == "__main__":
    unittest.main()
