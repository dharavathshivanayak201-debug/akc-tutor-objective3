import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)
from akc_tutor.policy import respond


class PolicyTests(unittest.TestCase):
    def test_low(self):
        self.assertEqual(respond("fractions", "", .44, "neutral")["action"], "worked_example")

    def test_boundary_low_is_medium(self):
        self.assertEqual(respond("fractions", "", .45, "neutral")["knowledge_level"], "medium")

    def test_boundary_high_is_medium(self):
        self.assertEqual(respond("fractions", "", .75, "neutral")["knowledge_level"], "medium")

    def test_high(self):
        self.assertEqual(respond("fractions", "", .76, "neutral")["action"], "challenge")

    def test_distress_override(self):
        row = respond("fractions", "", .9, "sadness")
        self.assertEqual(row["knowledge_level"], "high")
        self.assertEqual(row["action"], "hint")

    def test_no_emotion_no_override(self):
        self.assertEqual(respond("fractions", "", .9, "sadness", use_emotion=False)["action"], "challenge")

    def test_no_knowledge_fixed_medium(self):
        self.assertEqual(respond("fractions", "", .9, "neutral", use_knowledge=False)["knowledge_level"], "medium")

    def test_nominal_medium_uses_high(self):
        self.assertEqual(respond("fractions", "", .8367195489743569, "neutral")["knowledge_level"], "high")

    def test_text_is_not_a_generation_input(self):
        self.assertEqual(respond("fractions", "one", .5), respond("fractions", "two", .5))


class MetricTests(unittest.TestCase):
    def test_auc_ties(self):
        self.assertEqual(reproduce.auc([0, 1, 0, 1], [.5] * 4), .5)

    def test_auc_perfect(self):
        self.assertEqual(reproduce.auc([0, 1, 0, 1], [0, 1, 0, 1]), 1.)

    def test_fixed_label_macro(self):
        self.assertEqual(reproduce.classification([0, 1], [0, 1], labels=4)["macro_f1"], .5)

    def test_exact_signflip(self):
        self.assertEqual(reproduce.exact_signflip([.5, .5, .5, 0]), .25)

    def test_exact_all_ties(self):
        self.assertEqual(reproduce.exact_signflip([0] * 24), 1.)

    def test_holm(self):
        np.testing.assert_allclose(reproduce.holm([.01, .03, .2, 1]), [.04, .09, .4, 1])

    def test_saved_policy_replay(self):
        self.assertEqual(reproduce.phase4()["responses_replayed"], 96)

    def test_final_ratings(self):
        rows = reproduce.phase5()
        self.assertEqual(rows["ratings"], 192)
        self.assertLess(rows["criteria"][-1]["exact_holm_p"], .05)
        self.assertGreater(rows["criteria"][1]["exact_holm_p"], .05)

    def test_preserved_input_integrity(self):
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_integrity.py")], check=True, capture_output=True)

    def test_phase12_saved_metrics_and_bootstraps(self):
        rows = reproduce.phase12()
        self.assertEqual(rows["1"]["n"], 4590)
        self.assertEqual(rows["2"]["bootstrap_replicates"], 1000)

    def test_notebook_python_cells_compile_without_colab_magics(self):
        notebooks = list((ROOT / "research/code").glob("phase*/*.ipynb"))
        self.assertEqual(len(notebooks), 5)
        for path in notebooks:
            data = json.loads(path.read_text())
            for index, cell in enumerate(data["cells"]):
                if cell["cell_type"] != "code":
                    continue
                code = "".join(cell["source"])
                code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith(("!", "%")))
                compile(code, f"{path.name}:cell{index}", "exec")


if __name__ == "__main__":
    unittest.main()
