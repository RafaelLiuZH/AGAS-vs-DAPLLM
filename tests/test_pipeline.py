import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agas_attacker import AGASAttacker
from src.models.padllm import PADLLMDetector

class TestPipeline(unittest.TestCase):
    def test_agas_attacker(self):
        R_normal = np.zeros((10, 20))
        attacker = AGASAttacker(num_bots=5, num_items=20)
        R_comb, labels = attacker.inject_poisoning_profiles(R_normal)
        self.assertEqual(R_comb.shape[0], 15)
        self.assertEqual(sum(labels), 5)

    def test_padllm_detector(self):
        R_comb = np.random.choice([0, 1, 2, 3, 4, 5], size=(15, 20))
        detector = PADLLMDetector(num_items=20, label_dim=8)
        preds, scores, lat = detector.predict(R_comb)
        self.assertEqual(len(preds), 15)
        self.assertGreater(lat, 0)

if __name__ == "__main__":
    unittest.main()
