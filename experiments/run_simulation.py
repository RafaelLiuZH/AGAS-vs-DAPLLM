import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agas_attacker import AGASAttacker
from src.models.padllm import PADLLMDetector
from src.explainability.shap_explainer import SHAPExplainer
from src.utils.metrics import evaluate_predictions

def main():
    print("=" * 70)
    print("RUNNING BENCHMARK: AGAS (Black-Box Attack) vs. PAD-LLM (Black-Box Defense)")
    print("=" * 70)

    np.random.seed(42)
    R_normal = np.random.choice([0, 1, 2, 3, 4, 5], size=(100, 30), p=[0.7, 0.05, 0.05, 0.05, 0.08, 0.07])
    
    attacker = AGASAttacker(num_bots=10, num_items=30)
    R_combined, y_true = attacker.inject_poisoning_profiles(R_normal, target_item_idx=5)

    detector = PADLLMDetector(num_items=30, label_dim=8)
    y_pred, scores, latency_ms = detector.predict(R_combined)

    precision, recall, f1 = evaluate_predictions(y_true, y_pred)
    
    explainer = SHAPExplainer(label_dim=8)
    bot_indices = np.where(y_true == 1)[0]
    shap_summary = explainer.explain(detector, R_combined, bot_indices)

    print(f"Post-Attack Dataset Size: {R_combined.shape[0]} Users | {R_combined.shape[1]} Items")
    print(f"Detection Results -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
    print(f"Average Detection Latency: {latency_ms:.4f} ms/user")
    print("--- SHAP Feature Attribution Weights ---")
    for k, v in shap_summary.items():
        print(f"  {k}: {v*100:.2f}%")

    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
