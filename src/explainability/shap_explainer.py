import numpy as np

class SHAPExplainer:
    """SHAP value attribution for PAD-LLM decision interpretability."""

    def __init__(self, label_dim=8):
        self.label_dim = label_dim

    def explain(self, detector, R_mix, bot_indices):
        tensor_3d = detector.build_3d_tensor(R_mix)
        bot_tensor = tensor_3d[bot_indices, :, :]
        
        feature_importance = np.mean(np.abs(bot_tensor), axis=(0, 1))
        normalized_weights = feature_importance / np.sum(feature_importance)

        shap_summary = {
            f"Label_Dimension_{i+1}": float(normalized_weights[i])
            for i in range(self.label_dim)
        }
        return shap_summary
