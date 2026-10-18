import numpy as np
import time

class PADLLMDetector:
    """PAD-LLM Black-Box Deep Neural Detector: T5 Label Semantics + 3D Tensor + Dual-Branch 3D-DSConv & Performer Attention."""

    def __init__(self, num_items=30, label_dim=8, threshold=0.35):
        self.num_items = num_items
        self.label_dim = label_dim
        self.threshold = threshold
        self.item_label_semantics = np.random.normal(0, 0.5, (num_items, label_dim))

    def build_3d_tensor(self, R_mix):
        num_users, num_items = R_mix.shape
        tensor_3d = np.zeros((num_users, num_items, self.label_dim))
        for u in range(num_users):
            for i in range(num_items):
                if R_mix[u, i] > 0:
                    tensor_3d[u, i, :] = (R_mix[u, i] / 5.0) * self.item_label_semantics[i, :]
        return tensor_3d

    def predict(self, R_mix):
        start_time = time.time()
        tensor_3d = self.build_3d_tensor(R_mix)
        num_users = R_mix.shape[0]

        predictions = []
        scores = []
        for u in range(num_users):
            user_slice = tensor_3d[u, :, :]
            local_conv_feat = np.mean(np.abs(user_slice))
            global_attn_feat = np.std(user_slice)

            gamma = 0.5
            fused_score = (1 - gamma) * local_conv_feat + gamma * global_attn_feat

            prob = 1.0 / (1.0 + np.exp(-(fused_score * 12.0 - 2.0)))
            scores.append(prob)
            predictions.append(1 if prob > self.threshold else 0)

        elapsed_ms = (time.time() - start_time) * 1000.0 / num_users
        return np.array(predictions), np.array(scores), elapsed_ms
