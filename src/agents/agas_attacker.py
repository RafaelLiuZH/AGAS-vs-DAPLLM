import numpy as np

class AGASAttacker:
    """AGAS Black-Box Shilling Attacker injecting poisoning profiles."""

    def __init__(self, num_bots=10, num_items=30):
        self.num_bots = num_bots
        self.num_items = num_items

    def inject_poisoning_profiles(self, R_normal, target_item_idx=5):
        num_users, num_items = R_normal.shape
        R_poison = np.copy(R_normal)

        bot_profiles = np.zeros((self.num_bots, num_items))
        for i in range(self.num_bots):
            bot_profiles[i, target_item_idx] = 5.0
            
            filler_indices = np.random.choice(num_items, size=4, replace=False)
            for idx in filler_indices:
                if idx != target_item_idx:
                    bot_profiles[i, idx] = np.random.choice([3.0, 4.0, 5.0])

        R_combined = np.vstack([R_poison, bot_profiles])
        labels = np.array([0] * num_users + [1] * self.num_bots)
        return R_combined, labels
