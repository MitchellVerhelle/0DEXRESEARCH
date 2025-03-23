import numpy as np

class AirdropPolicy:
    """
    Default policy that assigns a token reward equal to the normalized airdrop_points.
    """
    def calculate_tokens(self, airdrop_points, user):
        # Returns a normalized token reward in [0,1].
        return airdrop_points

class LinearAirdropPolicy(AirdropPolicy):
    """
    Linear policy: tokens = factor * airdrop_points.
    Default factor is 1, so a normalized score of 1 yields a reward of 1.
    """
    def __init__(self, factor=1.0):
        self.factor = factor

    def calculate_tokens(self, airdrop_points, user):
        return self.factor * airdrop_points

class ExponentialAirdropPolicy(AirdropPolicy):
    """
    Exponential policy: tokens = factor * (exp(airdrop_points / scaling) - 1).
    With airdrop_points in [0,1], use scaling=1 so that:
      - For 0, reward = 0;
      - For 1, reward = factor*(e^1 - 1) ~ factor*1.718.
    You can adjust factor as needed.
    """
    def __init__(self, factor=1.0, scaling=1.0):
        self.factor = factor
        self.scaling = scaling

    def calculate_tokens(self, airdrop_points, user):
        # Optionally cap airdrop_points to avoid extreme values.
        points = min(airdrop_points, 1.0)
        return self.factor * (np.exp(points / self.scaling) - 1)

class TieredConstantAirdropPolicy(AirdropPolicy):
    """
    Tiered Constant policy on a normalized scale.
    
    Default tiers (normalized):
      - If points < 0.2, reward = 0.1.
      - If 0.2 <= points < 0.6, reward = 0.4.
      - If points >= 0.6, reward = 1.0.
    """
    def __init__(self, tiers=None):
        if tiers is None:
            self.tiers = [(0.2, 0.1), (0.6, 0.4), (np.inf, 1.0)]
        else:
            self.tiers = tiers

    def calculate_tokens(self, airdrop_points, user):
        for threshold, token_amt in self.tiers:
            if airdrop_points < threshold:
                return token_amt
        return self.tiers[-1][1]

class TieredLinearAirdropPolicy(AirdropPolicy):
    """
    Tiered Linear policy on a normalized scale.
    Default tiers (normalized):
      - For points up to 0.2, use factor 1.0.
      - For points between 0.2 and 0.6, use factor 1.5.
      - For points above 0.6, use factor 2.0.
      
    The reward is computed cumulatively.
    """
    def __init__(self, tiers=None):
        if tiers is None:
            self.tiers = [(0.2, 1.0), (0.6, 1.5), (np.inf, 2.0)]
        else:
            self.tiers = tiers

    def calculate_tokens(self, airdrop_points, user):
        tokens = 0.0
        prev_threshold = 0.0
        for threshold, factor in self.tiers:
            if airdrop_points <= threshold:
                tokens += (airdrop_points - prev_threshold) * factor
                return tokens
            else:
                tokens += (threshold - prev_threshold) * factor
                prev_threshold = threshold
        return tokens

class TieredExponentialAirdropPolicy(AirdropPolicy):
    """
    Tiered Exponential policy on a normalized scale.
    Default tiers (normalized):
      - For points up to 0.2: factor=1.0, scaling=0.2.
      - For points between 0.2 and 0.6: factor=1.5, scaling=0.4.
      - For points above 0.6: factor=2.0, scaling=0.4.
      
    The reward is computed cumulatively.
    """
    def __init__(self, tiers=None):
        if tiers is None:
            self.tiers = [
                (0.2, {'factor': 1.0, 'scaling': 0.2}),
                (0.6, {'factor': 1.5, 'scaling': 0.4}),
                (np.inf, {'factor': 2.0, 'scaling': 0.4})
            ]
        else:
            self.tiers = tiers

    def calculate_tokens(self, airdrop_points, user):
        tokens = 0.0
        prev_threshold = 0.0
        for threshold, params in self.tiers:
            factor = params.get('factor', 1.0)
            scaling = params.get('scaling', 0.2)
            if airdrop_points <= threshold:
                tokens += factor * (np.exp((airdrop_points - prev_threshold) / scaling) - 1)
                return tokens
            else:
                tokens += factor * (np.exp((threshold - prev_threshold) / scaling) - 1)
                prev_threshold = threshold
        return tokens
