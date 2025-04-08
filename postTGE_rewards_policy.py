import numpy as np

class EngagementMultiplierPolicy:
    """
    A post-TGE rewards policy that calculates a multiplier for a user's token rewards
    based on their engagement on the platform after TGE.

    multiplier = 1 + gamma * (active_days / simulation_horizon) ^ delta
    """
    def __init__(self, gamma=0.5, delta=1.0, simulation_horizon=60):
        self.gamma = gamma
        self.delta = delta
        self.simulation_horizon = simulation_horizon

    def calculate_multiplier(self, active_days):
        fraction_active = active_days / self.simulation_horizon
        multiplier = 1.0 + self.gamma * (fraction_active ** self.delta)
        return multiplier


class GenericPostTGERewardPolicy:
    """
    A generic post-TGE reward policy that applies an engagement multiplier
    to a user's token rewards. 
    """
    def __init__(self, engagement_policy=None):
        self.engagement_policy = engagement_policy if engagement_policy is not None else EngagementMultiplierPolicy()
    
    def apply_rewards(self, user, active_days):
        """
        Update the user's token rewards by applying the engagement multiplier.
        Only apply if user is active. (But we check that outside in user.step already.)
        """
        multiplier = self.engagement_policy.calculate_multiplier(active_days)
        user.tokens *= multiplier
