import numpy as np
from abc import ABC, abstractmethod
from airdrop_policy import AirdropPolicy

class User(ABC):
    """
    Abstract base class for all users in the simulation.
    """
    def __init__(self, wealth, user_id, airdrop_policy=None):
        self.user_id = user_id
        self.wealth = wealth
        self.airdrop_policy = airdrop_policy if airdrop_policy is not None else AirdropPolicy()
        self.airdrop_points = 0.0
        self.tokens = 0.0
        self.active = True

    def update_airdrop_points(self, dt):
        """
        Simulate "farming" for airdrop points
        Crypto airdrops: An evolutionary approach by Darcy W. E. Allen (2024)
        """
        delta_points = (self.interaction_rate * (self.endowment - self.airdrop_points) - self.decay_rate * self.airdrop_points) * dt
        self.airdrop_points += delta_points
        if self.airdrop_points < 0:
            self.airdrop_points = 0

    @abstractmethod
    def step(self, phase):
        """
        Advance the user's state based on the current simulation phase.
        Phases can be: 'PreTGE', 'TGE', 'PostTGE', etc.
        """
        pass

class RegularUser(User):
    def __init__(self, wealth, user_id, user_size, airdrop_policy=None):
        super().__init__(wealth, user_id, airdrop_policy)
        self.user_size = user_size
        self.decay_rate = 0.1
        self.active_days = 0

        if user_size == 'small':
            self.interaction_rate = np.random.poisson(lam=1)
            self.endowment = np.random.poisson(lam=1)
        elif user_size == 'medium':
            self.interaction_rate = np.random.poisson(lam=3)
            self.endowment = np.random.poisson(lam=3)
        elif user_size == 'large':
            self.interaction_rate = np.random.poisson(lam=5)
            self.endowment = np.random.poisson(lam=5)
        else:
            self.interaction_rate = 1
            self.endowment = 1
        
        self.endowment += 0.1 #initial baseline endowment for users.

    def step(self, phase, current_price=None, baseline_price=None, postTGE_rewards_policy=None):
        if phase == 'PreTGE':
            self.update_airdrop_points(dt=1)
        elif phase == 'TGE':
            self.tokens = self.airdrop_policy.calculate_tokens(self.airdrop_points, self)
        elif phase == 'PostTGE':
            # Decide whether to remain active, based on:
            # 1) user_size base probability
            # 2) current price vs. baseline price
            # 3) potential future multiplier from the postTGE rewards policy

            size_base = {'small': 0.4, 'medium': 0.7, 'large': 0.9}.get(self.user_size, 0.5)

            # Price-based confidence factor
            if current_price is not None and baseline_price is not None:
                price_ratio = current_price / baseline_price
                # e.g. if ratio=0.8 => confidence=0.9
                confidence_factor = 1.0 + 0.5 * (price_ratio - 1.0)
            else:
                confidence_factor = 1.0

            # If the user’s expected reward is high, they might be more likely to stay
            if postTGE_rewards_policy is not None:
                # future multiplier if they remain active next period:
                user_future_multiplier = postTGE_rewards_policy.engagement_policy.calculate_multiplier(
                    self.active_days + 1
                )
                # scale that to something modest
                reward_incentive_factor = 1.0 + 0.2 * (user_future_multiplier - 1.0)
            else:
                reward_incentive_factor = 1.0

            prob_stay = size_base * confidence_factor * reward_incentive_factor
            # Clip to [0,1]
            prob_stay = max(0.0, min(1.0, prob_stay))

            self.active = (np.random.rand() < prob_stay)

            if self.active:
                self.active_days += 1
                # If your policy awards incremental tokens each month based on activity:
                if postTGE_rewards_policy is not None:
                    postTGE_rewards_policy.apply_rewards(self, self.active_days)

class SybilUser(User):
    """
    Represents a sybil user with lower interaction and immediate exit post-TGE.
    """
    def __init__(self, wealth, user_id, airdrop_policy=None):
        super().__init__(wealth, user_id, airdrop_policy)
        self.interaction_rate = np.random.poisson(lam=0.5)
        self.endowment = np.random.poisson(lam=0.5)
        self.decay_rate = 0.1 # Consider changing this for SybilUsers

    def step(self, phase, **kwargs):
        if phase == 'PreTGE':
            self.update_airdrop_points(dt=1)
        elif phase == 'TGE':
            self.tokens = self.airdrop_policy.calculate_tokens(self.airdrop_points, self)
        elif phase == 'PostTGE':
            # Sybils exit immediately
            self.active = False
