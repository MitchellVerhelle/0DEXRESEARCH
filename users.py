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

    @abstractmethod
    def step(self, phase):
        """
        Advance the user's state based on the current simulation phase.
        Phases can be: 'PreTGE', 'TGE', 'PostTGE', etc.
        """
        pass

class RegularUser(User):
    """
    Represents a non-sybil user with varying interaction rates based on user size.
    """
    def __init__(self, wealth, user_id, user_size, airdrop_policy=None):
        super().__init__(wealth, user_id, airdrop_policy)
        self.user_size = user_size

        # Set interaction rate based on user size
        if user_size == 'small':
            self.interaction_rate = np.random.poisson(lam=1)
        elif user_size == 'medium':
            self.interaction_rate = np.random.poisson(lam=3)
        elif user_size == 'large':
            self.interaction_rate = np.random.poisson(lam=5)
        else:
            self.interaction_rate = 1

    def step(self, phase):
        if phase == 'PreTGE':
            # Simulate "farming" for airdrop points
            delta = self.interaction_rate * np.random.uniform(0.5, 1.5)
            self.airdrop_points += delta
        elif phase == 'TGE':
            # Assign tokens based on accumulated airdrop points
            self.tokens = self.airdrop_policy.calculate_tokens(self.airdrop_points, self)
        elif phase == 'PostTGE':
            # Decide if user remains active based on size
            if self.user_size == 'small':
                prob_stay = 0.4
            elif self.user_size == 'medium':
                prob_stay = 0.8
            elif self.user_size == 'large':
                prob_stay = 0.9
            else:
                prob_stay = 0.5
            self.active = (np.random.rand() < prob_stay)

class SybilUser(User):
    """
    Represents a sybil user with lower interaction and immediate exit post-TGE.
    """
    def __init__(self, wealth, user_id, airdrop_policy=None):
        super().__init__(wealth, user_id, airdrop_policy)
        self.interaction_rate = np.random.poisson(lam=0.5)

    def step(self, phase):
        if phase == 'PreTGE':
            delta = self.interaction_rate * np.random.uniform(0.5, 1.0)
            self.airdrop_points += delta
        elif phase == 'TGE':
            self.tokens = self.airdrop_policy.calculate_tokens(self.airdrop_points, self)
        elif phase == 'PostTGE':
            # Sybils exit immediately
            self.active = False
