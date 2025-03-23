import numpy as np
from airdrop_policy import AirdropPolicy
from users import RegularUser, SybilUser

class UserPool:
    """
    Generates and manages a collection of users (both regular and sybil).
    """
    def __init__(self, num_users, airdrop_policy=None):
        self.num_users = num_users
        self.airdrop_policy = airdrop_policy if airdrop_policy is not None else AirdropPolicy()
        self.users = []
        self.generate_users()

    def generate_users(self):
        sybil_percentage = 0.3
        num_sybil = int(self.num_users * sybil_percentage)
        num_regular = self.num_users - num_sybil

        # Distribute regular user sizes
        small_percentage = 0.6
        medium_percentage = 0.3
        large_percentage = 0.1

        num_small = int(num_regular * small_percentage)
        num_medium = int(num_regular * medium_percentage)
        num_large = num_regular - num_small - num_medium

        user_id = 0

        # Generate wealth for regular users
        wealth_small = np.random.lognormal(mean=6, sigma=1.5, size=num_small)
        wealth_medium = np.random.lognormal(mean=7, sigma=1.2, size=num_medium)
        wealth_large = np.random.lognormal(mean=8, sigma=1.0, size=num_large)

        for w in wealth_small:
            self.users.append(RegularUser(w, user_id, 'small', self.airdrop_policy))
            user_id += 1
        for w in wealth_medium:
            self.users.append(RegularUser(w, user_id, 'medium', self.airdrop_policy))
            user_id += 1
        for w in wealth_large:
            self.users.append(RegularUser(w, user_id, 'large', self.airdrop_policy))
            user_id += 1

        # Generate wealth for sybil users
        wealth_sybil = np.random.lognormal(mean=5, sigma=1.0, size=num_sybil)
        for w in wealth_sybil:
            self.users.append(SybilUser(w, user_id, self.airdrop_policy))
            user_id += 1

        # Shuffle the list so user types are interspersed
        np.random.shuffle(self.users)

    def step_all(self, phase):
        for user in self.users:
            user.step(phase)

    def get_active_users(self):
        return [user for user in self.users if user.active]
