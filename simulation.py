import numpy as np
from user_pool import UserPool
from vesting import PostTGERewardsManager
from preTGE_rewards import GenericPreTGERewardPolicy
from postTGE_rewards_policy import GenericPostTGERewardPolicy
from airdrop_policy import LinearAirdropPolicy
from users import RegularUser, SybilUser
from activity_stats import generate_stats

class MonteCarloSimulation:
    def __init__(self, num_users=1500000, total_supply=100_000_000, preTGE_steps=100, simulation_horizon=60,
                 airdrop_policy=None, preTGE_rewards_policy=None, postTGE_rewards_policy=None, airdrop_allocation_fraction=0.15,
                 initial_price=10.0):
        """
        Parameters:
          - num_users: Total number of simulated users.
          - total_supply: Total token supply.
          - preTGE_steps: Number of simulation steps in the pre-TGE phase.
          - simulation_horizon: Number of months to simulate post-TGE.
          - airdrop_policy: An airdrop conversion policy (default: LinearAirdropPolicy).
          - preTGE_rewards_policy: A pre-TGE rewards policy (default: GenericPreTGERewardPolicy).
          - airdrop_allocation_fraction: Fraction of total supply allocated for the airdrop.
          - initial_price: Projected initial token price.
        """
        self.num_users = num_users
        self.total_supply = total_supply
        self.preTGE_steps = preTGE_steps
        self.simulation_horizon = simulation_horizon  # in months
        self.airdrop_policy = airdrop_policy if airdrop_policy is not None else LinearAirdropPolicy()
        self.preTGE_rewards_policy = preTGE_rewards_policy if preTGE_rewards_policy is not None else GenericPreTGERewardPolicy()
        self.postTGE_rewards_policy = postTGE_rewards_policy if postTGE_rewards_policy is not None else GenericPostTGERewardPolicy()
        self.initial_price = initial_price
        
        self.user_pool = UserPool(num_users=self.num_users, airdrop_policy=self.airdrop_policy)
        self.post_tge_manager = PostTGERewardsManager(total_supply=self.total_supply)
        self.airdrop_allocation_fraction = airdrop_allocation_fraction
    
    def simulate_preTGE(self):
        """
        Simulate the pre-TGE phase.
        Each step, users perform their 'PreTGE' behavior.
        Then, activity stats are generated (via generate_stats) and used by the preTGE rewards policy
        to accumulate airdrop points.
        """
        for _ in range(self.preTGE_steps):
            self.user_pool.step_all('PreTGE')
        
        if self.preTGE_rewards_policy is not None:
            for user in self.user_pool.users:
                stats = generate_stats(user)
                user.airdrop_points += self.preTGE_rewards_policy.calculate_points(stats, user)
        
        # Normalize airdrop points to [0,1]
        max_points = max(u.airdrop_points for u in self.user_pool.users) or 1
        for u in self.user_pool.users:
            u.airdrop_points /= max_points

    def simulate_TGE(self):
        """At TGE, convert accumulated airdrop points to tokens."""
        self.user_pool.step_all('TGE')
    
    def simulate_postTGE(self):
        """
        Simulate the post-TGE phase in a stepwise manner.
        
        For each month:
          1. Compute vesting (unlocked tokens) via PostTGERewardsManager.
          2. Calculate a baseline price using a supply/demand model.
          3. Update the dynamic multiplier via a jump-diffusion step.
          4. Update the dynamic price = baseline price * dynamic multiplier.
          5. Update user retention (active fraction) based on the current dynamic price.
          6. Let users perform their 'PostTGE' behavior.
          7. The updated active fraction feeds back into the next time step.
        """
        months = np.arange(0, self.simulation_horizon + 1)
        total_unlocked_history = []
        unlocked_history = {group: [] for group in self.post_tge_manager.schedules.keys()}
        baseline_prices = []

        # Compute vesting (baseline supply) at each month.
        for month in months:
            allocations = self.post_tge_manager.get_unlocked_allocations(month)
            total_unlocked = sum(allocations.values())
            total_unlocked_history.append(total_unlocked)
            for group, tokens in allocations.items():
                unlocked_history[group].append(tokens)
            # Compute baseline price using a simplified supply/demand model:
            # Assume: circulating_supply = TGE_total + (total_unlocked - unlocked_at_TGE)
            # effective_supply = TGE_total + (total_unlocked - unlocked_at_TGE) * (1 - buyback_rate)
            # combined_supply = 0.5 * (circulating_supply + effective_supply)
            TGE_total = self.airdrop_allocation_fraction * self.total_supply
            if month == 0:
                unlocked_at_TGE = total_unlocked
            circulating_supply = TGE_total + (total_unlocked - unlocked_at_TGE)
            effective_supply = TGE_total + (total_unlocked - unlocked_at_TGE) * (1 - 0.2)  # using buyback_rate=0.2
            combined_supply = 0.5 * (circulating_supply + effective_supply)
            baseline = self.initial_price * (TGE_total / combined_supply) ** 1.0
            baseline_prices.append(baseline)
        
        # Now iterate month-by-month updating dynamic price and user retention.
        T = self.simulation_horizon
        dt = 1
        dynamic_prices = np.zeros(T+1)
        dynamic_prices[0] = baseline_prices[0]
        dynamic_multiplier = np.ones(T+1)
        active_fraction_history = np.zeros(T+1)
        active_fraction_history[0] = 0.1  # initial active fraction
        p0 = self.initial_price
        # Parameters for jump-diffusion:
        mu = 0.0
        sigma = 0.05
        jump_intensity = 0.1
        jump_mean = -0.05
        jump_std = 0.1
        # Parameters for retention dynamics:
        alpha_rate = 0.1
        beta_rate = 0.01
        theta = 0.85
        sigma_de = 0.01

        for t in range(T):
            # Update dynamic multiplier via jump-diffusion.
            diffusion = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn())
            jump = 1.0
            if np.random.rand() < jump_intensity * dt:
                jump = 1.0 + np.random.normal(jump_mean, jump_std)
            dynamic_multiplier[t+1] = dynamic_multiplier[t] * diffusion * jump
            dynamic_prices[t+1] = baseline_prices[t+1] * dynamic_multiplier[t+1]

            # Update active fraction using Euler–Maruyama.
            A = active_fraction_history[t]
            p_t = dynamic_prices[t]
            dA_dt = alpha_rate * (1 - A) - beta_rate * (1 + theta * ((p_t - p0) / p0)) * A
            noise = np.random.normal(0, sigma_de)
            A_new = A + (dA_dt + noise) * dt
            A_new = max(0.0, min(1.0, A_new))
            active_fraction_history[t+1] = A_new

            # Update user activity for this time step.
            for user in self.user_pool.users:
                user.step(
                    'PostTGE',
                    current_price=dynamic_prices[t+1],
                    baseline_price=baseline_prices[t+1],
                    postTGE_rewards_policy=self.postTGE_rewards_policy
                )
        
        return {
            "months": months,
            "total_unlocked_history": total_unlocked_history,
            "unlocked_history": unlocked_history,
            "dynamic_prices": dynamic_prices,
            "active_fraction_history": active_fraction_history
        }

    def run(self):
        print("=== Running Pre-TGE Simulation ===")
        self.simulate_preTGE()
        print("Pre-TGE simulation complete.")

        print("=== Running TGE Simulation ===")
        self.simulate_TGE()
        print("TGE simulation complete.")

        # Scale TGE tokens to match airdrop_allocation_fraction of total_supply.
        raw_TGE_total = sum(user.tokens for user in self.user_pool.users)
        scaled_TGE_total = self.airdrop_allocation_fraction * self.total_supply
        if raw_TGE_total > 0:
            for user in self.user_pool.users:
                user.tokens = user.tokens * (scaled_TGE_total / raw_TGE_total)
        else:
            for user in self.user_pool.users:
                user.tokens = 0
        scaled_total = sum(user.tokens for user in self.user_pool.users)
        print(f"TGE tokens assigned (scaled to {self.airdrop_allocation_fraction*100:.0f}%): {scaled_total:.2f}")

        distribution = {"small": 0.0, "medium": 0.0, "large": 0.0, "sybil": 0.0}
        for u in self.user_pool.users:
            if isinstance(u, SybilUser):
                distribution["sybil"] += u.tokens
            elif isinstance(u, RegularUser):
                if u.user_size == 'small':
                    distribution["small"] += u.tokens
                elif u.user_size == 'medium':
                    distribution["medium"] += u.tokens
                elif u.user_size == 'large':
                    distribution["large"] += u.tokens
        if scaled_total > 0:
            for k in distribution:
                distribution[k] = (distribution[k] / scaled_total) * 100.0

        print("Distribution by user type (percent):", distribution)

        print("=== Running Post-TGE Simulation (Dynamic Phase) ===")
        postTGE_results = self.simulate_postTGE()
        print("Post-TGE simulation complete.")

        return {
            "scaled_TGE_total": scaled_TGE_total,
            **postTGE_results,
            "distribution": distribution
        }

if __name__ == '__main__':
    sim = MonteCarloSimulation(
        num_users=10000,
        total_supply=100_000_000,
        preTGE_steps=50,
        simulation_horizon=60,
        airdrop_policy=LinearAirdropPolicy(),
        preTGE_rewards_policy=GenericPreTGERewardPolicy(),
        postTGE_rewards_policy=GenericPostTGERewardPolicy(),
        airdrop_allocation_fraction=0.15,
        initial_price=10.0
    )
    results = sim.run()
    print("Simulation finished.")
