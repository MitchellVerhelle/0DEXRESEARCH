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
                 initial_price=10.0, buyback_rate=0.2, elasticity=0.5, demand_series=None):
        """
        Parameters:
          - demand_series: Array-like sequence of raw demand values that will drive drift.
        """
        self.num_users = num_users
        self.total_supply = total_supply
        self.preTGE_steps = preTGE_steps
        self.simulation_horizon = simulation_horizon  # in months
        self.airdrop_policy = airdrop_policy if airdrop_policy is not None else LinearAirdropPolicy()
        self.preTGE_rewards_policy = preTGE_rewards_policy if preTGE_rewards_policy is not None else GenericPreTGERewardPolicy()
        self.postTGE_rewards_policy = postTGE_rewards_policy if postTGE_rewards_policy is not None else GenericPostTGERewardPolicy()
        self.initial_price = initial_price
        self.buyback_rate = buyback_rate
        self.elasticity = elasticity

        self.user_pool = UserPool(num_users=self.num_users, airdrop_policy=self.airdrop_policy)
        self.post_tge_manager = PostTGERewardsManager(total_supply=self.total_supply)
        self.airdrop_allocation_fraction = airdrop_allocation_fraction
        self.demand_series = demand_series

    def simulate_preTGE(self):
        for _ in range(self.preTGE_steps):
            self.user_pool.step_all('PreTGE')
        
        if self.preTGE_rewards_policy is not None:
            for user in self.user_pool.users:
                stats = generate_stats(user)
                user.airdrop_points += self.preTGE_rewards_policy.calculate_points(stats, user)
        
        max_points = max(u.airdrop_points for u in self.user_pool.users) or 1
        for u in self.user_pool.users:
            u.airdrop_points /= max_points

    def simulate_TGE(self):
        self.user_pool.step_all('TGE')
    
    def simulate_postTGE(self):
        """
        Simulate the post-TGE phase by, for each time step:
          - Computing a vesting-based baseline price from unlocked allocations,
          - Computing drift from external demand and from effective user participation,
          - Computing a one-period multiplier using a jump-diffusion process,
          - Final price = baseline * multiplier.
        
        The effective user participation takes into account both each user’s tokens and
        their original endowment. In particular, we define:
        
          effective_weight = tokens × (1 + 0.1 × active_days) + β × endowment,
        
        with β=1.0 (by default). Then the weighted active fraction is computed over all users
        and is used to boost (or depress) drift.
        """
        months = np.arange(0, self.simulation_horizon + 1)
        num_steps = len(months)
        
        # Normalize the demand series.
        if self.demand_series is not None:
            demand_values = np.array(self.demand_series, dtype=float)
            max_demand = demand_values.max()
            normalized_demand = demand_values / max_demand
            if len(normalized_demand) < num_steps:
                pad_length = num_steps - len(normalized_demand)
                normalized_demand = np.pad(normalized_demand, (0, pad_length),
                                           mode='constant', constant_values=normalized_demand[-1])
            else:
                normalized_demand = normalized_demand[:num_steps]
        else:
            normalized_demand = np.ones(num_steps) * 0.5
        
        total_unlocked_history = []
        unlocked_history = {group: [] for group in self.post_tge_manager.schedules.keys()}

        # Precompute baseline constants.
        TGE_total = self.airdrop_allocation_fraction * self.total_supply
        allocations0 = self.post_tge_manager.get_unlocked_allocations(0)
        unlocked_at_TGE_total = sum(allocations0.values())
        
        # Parameters for drift and diffusion.
        base_mu = 0.0
        k = 0.5                  # Sensitivity to external (normalized) demand.
        reference = 0.5
        k_activity = 0.3         # Sensitivity to effective user activity.
        ref_activity = 0.5       # Reference effective active fraction.
        drift_min = -1.0
        drift_max = 1.0
        sigma = 0.2
        jump_intensity = 0.3
        jump_mean = -0.1
        jump_std = 0.15
        dt = 1

        # Parameter for endowment influence.
        beta = 1.0

        final_prices = np.zeros(num_steps)
        active_fraction_history = np.zeros(num_steps)

        # Time step 0: compute baseline price.
        alloc = self.post_tge_manager.get_unlocked_allocations(0)
        total_unlocked = sum(alloc.values())
        total_unlocked_history.append(total_unlocked)
        for group, tokens in alloc.items():
            unlocked_history[group].append(tokens)
        circulating_supply = TGE_total + (total_unlocked - unlocked_at_TGE_total)
        effective_supply = TGE_total + (total_unlocked - unlocked_at_TGE_total) * (1 - self.buyback_rate)
        combined_supply = 0.5 * (circulating_supply + effective_supply)
        baseline = self.initial_price * (TGE_total / combined_supply) ** self.elasticity
        final_prices[0] = baseline
        active_fraction_history[0] = 0.1

        # For time steps 1...T.
        for t in range(1, num_steps):
            # Compute the vesting-based baseline price for time t.
            alloc = self.post_tge_manager.get_unlocked_allocations(t)
            tot_unlocked = sum(alloc.values())
            total_unlocked_history.append(tot_unlocked)
            for group, tokens in alloc.items():
                unlocked_history[group].append(tokens)
            circulating_supply = TGE_total + (tot_unlocked - unlocked_at_TGE_total)
            effective_supply = TGE_total + (tot_unlocked - unlocked_at_TGE_total) * (1 - self.buyback_rate)
            combined_supply = 0.5 * (circulating_supply + effective_supply)
            baseline = self.initial_price * (TGE_total / combined_supply) ** self.elasticity

            # Compute drift from external demand.
            drift = base_mu + k * (normalized_demand[t] - reference)
            log_noise = np.random.lognormal(mean=0, sigma=0.01) - 1.0
            drift += log_noise

            # Compute effective user weight across the population.
            total_eff = 0.0
            active_eff = 0.0
            for user in self.user_pool.users:
                # Now incorporating both tokens and endowment.
                eff = user.tokens * (1 + 0.1 * user.active_days) + beta * user.endowment
                total_eff += eff
                if user.active:
                    active_eff += eff
            weighted_active_fraction = (active_eff / total_eff) if total_eff > 0 else ref_activity
            drift += k_activity * (weighted_active_fraction - ref_activity)
            drift = np.clip(drift, drift_min, drift_max)

            # Compute the one-period multiplier from jump-diffusion.
            multiplier = np.exp((drift - 0.5 * sigma**2) * dt +
                                  sigma * np.sqrt(dt) * np.random.randn())
            if np.random.rand() < jump_intensity * dt:
                multiplier *= (1.0 + np.random.normal(jump_mean, jump_std))
            final_prices[t] = baseline * multiplier

            # Update user state.
            for user in self.user_pool.users:
                user.step(
                    phase='PostTGE',
                    current_price=final_prices[t],
                    baseline_price=baseline,
                    postTGE_rewards_policy=self.postTGE_rewards_policy
                )
            active_users = sum(1 for user in self.user_pool.users if user.active)
            active_fraction_history[t] = active_users / len(self.user_pool.users)
            
        return {
            "months": months,
            "dynamic_prices": final_prices,
            "active_fraction_history": active_fraction_history,
            "total_unlocked_history": total_unlocked_history,
            "unlocked_history": unlocked_history
        }

    def run(self):
        print("=== Running Pre-TGE Simulation ===")
        self.simulate_preTGE()
        print("Pre-TGE simulation complete.")

        print("=== Running TGE Simulation ===")
        self.simulate_TGE()
        print("TGE simulation complete.")

        raw_TGE_total = sum(user.tokens for user in self.user_pool.users)
        scaled_TGE_total = self.airdrop_allocation_fraction * self.total_supply
        if raw_TGE_total > 0:
            for user in self.user_pool.users:
                user.tokens *= (scaled_TGE_total / raw_TGE_total)
        else:
            for user in self.user_pool.users:
                user.tokens = 0
        print(f"TGE tokens assigned (scaled to {self.airdrop_allocation_fraction*100:.0f}%): {scaled_TGE_total:.2f}")

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
        if scaled_TGE_total > 0:
            for k in distribution:
                distribution[k] = (distribution[k] / scaled_TGE_total) * 100.0

        print("=== Running Post-TGE Simulation (Dynamic Price Evolution) ===")
        postTGE_results = self.simulate_postTGE()
        print("Post-TGE simulation complete.")

        results = {
            "scaled_TGE_total": scaled_TGE_total,
            "months": postTGE_results["months"],
            "dynamic_prices": postTGE_results["dynamic_prices"],
            "active_fraction_history": postTGE_results["active_fraction_history"],
            "total_unlocked_history": postTGE_results["total_unlocked_history"],
            "unlocked_history": postTGE_results["unlocked_history"],
            "distribution": distribution
        }
        return results

if __name__ == '__main__':
    # For testing purposes.
    demand_values = np.array([
        45, 25, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 5, 15, 5,
        12, 1, 43, 5, 11, 1,
        66, 2, 3, 1, 77, 3, 3, 3,
        69, 3, 1, 3, 65, 2, 1, 2, 47,
        2, 4, 2, 38, 2, 1, 2, 37, 3,
        2, 2, 22, 3, 1, 2, 18, 1, 2,
        1, 21, 2, 1, 3
    ], dtype=float)
    sim = MonteCarloSimulation(
        num_users=10000,
        total_supply=100_000_000,
        preTGE_steps=50,
        simulation_horizon=60,
        airdrop_policy=LinearAirdropPolicy(),
        preTGE_rewards_policy=GenericPreTGERewardPolicy(),
        postTGE_rewards_policy=GenericPostTGERewardPolicy(),
        airdrop_allocation_fraction=0.15,
        initial_price=10.0,
        demand_series=demand_values
    )
    results = sim.run()
    print("Simulation finished.")
