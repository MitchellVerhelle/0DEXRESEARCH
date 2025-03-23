import numpy as np
import matplotlib.pyplot as plt

from user_pool import UserPool
from postTGE_rewards import PostTGERewardsManager
from airdrop_policy import (
    LinearAirdropPolicy,
    ExponentialAirdropPolicy,
    TieredLinearAirdropPolicy,
    TieredConstantAirdropPolicy,
    TieredExponentialAirdropPolicy
)
from preTGE_rewards import (
    DydxRetroTieredRewardPolicy,
    VertexMakerTakerRewardPolicy,
    JupiterVolumeTierRewardPolicy,
    AevoBoostedVolumeRewardPolicy,
    HelixLoyaltyPointsRewardPolicy,
    GameLikeMMRRewardPolicy
)
from users import RegularUser, SybilUser

from plot_helper import (plot_price_evolution_grid, plot_airdrop_distribution_grid)


class MonteCarloSimulation:
    def __init__(self, num_users=1500000, total_supply=100_000_000, preTGE_steps=100, simulation_horizon=60,
                 airdrop_policy=None, preTGE_rewards_policy=None, airdrop_allocation_fraction=0.15):
        """
        Parameters:
          - num_users: Total number of simulated users.
          - total_supply: Total token supply.
          - preTGE_steps: Number of simulation steps in the pre-TGE phase.
          - simulation_horizon: Number of months to simulate post-TGE vesting.
          - airdrop_policy: An instance of an airdrop conversion policy (defaults to Linear).
          - preTGE_rewards_policy: An instance of a pre-TGE rewards policy to compute user airdrop points.
        """
        self.num_users = num_users
        self.total_supply = total_supply
        self.preTGE_steps = preTGE_steps
        self.simulation_horizon = simulation_horizon  # in months
        self.airdrop_policy = airdrop_policy if airdrop_policy is not None else LinearAirdropPolicy()
        self.preTGE_rewards_policy = preTGE_rewards_policy
        
        self.user_pool = UserPool(num_users=self.num_users, airdrop_policy=self.airdrop_policy)
        self.post_tge_manager = PostTGERewardsManager(total_supply=self.total_supply)

        self.airdrop_allocation_fraction = airdrop_allocation_fraction
    
    def simulate_preTGE(self):
        """
        Simulate the pre-TGE phase.
        If a preTGE_rewards_policy is provided, generate simulated activity stats for each user 
        (based on user size) and compute airdrop points via that policy.
        Otherwise, use the default incremental behavior.
        """
        for _ in range(self.preTGE_steps):
            self.user_pool.step_all('PreTGE')
        
        if self.preTGE_rewards_policy is not None:
            for user in self.user_pool.users:
                if hasattr(user, 'user_size'):
                    if user.user_size == 'small':
                        trading_volume = np.random.uniform(100, 1000)
                        maker_volume = np.random.uniform(0.3, 0.7) * trading_volume
                        taker_volume = trading_volume - maker_volume
                        stats = {
                            'trading_volume': trading_volume,
                            'maker_volume': maker_volume,
                            'taker_volume': taker_volume,
                            'qscore': np.random.uniform(50, 150),
                            'referral_points': np.random.uniform(0, 50)
                        }
                    elif user.user_size == 'medium':
                        trading_volume = np.random.uniform(1000, 10000)
                        maker_volume = np.random.uniform(0.3, 0.7) * trading_volume
                        taker_volume = trading_volume - maker_volume
                        stats = {
                            'trading_volume': trading_volume,
                            'maker_volume': maker_volume,
                            'taker_volume': taker_volume,
                            'qscore': np.random.uniform(100, 200),
                            'referral_points': np.random.uniform(0, 100)
                        }
                    elif user.user_size == 'large':
                        trading_volume = np.random.uniform(10000, 100000)
                        maker_volume = np.random.uniform(0.3, 0.7) * trading_volume
                        taker_volume = trading_volume - maker_volume
                        stats = {
                            'trading_volume': trading_volume,
                            'maker_volume': maker_volume,
                            'taker_volume': taker_volume,
                            'qscore': np.random.uniform(150, 300),
                            'referral_points': np.random.uniform(0, 150)
                        }
                    else:
                        trading_volume = np.random.uniform(1000, 5000)
                        stats = {'trading_volume': trading_volume}
                else:
                    trading_volume = np.random.uniform(50, 500)
                    stats = {'trading_volume': trading_volume}
                
                user.airdrop_points += self.preTGE_rewards_policy.calculate_points(stats, user)
        
        # Normalize to [0,1]
        max_points = max(u.airdrop_points for u in self.user_pool.users)
        if max_points > 0:
            for u in self.user_pool.users:
                u.airdrop_points /= max_points
    
    def simulate_TGE(self):
        """At TGE, convert accumulated airdrop points to tokens."""
        self.user_pool.step_all('TGE')
    
    def simulate_postTGE(self):
        """
        Simulate post-TGE vesting.
        Returns:
          - months: Array of months since TGE.
          - total_unlocked_history: List of total tokens unlocked over time.
          - unlocked_history: Dict mapping each allocation group to its unlocked tokens over time.
        """
        months = np.arange(0, self.simulation_horizon + 1)
        total_unlocked_history = []
        unlocked_history = {group: [] for group in self.post_tge_manager.schedules.keys()}
        for month in months:
            allocations = self.post_tge_manager.get_unlocked_allocations(month)
            total_unlocked = sum(allocations.values())
            total_unlocked_history.append(total_unlocked)
            for group, tokens in allocations.items():
                unlocked_history[group].append(tokens)
        return months, total_unlocked_history, unlocked_history

    def run(self):
        print("=== Running Pre-TGE Simulation ===")
        self.simulate_preTGE()
        print("Pre-TGE simulation complete.")

        print("=== Running TGE Simulation ===")
        self.simulate_TGE()
        print("TGE simulation complete.")
        
        # Scale the tokens so that the total equals airdrop_allocation_fraction of total_supply.
        raw_TGE_total = sum(user.tokens for user in self.user_pool.users)
        scaled_TGE_total = self.airdrop_allocation_fraction * self.total_supply

        if raw_TGE_total > 0:
            for user in self.user_pool.users:
                user.tokens = user.tokens * (scaled_TGE_total / raw_TGE_total)
        else:
            for user in self.user_pool.users:
                user.tokens = 0

        # Now recalc the total after scaling.
        scaled_total = sum(user.tokens for user in self.user_pool.users)
        print(f"TGE tokens assigned (scaled to {self.airdrop_allocation_fraction*100:.0f}%): {scaled_total:.2f}")

        # Compute distribution based on the scaled total.
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
    
        print("=== Running Post-TGE Simulation ===")
        months, total_unlocked_history, unlocked_history = self.simulate_postTGE()
        print("Post-TGE simulation complete.")
        
        return scaled_TGE_total, months, total_unlocked_history, unlocked_history, distribution

def compute_token_price(TGE_total, total_unlocked_history, users, distribution=None, 
                        base_price=10.0, elasticity=1.0, buyback_rate=0.2, alpha=0.5):
    """
    Computes token price over time based on a supply/demand model that incorporates both
    circulating supply and effective supply, with user behavior weighted by distribution.

    If a distribution dictionary is provided (with keys "small", "medium", "large", "sybil" 
    summing to 100), then compute average sell weight as:
        avg_sell_weight = (small_pct*1.0 + medium_pct*0.8 + large_pct*0.3 + sybil_pct*1.0) / 100
    Otherwise, compute the average sell weight from user objects.

    Then, define:
      circulating_supply(t) = TGE_total + (postTGE_unlocked(t) - postTGE_unlocked(0)) * avg_sell_weight
      effective_supply(t) = TGE_total + (postTGE_unlocked(t) - postTGE_unlocked(0)) * (avg_sell_weight*(1 - buyback_rate))
      combined_supply = alpha * circulating_supply + (1 - alpha) * effective_supply

    Price is:
      price(t) = base_price * (TGE_total / combined_supply(t))^elasticity
    """
    if distribution is not None:
        # Use distribution percentages to compute avg sell weight.
        avg_sell_weight = (
            distribution.get("small", 0) * 1.0 +
            distribution.get("medium", 0) * 0.8 +
            distribution.get("large", 0) * 0.3 +
            distribution.get("sybil", 0) * 1.0
        ) / 100.0
    else:
        def user_sell_weight(user):
            from users import RegularUser, SybilUser
            if isinstance(user, SybilUser):
                return 1.0
            if isinstance(user, RegularUser):
                if user.user_size == 'small':
                    return 1.0
                elif user.user_size == 'medium':
                    return 0.8
                elif user.user_size == 'large':
                    return 0.3
                else:
                    return 1.0
            return 1.0
        weights = [user_sell_weight(u) for u in users]
        avg_sell_weight = np.mean(weights)
    
    total_unlocked = np.array(total_unlocked_history)
    initial_additional = total_unlocked[0]
    
    circulating_supply = TGE_total + (total_unlocked - initial_additional) * avg_sell_weight
    effective_supply = TGE_total + (total_unlocked - initial_additional) * (avg_sell_weight * (1 - buyback_rate))
    effective_supply = np.maximum(effective_supply, 1)
    
    combined_supply = alpha * circulating_supply + (1 - alpha) * effective_supply
    prices = base_price * (TGE_total / combined_supply) ** elasticity
    return prices

if __name__ == '__main__':
    # Define lists of airdrop conversion policies and pre-TGE rewards policies.
    airdrop_policies = [
        ("Linear", LinearAirdropPolicy()),
        ("Exponential", ExponentialAirdropPolicy()),
        ("Tiered Linear", TieredLinearAirdropPolicy()),
        ("Tiered Constant", TieredConstantAirdropPolicy()),
        ("Tiered Exponential", TieredExponentialAirdropPolicy())
    ]
    
    preTGE_policies = [
        ("dYdX Retro", DydxRetroTieredRewardPolicy()),
        ("Vertex Maker/Taker", VertexMakerTakerRewardPolicy()),
        ("Jupiter Volume Tier", JupiterVolumeTierRewardPolicy()),
        ("Aevo Boosted Volume", AevoBoostedVolumeRewardPolicy()),
        ("Helix Loyalty", HelixLoyaltyPointsRewardPolicy()),
        ("Game-like MMR", GameLikeMMRRewardPolicy())
    ]
    
    # Simulation parameters.
    num_users = 1000
    total_supply = 100_000_000
    preTGE_steps = 50
    simulation_horizon = 60  # months
    base_price = 10.0
    elasticity = 1.0
    buyback_rate = 0.2  # 20% of additional unlocked tokens are removed.
    
    # Store results for each combination.
    results = {}
    
    for pre_name, pre_policy in preTGE_policies:
        for ad_name, ad_policy in airdrop_policies:
            combo_name = f"{pre_name} + {ad_name}"
            print(f"\nRunning simulation for: {combo_name}")
            sim = MonteCarloSimulation(
                num_users=num_users,
                total_supply=total_supply,
                preTGE_steps=preTGE_steps,
                simulation_horizon=simulation_horizon,
                airdrop_policy=ad_policy,
                preTGE_rewards_policy=pre_policy
            )
            TGE_total, months, total_unlocked_history, unlocked_history, dist = sim.run()
            prices = compute_token_price(
                TGE_total, total_unlocked_history, sim.user_pool.users, 
                distribution=dist, base_price=base_price, elasticity=elasticity, 
                buyback_rate=buyback_rate, alpha=0.5
            )
            results[combo_name] = {
                "TGE_total": TGE_total,
                "months": months,
                "total_unlocked_history": total_unlocked_history,
                "unlocked_history": unlocked_history,
                "prices": prices,
                "TGE_tokens": [user.tokens for user in sim.user_pool.users],
                "distribution": dist
            }
    # Plot histograms of tokens assigned at TGE for each combination.
    plt.figure(figsize=(12, 8))
    for combo_name, data in results.items():
        tokens = np.array(data["TGE_tokens"])
        # Filter out non-finite values (e.g., inf or NaN).
        finite_tokens = tokens[np.isfinite(tokens)]
        if finite_tokens.size > 0:
            plt.hist(finite_tokens, bins=30, alpha=0.5, label=combo_name,
                    histtype='step', linewidth=1.5)
    plt.xlabel("Tokens Assigned at TGE")
    plt.ylabel("Number of Users")
    plt.title("Histogram of TGE Tokens Distribution for All Combinations")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True)
    plt.show(block=True)
    
    # For one chosen combination (e.g., "dYdX Retro + Linear"), plot vesting graphs.
    chosen = "dYdX Retro + Linear"
    chosen_result = results[chosen]
    
    # Plot total unlocked tokens over time.
    plt.figure(figsize=(10, 6))
    plt.plot(chosen_result["months"], chosen_result["total_unlocked_history"],
             label=f"Total Unlocked Tokens ({chosen})", color='blue')
    plt.xlabel("Months since TGE")
    plt.ylabel("Total Unlocked Tokens")
    plt.title("Post-TGE Vesting: Total Unlocked Tokens Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot stacked area chart for unlocked tokens per group.
    plt.figure(figsize=(10, 6))
    groups = list(chosen_result["unlocked_history"].keys())
    data_stack = np.vstack([chosen_result["unlocked_history"][group] for group in groups])
    plt.stackplot(chosen_result["months"], data_stack, labels=groups)
    plt.xlabel("Months since TGE")
    plt.ylabel("Tokens Unlocked")
    plt.title("Post-TGE Vesting: Unlocked Tokens per Group Over Time")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    # Build a heatmap of final token prices (month 60) for each combination.
    pre_labels = [p[0] for p in preTGE_policies]
    ad_labels = [a[0] for a in airdrop_policies]
    heatmap = np.zeros((len(pre_labels), len(ad_labels)))
    for i, pre_name in enumerate(pre_labels):
        for j, ad_name in enumerate(ad_labels):
            combo_name = f"{pre_name} + {ad_name}"
            final_price = results[combo_name]["prices"][-1]
            heatmap[i, j] = final_price
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="viridis", aspect="auto")
    plt.colorbar(label="Final Token Price (USD)")
    plt.xticks(ticks=np.arange(len(ad_labels)), labels=ad_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(pre_labels)), labels=pre_labels)
    plt.title("Heatmap: Final Token Price (Month 60) for Each Combination")
    for i in range(len(pre_labels)):
        for j in range(len(ad_labels)):
            plt.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="w")
    plt.tight_layout()
    plt.show()
    
    # Plot a grid of token price evolution curves (excluding t=0) for each combination.
    plot_price_evolution_grid(results, pre_labels, ad_labels, max_rows_per_fig=6)

    # Plot where airdrop went
    pre_labels = [p[0] for p in preTGE_policies]
    ad_labels = [a[0] for a in airdrop_policies]
    plot_airdrop_distribution_grid(results, pre_labels, ad_labels)