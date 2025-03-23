import numpy as np

from user_pool import UserPool
from postTGE_rewards import PostTGERewardsManager
from airdrop_policy import LinearAirdropPolicy
from users import RegularUser, SybilUser


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

def compute_token_price(TGE_total, total_unlocked_history, users, 
                        base_price=10.0, elasticity=1.0, buyback_rate=0.2, alpha=0.5,
                        distribution=None):
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

def simulate_price_evolution_dynamic(TGE_total, total_unlocked_history, users, base_price=10.0, 
                                     elasticity=1.0, buyback_rate=0.2, alpha=0.5, 
                                     mu=0.0, sigma=0.05, jump_intensity=0.1, jump_mean=-0.05, jump_std=0.1, distribution=None):
        """
        Simulates the dynamic evolution of the token price by combining a supply/demand model 
        (based on the fixed TGE allocation and vesting schedule) with a jump-diffusion process 
        that represents volatility shocks and price jumps.
        
        Parameters:
        TGE_total: The (scaled) TGE token total (a fixed amount, e.g. 15% of total supply).
        total_unlocked_history: Array or list of total unlocked tokens over time (from vesting).
        users: List of user objects (used to compute average sell weight, if needed).
        base_price: The base price at TGE.
        elasticity: Price elasticity parameter.
        buyback_rate: Fraction of unlocked tokens that are effectively removed (via buybacks/burns).
        alpha: Weighting parameter between raw circulating supply and effective supply.
        mu: Drift rate for the diffusion component (per time unit, e.g. per month).
        sigma: Volatility for the diffusion component.
        jump_intensity: Probability per time step of a jump event.
        jump_mean: Mean percentage jump (e.g. -0.05 means an average 5% drop when a jump occurs).
        jump_std: Standard deviation of the jump size.
        
        Returns:
        prices: Array of simulated prices over time.
        
        The function first computes a baseline supply/demand price using your existing logic,
        then simulates a jump-diffusion multiplier over time, and finally returns the product.
        """
        # 1. Compute the baseline supply/demand price (without dynamic evolution).
        supply_price = compute_token_price(TGE_total, total_unlocked_history, users, base_price, elasticity, buyback_rate, alpha, distribution)
        # supply_price is an array, one value per time step.
        
        # 2. Simulate a jump-diffusion process for dynamic volatility.
        n = len(total_unlocked_history)
        dt = 1  # assume one time unit per step (e.g. 1 month)
        P_jump = np.zeros(n)
        P_jump[0] = 1.0  # start with no multiplier at TGE
        for t in range(1, n):
            # Continuous diffusion component (geometric Brownian motion).
            diffusion = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn())
            # Jump component: with probability jump_intensity * dt, apply a jump.
            if np.random.rand() < jump_intensity * dt:
                jump = 1.0 + np.random.normal(jump_mean, jump_std)
            else:
                jump = 1.0
            P_jump[t] = P_jump[t-1] * diffusion * jump
        
        # 3. Combine the baseline supply price with the dynamic multiplier.
        prices = supply_price * P_jump
        return prices