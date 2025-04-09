import numpy as np

class PostTGERewardsSimulator:
    """
    Simulator for post-TGE token price evolution using a supply/demand model
    combined with a jump-diffusion process.
    """
    def __init__(self, TGE_total, total_unlocked_history, users, base_price=10.0, elasticity=1.0,
                 buyback_rate=0.2, alpha=0.5, sigma=0.05, jump_intensity=0.1,
                 jump_mean=-0.05, jump_std=0.1, distribution=None, demand_series=None):
        self.TGE_total = TGE_total
        self.total_unlocked_history = np.array(total_unlocked_history)
        self.users = users
        self.base_price = base_price
        self.elasticity = elasticity
        self.buyback_rate = buyback_rate
        self.alpha = alpha
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.distribution = distribution
        self.demand_series = demand_series

    def compute_token_price(self):
        total_unlocked = self.total_unlocked_history
        initial_unlocked = total_unlocked[0]
        
        if self.distribution is not None:
            avg_sell_weight = (
                self.distribution.get("small", 0) * 1.0 +
                self.distribution.get("medium", 0) * 0.8 +
                self.distribution.get("large", 0) * 0.3 +
                self.distribution.get("sybil", 0) * 1.0
            ) / 100.0
        else:
            avg_sell_weight = 1.0
        
        circulating_supply = self.TGE_total + (total_unlocked - initial_unlocked) * avg_sell_weight
        effective_supply = self.TGE_total + (total_unlocked - initial_unlocked) * (avg_sell_weight * (1 - self.buyback_rate))
        effective_supply = np.maximum(effective_supply, 1)
        combined_supply = self.alpha * circulating_supply + (1 - self.alpha) * effective_supply
        
        baseline_prices = self.base_price * (self.TGE_total / combined_supply) ** self.elasticity
        return baseline_prices

    def simulate_price_evolution(self, dt=1):
        """
        Simulate dynamic token price evolution with drift affected by both external demand
        and effective user activity. Effective user activity boosts the drift if active users (weighted
        by their accumulated activity) hold more tokens.
        """
        n = len(self.total_unlocked_history)
        baseline_prices = self.compute_token_price()
        P_jump = np.ones(n)
        
        # Process demand driver.
        if self.demand_series is not None:
            demand_values = np.array(self.demand_series, dtype=float)
            max_demand = demand_values.max()
            normalized_demand = demand_values / max_demand
            if len(normalized_demand) < n:
                pad_length = n - len(normalized_demand)
                normalized_demand = np.pad(normalized_demand, (0, pad_length), mode='constant', 
                                           constant_values=normalized_demand[-1])
            else:
                normalized_demand = normalized_demand[:n]
        else:
            normalized_demand = np.ones(n) * 0.5
        
        # Drift parameters.
        base_mu = 0.0
        k = 0.03                 # External demand sensitivity.
        reference = 0.5
        k_activity = 0.1         # Sensitivity to effective user activity.
        ref_activity = 0.5       # Reference effective active token fraction.
        drift_min = -0.1
        drift_max = 0.5
        
        # Helper: effective token value based on activity.
        def effective_tokens(user):
            # Increase a user's influence by a factor based on active_days.
            return user.tokens * (1 + 0.1 * user.active_days)
        
        for t in range(1, n):
            # Compute drift from external demand.
            drift_t = base_mu + k * (normalized_demand[t] - reference)
            log_noise = np.random.lognormal(mean=0, sigma=0.01) - 1.0
            drift_t += log_noise
            
            # Compute effective active fraction.
            total_eff = sum(effective_tokens(user) for user in self.users)
            active_eff = sum(effective_tokens(user) for user in self.users if user.active)
            weighted_active_fraction = (active_eff / total_eff) if total_eff > 0 else ref_activity
            
            # Add drift from user activity.
            drift_t += k_activity * (weighted_active_fraction - ref_activity)
            drift_t = np.clip(drift_t, drift_min, drift_max)
            
            diffusion = np.exp((drift_t - 0.5 * self.sigma**2) * dt +
                               self.sigma * np.sqrt(dt) * np.random.randn())
            if np.random.rand() < self.jump_intensity * dt:
                jump = 1.0 + np.random.normal(self.jump_mean, self.jump_std)
            else:
                jump = 1.0
            P_jump[t] = P_jump[t-1] * diffusion * jump
        prices = baseline_prices * P_jump
        return prices