import numpy as np

class PostTGERewardsSimulator:
    """
    Simulator for post-TGE token price evolution using a supply/demand model
    combined with a jump-diffusion process.

    The baseline price at time t is given by:
      price(t) = base_price * (TGE_total / combined_supply(t))**elasticity

    where:
      - TGE_total: The token allocation at TGE (e.g., airdrop allocation).
      - combined_supply(t) is a weighted average of:
          circulating_supply(t) = TGE_total + (postTGE_unlocked(t) - postTGE_unlocked(0)) * avg_sell_weight,
          effective_supply(t) = TGE_total + (postTGE_unlocked(t) - postTGE_unlocked(0)) * (avg_sell_weight*(1 - buyback_rate))
          with combined_supply = alpha * circulating_supply + (1 - alpha) * effective_supply.
    
    Then a dynamic multiplier is applied via a jump-diffusion process:
      - Diffusion (GBM component):
          multiplier = exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  where Z ~ N(0,1)
      - Jump component:
          With probability (jump_intensity * dt), a jump occurs:
            jump = 1 + N(jump_mean, jump_std)
          Otherwise, jump = 1.

    Final price: price(t) = baseline_price(t) * cumulative_multiplier(t)

    Parameter estimates:
      - base_price: 10.0 (reference token price at TGE) [Source: Vertex docs & common practice]
      - elasticity: 1.0 [Assumed; see economic models in crypto]
      - buyback_rate: 0.2 (20% removal of unlocked tokens) [Assumed]
      - alpha: 0.5 (equal weighting) [Assumed]
      - mu: 0.0 (neutral drift)
      - sigma: 0.05 (monthly volatility, e.g., from Investopedia's GBM discussions)
      - jump_intensity: 0.1 per month [Assumed based on empirical data]
      - jump_mean: -0.05 (average 5% drop when jump occurs)
      - jump_std: 0.1 [Assumed variability]
      - distribution: A dict with keys "small", "medium", "large", "sybil" that sum to 100,
                      representing the user token distribution percentages.
                      (Used to compute avg_sell_weight)

    Sources:
      - Geometric Brownian Motion: https://www.investopedia.com/terms/g/geometric-brownian-motion.asp
      - Jump-Diffusion Models: https://www.sciencedirect.com/topics/engineering/jump-diffusion
    """
    def __init__(self, TGE_total, total_unlocked_history, users, base_price=10.0, elasticity=1.0,
                 buyback_rate=0.2, alpha=0.5, mu=0.0, sigma=0.05, jump_intensity=0.1,
                 jump_mean=-0.05, jump_std=0.1, distribution=None):
        self.TGE_total = TGE_total
        self.total_unlocked_history = np.array(total_unlocked_history)
        self.users = users
        self.base_price = base_price
        self.elasticity = elasticity
        self.buyback_rate = buyback_rate
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.distribution = distribution

    def compute_token_price(self):
        """
        Compute the baseline supply/demand price based on unlocked tokens.

        If distribution is provided, compute avg_sell_weight as:
          avg_sell_weight = (small*1.0 + medium*0.8 + large*0.3 + sybil*1.0) / 100
        Otherwise, defaults to 1.0.
        """
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
        Simulate dynamic token price evolution using a jump-diffusion process.
        Returns an array of simulated prices over the simulation horizon.
        """
        n = len(self.total_unlocked_history)
        baseline_prices = self.compute_token_price()
        P_jump = np.zeros(n)
        P_jump[0] = 1.0
        for t in range(1, n):
            diffusion = np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * np.random.randn())
            if np.random.rand() < self.jump_intensity * dt:
                jump = 1.0 + np.random.normal(self.jump_mean, self.jump_std)
            else:
                jump = 1.0
            P_jump[t] = P_jump[t-1] * diffusion * jump
        prices = baseline_prices * P_jump
        return prices