import numpy as np

class PreTGERewardsPolicy:
    """
    Base class for pre-TGE rewards policies.
    
    Subclasses must implement calculate_points(activity_stats, user)
    to determine how many reward points a user earns based on their activity.
    """
    def calculate_points(self, activity_stats, user):
        raise NotImplementedError("Subclasses should implement this method.")


# ======================
# dYdX Retroactive (Tiered Fixed Reward) Policy
# ======================
class DydxRetroTieredRewardPolicy(PreTGERewardsPolicy):
    """
    Mimics the retroactive airdrop tiers used by dYdX.
    
    Based on total trading volume (in USD), a fixed reward (points) is assigned.
    Default tiers (example values):
      - volume < 1,000 USD  => 310 points (base deposit bonus)
      - 1,000 <= volume < 10,000 USD  => 1,163 points
      - 10,000 <= volume < 100,000 USD  => 2,500 points
      - 100,000 <= volume < 1,000,000 USD  => 6,414 points
      - volume >= 1,000,000 USD  => 9,530 points
    """
    def __init__(self, tiers=None):
        if tiers is None:
            self.tiers = [
                (1000, 310),
                (10000, 1163),
                (100000, 2500),
                (1000000, 6414),
                (np.inf, 9530)
            ]
        else:
            self.tiers = tiers

    def calculate_points(self, activity_stats, user):
        volume = activity_stats.get('trading_volume', 0)
        for threshold, points in self.tiers:
            if volume < threshold:
                return points
        return self.tiers[-1][1]


# ======================
# Vertex Maker/Taker Reward Policy
# ======================
class VertexMakerTakerRewardPolicy(PreTGERewardsPolicy):
    """
    Implements a rewards scheme inspired by Vertex’s pre-launch program.
    
    Users earn points based on:
      - maker_volume weighted at 37.5%
      - taker_volume weighted at 37.5%
      - a Q-score (liquidity quality) weighted at 25%
    Additionally, a referral bonus (e.g., 25% of the referees’ points) is added.
    
    activity_stats should include:
      'maker_volume', 'taker_volume', 'qscore', and optionally 'referral_points'
    """
    def __init__(self, maker_weight=0.375, taker_weight=0.375, qscore_weight=0.25, referral_rate=0.25):
        self.maker_weight = maker_weight
        self.taker_weight = taker_weight
        self.qscore_weight = qscore_weight
        self.referral_rate = referral_rate

    def calculate_points(self, activity_stats, user):
        maker = activity_stats.get('maker_volume', 0)
        taker = activity_stats.get('taker_volume', 0)
        qscore = activity_stats.get('qscore', 0)
        base_points = (maker * self.maker_weight +
                       taker * self.taker_weight +
                       qscore * self.qscore_weight)
        referral_points = activity_stats.get('referral_points', 0)
        bonus = referral_points * self.referral_rate
        return base_points + bonus


# ======================
# Jupiter Volume Tier Reward Policy
# ======================
class JupiterVolumeTierRewardPolicy(PreTGERewardsPolicy):
    """
    Mimics the tiered, piecewise constant reward system as seen in Jupiter's airdrop.
    
    Users are awarded a fixed number of points based on swap volume thresholds.
    Example tiers (in USD volume):
      - volume >= 1,000: 50 points
      - volume >= 29,000: 250 points
      - volume >= 500,000: 3,000 points
      - volume >= 3,000,000: 10,000 points
      - volume >= 14,000,000: 20,000 points
    
    activity_stats should include:
      'swap_volume'
    """
    def __init__(self, tiers=None):
        if tiers is None:
            self.tiers = [
                (1000, 50),
                (29000, 250),
                (500000, 3000),
                (3000000, 10000),
                (14000000, 20000)
            ]
        else:
            self.tiers = tiers

    def calculate_points(self, activity_stats, user):
        volume = activity_stats.get('swap_volume', 0)
        reward = 0
        for threshold, points in self.tiers:
            if volume >= threshold:
                reward = points
            else:
                break
        return reward


# ======================
# Aevo Boosted Volume Reward Policy
# ======================
class AevoBoostedVolumeRewardPolicy(PreTGERewardsPolicy):
    """
    Mimics Aevo's gamified reward system that applies a base boost (based on trailing volume)
    and a chance-based "lucky boost" to each trade.
    
    activity_stats should include:
      'trade_volume': the notional volume for the current trade,
      'trailing_volume': the total volume over a trailing 7-day window.
    
    The base boost scales from 1× (no boost) to base_max (e.g., 4×) as trailing_volume increases.
    A lucky boost is applied probabilistically. Default probabilities:
      - 10x boost with 10% chance,
      - 50x boost with 2.5% chance,
      - 100x boost with 1% chance.
    """
    def __init__(self, base_max=4.0, lucky_probs=None):
        self.base_max = base_max
        if lucky_probs is None:
            # Sorted by multiplier values in ascending order
            self.lucky_probs = {10: 0.10, 50: 0.025, 100: 0.01}
        else:
            self.lucky_probs = lucky_probs

    def calculate_points(self, activity_stats, user):
        trade_volume = activity_stats.get('trade_volume', 0)
        trailing_volume = activity_stats.get('trailing_volume', 0)
        # Determine base boost multiplier.
        threshold = 5000000  # Example threshold for maximum boost
        base_multiplier = 1 + (self.base_max - 1) * min(trailing_volume / threshold, 1)
        
        # Determine lucky boost multiplier using probability.
        lucky_multiplier = 1
        rnd = np.random.rand()
        cumulative = 0
        for multiplier, prob in sorted(self.lucky_probs.items(), key=lambda x: x[1]):
            cumulative += prob
            if rnd < cumulative:
                lucky_multiplier = multiplier
                break

        # Final boosted points: the trade volume is amplified by the sum of boosts minus 1 
        # (since a base multiplier of 1 means no extra boost).
        return trade_volume * (base_multiplier + lucky_multiplier - 1)


# ======================
# Helix (Injective) Loyalty Points Reward Policy
# ======================
class HelixLoyaltyPointsRewardPolicy(PreTGERewardsPolicy):
    """
    Mimics the multi-factor loyalty point system used by Injective/Helix.
    
    This policy rewards users based on:
      - trading volume (weighted linearly),
      - diversity bonus (number of unique markets traded),
      - loyalty bonus (active days multiplied by volume).
    
    activity_stats should include:
      'trading_volume', 'active_days', 'unique_markets'
    """
    def __init__(self, volume_weight=1.0, diversity_bonus=100, loyalty_bonus=0.1):
        self.volume_weight = volume_weight
        self.diversity_bonus = diversity_bonus
        self.loyalty_bonus = loyalty_bonus

    def calculate_points(self, activity_stats, user):
        volume = activity_stats.get('trading_volume', 0)
        active_days = activity_stats.get('active_days', 0)
        unique_markets = activity_stats.get('unique_markets', 0)
        return (self.volume_weight * volume +
                self.diversity_bonus * unique_markets +
                self.loyalty_bonus * active_days * volume)


# ======================
# Game-like (MMR) Reward Policy
# ======================
class GameLikeMMRRewardPolicy(PreTGERewardsPolicy):
    """
    Implements a game-like rewards system that is inspired by MMR (Match-Making Rating)
    used in competitive video games.
    
    activity_stats should include:
      'wins', 'losses', and 'consecutive_days' of activity.
    
    The reward is based on:
      - a base number of points,
      - bonus points proportional to win rate,
      - additional bonus for consistency (consecutive days active).
    """
    def __init__(self, base_points=1000, win_rate_weight=500, consistency_bonus=300):
        self.base_points = base_points
        self.win_rate_weight = win_rate_weight
        self.consistency_bonus = consistency_bonus

    def calculate_points(self, activity_stats, user):
        wins = activity_stats.get('wins', 0)
        losses = activity_stats.get('losses', 0)
        consecutive_days = activity_stats.get('consecutive_days', 0)
        total_games = wins + losses
        win_rate = wins / total_games if total_games > 0 else 0
        return (self.base_points +
                self.win_rate_weight * win_rate +
                self.consistency_bonus * consecutive_days)


# ======================
# Custom / Generic Reward Policy
# ======================
class CustomPreTGERewardPolicy(PreTGERewardsPolicy):
    """
    A flexible reward policy that accepts a custom function.
    
    The custom function should have the signature:
        custom_function(activity_stats, user) -> points
    """
    def __init__(self, custom_function):
        self.custom_function = custom_function

    def calculate_points(self, activity_stats, user):
        return self.custom_function(activity_stats, user)


# Example usage (for testing or integration):
if __name__ == '__main__':
    # Simulated activity stats for a given user:
    activity_dydx = {'trading_volume': 15000}
    activity_vertex = {'maker_volume': 5000, 'taker_volume': 3000, 'qscore': 200, 'referral_points': 100}
    activity_jupiter = {'swap_volume': 600000}
    activity_aevo = {'trade_volume': 25000, 'trailing_volume': 3000000}
    activity_helix = {'trading_volume': 10000, 'active_days': 20, 'unique_markets': 5}
    activity_mmr = {'wins': 8, 'losses': 2, 'consecutive_days': 10}

    # Dummy user object (could be extended from your user classes)
    class DummyUser:
        pass
    dummy_user = DummyUser()

    # Create policy instances:
    dydx_policy = DydxRetroTieredRewardPolicy()
    vertex_policy = VertexMakerTakerRewardPolicy()
    jupiter_policy = JupiterVolumeTierRewardPolicy()
    aevo_policy = AevoBoostedVolumeRewardPolicy()
    helix_policy = HelixLoyaltyPointsRewardPolicy()
    mmr_policy = GameLikeMMRRewardPolicy()

    # Calculate and print reward points for each:
    print("dYdX Retro Tiered Reward Points:", dydx_policy.calculate_points(activity_dydx, dummy_user))
    print("Vertex Maker/Taker Reward Points:", vertex_policy.calculate_points(activity_vertex, dummy_user))
    print("Jupiter Volume Tier Reward Points:", jupiter_policy.calculate_points(activity_jupiter, dummy_user))
    print("Aevo Boosted Volume Reward Points:", aevo_policy.calculate_points(activity_aevo, dummy_user))
    print("Helix Loyalty Points Reward Points:", helix_policy.calculate_points(activity_helix, dummy_user))
    print("Game-like MMR Reward Points:", mmr_policy.calculate_points(activity_mmr, dummy_user))
