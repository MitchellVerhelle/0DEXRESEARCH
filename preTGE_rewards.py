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

    Source: dYdX Foundation’s Retroactive Mining Rewards blog, 2021.
            https://dydx.foundation/blog

    The user is placed in one of five tiers based on trading volume (USD).
    Here, we adapt a piecewise constant reward structure:

    Example tiers (approx.):
      - volume < 1,000 USD  => 310 points
      - 1,000 <= volume < 10,000 USD  => 1,163 points
      - 10,000 <= volume < 100,000 USD => 2,584 points
      - 100,000 <= volume < 1,000,000 USD => 6,414 points
      - volume >= 1,000,000 USD => 9,529 points
    """
    def __init__(self, tiers=None):
        # Parameter estimates from:
        # https://cointelegraph.com/news/dydx-airdrop-how-to-claim-310-to-9529-dydx-for-free
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
# Vertex (Pre-TGE) Maker/Taker Reward Policy 
# ======================
class VertexMakerTakerRewardPolicy(PreTGERewardsPolicy):
    """
    Vertex Early User Airdrop (pre-TGE).
    
    As described: ~9% of supply was allocated to early adopters, typically
    awarding points for maker volume (heavier weighting), taker volume, 
    and possibly referral or deposit bonuses.

    Reference: https://mirror.xyz/vertexprotocol.eth / official docs
    According to known references, we can approximate:

      Score_i = maker_volume * maker_weight
              + taker_volume * taker_weight
              + referral_points * referral_rate
      (One could also add deposit volume or Q-score factors if desired.)

    Default weights are hypothetical but reflect the idea that maker volume
    and taker volume are equally weighted in the final pre-TGE push.

    The final token distribution pre-TGE was then proportional to each user's
    total Score_i / sum(Score_j). Here, we only produce the 'points' logic.
    """
    def __init__(self, maker_weight=0.6, taker_weight=0.3, referral_rate=0.1):
        # Parameter estimates:
        # maker_weight=0.6, taker_weight=0.3, referral_rate=0.1
        # mirror approximate emphasis from Vertex docs:
        # "75%/25% after launch" => for pre-TGE, we might skew to ~60%/30% + referral
        self.maker_weight = maker_weight
        self.taker_weight = taker_weight
        self.referral_rate = referral_rate

    def calculate_points(self, activity_stats, user):
        maker = activity_stats.get('maker_volume', 0)
        taker = activity_stats.get('taker_volume', 0)
        referrals = activity_stats.get('referral_points', 0)
        score = (maker * self.maker_weight
                 + taker * self.taker_weight
                 + referrals * self.referral_rate)
        return score


# ======================
# Jupiter Volume Tier Reward Policy
# ======================
class JupiterVolumeTierRewardPolicy(PreTGERewardsPolicy):
    """
    Jupiter's "Jupuary" Airdrop Tiers (pre-TGE).

    The actual airdrop was tiered by user categories (Swap Users, Expert Traders, etc.),
    each with multiple sub-tiers. Here, we approximate it as a piecewise function
    of total swap volume alone, ignoring categories. 
    Reference: https://beincrypto.com/jupiter-airdrop-guide

    Example tiers (in USD volume):
      - volume >= 1,000       =>  50 points
      - volume >= 29,000      => 250 points
      - volume >= 500,000     => 3,000 points
      - volume >= 3,000,000   => 10,000 points
      - volume >= 14,000,000  => 20,000 points
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
# Aevo "Farm Boost" Pre-TGE Reward Policy
# ======================
class AevoFarmBoostRewardPolicy(PreTGERewardsPolicy):
    """
    Aevo Retro + Farm Boost (pre-TGE).
    Reference: Aevo team's Mirror post: https://mirror.xyz/aevo.eth
               ~30M AEVO allocated pre-launch, with volume 'boosted' by some multiplier.

    We consider two volumes:
      - pre_volume: user’s volume before the “farm boost” campaign
      - farm_volume: user’s volume during the “farm boost” period
    Each user might have a personal boost multiplier B_i (1-4), plus possible
    deposit or first-mover bonuses.

    Formula (from compiled references):
      Score_i = pre_volume + B_i * farm_volume + deposit_bonus + early_trader_bonus + ...
    For simplicity, we store them in activity_stats as:
      - 'pre_volume'
      - 'farm_volume'
      - 'boost_mult' (user-specific multiplier, default 2.0)
      - 'deposit_bonus' (e.g. 0 or 100 points)
      - 'early_bonus' (e.g. 0 or 50 points)

    If not present, default them to zero or an appropriate fallback.
    """
    def calculate_points(self, activity_stats, user):
        pre_vol = activity_stats.get('pre_volume', 0)
        farm_vol = activity_stats.get('farm_volume', 0)
        boost = activity_stats.get('boost_mult', 1.0)  # B_i in [1..4]
        deposit_bonus = activity_stats.get('deposit_bonus', 0)
        early_bonus = activity_stats.get('early_bonus', 0)
        # Weighted sum:
        score = pre_vol + boost * farm_vol + deposit_bonus + early_bonus
        return score

# =============================================================================
# Generic Pre-TGE Reward Policy (Custom)
# =============================================================================
class GenericPreTGERewardPolicy(PreTGERewardsPolicy):
    """
    A custom/generic pre-TGE reward policy to be used when a DEX does not
    follow one of the established schemes.

    This policy allows you to define a weighted linear combination of multiple
    activity metrics. For example, a user may earn points based on:
      - trading volume ('volume')
      - number of interactions ('engagement')
      - referral count ('referrals')
      - deposit amount ('deposits')

    The final score is defined as:
    
      Score = w_volume * volume + w_engagement * engagement + w_referrals * referrals + w_deposits * deposits

    You can adjust the weights as needed.

    Source inspiration: General practices in crypto airdrop incentive design and community proposals,
    e.g., discussions on crypto forums (see https://forum.dydx.community).
    """
    def __init__(self, weights=None):
        # Default weights if not provided:
        # Emphasize volume most, then engagement, then referrals, then deposits.
        if weights is None:
            self.weights = {
                'volume': 0.5,
                'engagement': 0.3,
                'referrals': 0.1,
                'deposits': 0.1
            }
        else:
            self.weights = weights

    def calculate_points(self, activity_stats, user):
        score = 0
        for key, weight in self.weights.items():
            score += weight * activity_stats.get(key, 0)
        return score

# -----------------------------------------------------------------------------
# Example usage for testing:
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    dydx_stats = {'trading_volume': 15000}
    vertex_stats = {'maker_volume': 5000, 'taker_volume': 3000, 'referral_points': 20}
    jupiter_stats = {'swap_volume': 600000}
    aevo_stats = {'pre_volume': 20000, 'farm_volume': 5000, 'boost_mult': 2.0, 'deposit_bonus': 100}
    generic_stats = {'volume': 12000, 'engagement': 5, 'referrals': 2, 'deposits': 3000}
    
    class DummyUser:
        pass

    user = DummyUser()

    print("dYdX Retro Tiered Points:", DydxRetroTieredRewardPolicy().calculate_points(dydx_stats, user))
    print("Vertex Maker/Taker Points:", VertexMakerTakerRewardPolicy().calculate_points(vertex_stats, user))
    print("Jupiter Volume Tier Points:", JupiterVolumeTierRewardPolicy().calculate_points(jupiter_stats, user))
    print("Aevo Farm Boost Points:", AevoFarmBoostRewardPolicy().calculate_points(aevo_stats, user))
    print("Generic Pre-TGE Reward Points:", GenericPreTGERewardPolicy().calculate_points(generic_stats, user))