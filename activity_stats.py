import numpy as np

def generate_stats(user):
    """
    Generate activity statistics for a user based on their attributes.
    
    For RegularUser:
      - 'trading_volume': computed as endowment * volume_multiplier.
      - 'maker_volume' and 'taker_volume': derived from trading_volume.
      - 'qscore': random value (quality score) within a range based on user size.
      - 'referral_points': random value within a range based on user size.
      
      Additionally, for more detailed reward policies we add:
      - 'swap_volume': a separate measure of swap activity.
      - 'pre_volume': volume before the farm boost campaign.
      - 'farm_volume': volume during the farm boost period.
      - 'boost_mult': a user-specific boost multiplier (default between 1 and 4).
      - 'deposit_bonus': a bonus based on deposits (e.g., 0, 50, or 100).
      - 'early_bonus': a bonus for early trading (e.g., 0, 25, or 50).
      - 'volume': a generic volume measure (we set this equal to trading_volume).
      - 'engagement': a random value representing user interactions.
      - 'referrals': a count of referrals.
      - 'deposits': a value representing deposit amount.
      
    For users without a defined 'user_size', only a simplified 'trading_volume' is returned.
    
    Source: Adapted from assumptions based on Vertex and dYdX pre-TGE incentive designs.
    """
    if hasattr(user, 'user_size'):
        # Set parameters based on user size.
        if user.user_size == 'small':
            volume_multiplier = 50
            qscore_range = (50, 150)
            referral_range = (0, 50)
        elif user.user_size == 'medium':
            volume_multiplier = 150
            qscore_range = (100, 200)
            referral_range = (0, 100)
        elif user.user_size == 'large':
            volume_multiplier = 300
            qscore_range = (150, 300)
            referral_range = (0, 150)
        else:
            volume_multiplier = 100
            qscore_range = (100, 200)
            referral_range = (0, 100)
        
        # Basic stats.
        trading_volume = user.endowment * volume_multiplier
        maker_volume = trading_volume * np.random.uniform(0.3, 0.7)
        taker_volume = trading_volume - maker_volume
        
        # Additional keys for various policies.
        # For Jupiter Volume Tier Reward Policy.
        swap_volume = trading_volume * np.random.uniform(0.8, 1.2)
        
        # For Aevo Farm Boost Reward Policy.
        pre_volume = trading_volume * np.random.uniform(0.5, 0.8)
        farm_volume = trading_volume * np.random.uniform(0.2, 0.5)
        boost_mult = np.random.choice([1.0, 2.0, 3.0, 4.0])
        deposit_bonus = np.random.choice([0, 50, 100])
        early_bonus = np.random.choice([0, 25, 50])
        
        # For Generic Pre-TGE Reward Policy.
        volume_generic = trading_volume  # can be the same as trading_volume
        engagement = np.random.uniform(1, 10)
        referrals = np.random.randint(0, 5)
        deposits = np.random.uniform(100, 1000)
        
        stats = {
            'trading_volume': trading_volume,
            'maker_volume': maker_volume,
            'taker_volume': taker_volume,
            'qscore': np.random.uniform(*qscore_range),
            'referral_points': np.random.uniform(*referral_range),
            'swap_volume': swap_volume,
            'pre_volume': pre_volume,
            'farm_volume': farm_volume,
            'boost_mult': boost_mult,
            'deposit_bonus': deposit_bonus,
            'early_bonus': early_bonus,
            'volume': volume_generic,
            'engagement': engagement,
            'referrals': referrals,
            'deposits': deposits
        }
    else:
        stats = {'trading_volume': user.endowment * 100}
    return stats
