import numpy as np

def generate_stats(user):
    """
    Generate activity statistics for a user based on their attributes.
    
    For RegularUser:
      - 'trading_volume': computed as endowment * volume_multiplier.
      - 'maker_volume' and 'taker_volume': derived from trading_volume.
      - 'qscore': random value (quality score) within a range based on user size.
      - 'referral_points': random value within a range based on user size.
    
    For users without a defined 'user_size', a default 'trading_volume' is returned.
    
    Source: Assumptions based on Vertex and dYdX pre-TGE incentive designs.
    e.g., https://mirror.xyz/vertexprotocol.eth and 
          https://cointelegraph.com/news/dydx-airdrop-how-to-claim-310-to-9529-dydx-for-free
    """
    if hasattr(user, 'user_size'):
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
            
        trading_volume = user.endowment * volume_multiplier
        maker_volume = trading_volume * np.random.uniform(0.3, 0.7)
        taker_volume = trading_volume - maker_volume
        stats = {
            'trading_volume': trading_volume,
            'maker_volume': maker_volume,
            'taker_volume': taker_volume,
            'qscore': np.random.uniform(*qscore_range),
            'referral_points': np.random.uniform(*referral_range)
        }
    else:
        stats = {'trading_volume': user.endowment * 100}
    return stats
