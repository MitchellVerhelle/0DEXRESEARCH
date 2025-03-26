import numpy as np

class VestingSchedule:
    """
    Represents a vesting schedule for a token allocation.

    Parameters:
      - allocation: Total allocation (absolute tokens or fraction of total supply).
      - unlock_at_tge: Fraction unlocked at TGE.
      - lockup_duration: Duration (in months) with no additional unlock after TGE.
      - initial_cliff_unlock: Fraction unlocked immediately after lockup.
      - unlock_duration: Duration (in months) over which remaining tokens unlock linearly.
      - initial_cliff_delay: (For no lockup) Delay (in months) before the initial cliff unlock.
                            Defaults to 0 if lockup_duration > 0.
    
    Common vesting patterns (e.g., Team vesting: 20% TGE, 12-month lockup, then 36-month linear)
    are used by many crypto projects.
    Sources: Common vesting guidelines (e.g., https://messari.io/asset/unlock-schedules)
    """
    def __init__(self, allocation, unlock_at_tge, lockup_duration, initial_cliff_unlock, unlock_duration, initial_cliff_delay=0):
        self.allocation = allocation
        self.unlock_at_tge = unlock_at_tge
        self.lockup_duration = lockup_duration
        self.initial_cliff_unlock = initial_cliff_unlock
        self.unlock_duration = unlock_duration
        self.initial_cliff_delay = initial_cliff_delay

    def get_unlocked_fraction(self, months_elapsed):
        if months_elapsed < 0:
            return 0.0

        if self.lockup_duration > 0:
            if months_elapsed < self.lockup_duration:
                return self.unlock_at_tge
            elif months_elapsed < self.lockup_duration + self.unlock_duration:
                linear_progress = (months_elapsed - self.lockup_duration) / self.unlock_duration
                return self.unlock_at_tge + self.initial_cliff_unlock + (1 - self.unlock_at_tge - self.initial_cliff_unlock) * linear_progress
            else:
                return 1.0
        else:
            if months_elapsed < self.initial_cliff_delay:
                return self.unlock_at_tge
            elif months_elapsed < self.initial_cliff_delay + self.unlock_duration:
                linear_progress = (months_elapsed - self.initial_cliff_delay) / self.unlock_duration
                return self.unlock_at_tge + self.initial_cliff_unlock + (1 - self.unlock_at_tge - self.initial_cliff_unlock) * linear_progress
            else:
                return 1.0

    def get_unlocked_tokens(self, months_elapsed):
        fraction = self.get_unlocked_fraction(months_elapsed)
        return self.allocation * fraction

class PostTGERewardsManager:
    """
    Manages vesting schedules for different stakeholder groups.

    The distributions below are based on typical token allocation models:
      - Team: 20% (20% TGE, 12-month lockup, then linear over 36 months)
      - TGE Airdrop: 15% (fully unlocked)
      - Future Investors: 17% (linear over 48 months)
      - Treasury: 20% (linear over 48 months)
      - Investors: 8% (20% TGE, then linear over 36 months with 1-month cliff)
      - Rewards: 5% (fully unlocked)
      - Promotion: 3% (partial TGE + linear over 24 months with 1-month delay)
      - Advisors: 2% (partial TGE, 12-month lockup, then linear over 36 months)

    Sources: Typical vesting structures (e.g., https://docs.solana.com/economics)
    """
    def __init__(self, total_supply):
        self.total_supply = total_supply
        self.schedules = {
            "Team": VestingSchedule(
                allocation=total_supply * 0.20,
                unlock_at_tge=0.20,
                lockup_duration=12,
                initial_cliff_unlock=0.20,
                unlock_duration=36
            ),
            "TGE Airdrop": VestingSchedule(
                allocation=total_supply * 0.15,
                unlock_at_tge=1.0,
                lockup_duration=0,
                initial_cliff_unlock=0.0,
                unlock_duration=0
            ),
            "Future Investors": VestingSchedule(
                allocation=total_supply * 0.17,
                unlock_at_tge=0.0,
                lockup_duration=0,
                initial_cliff_unlock=0.0,
                unlock_duration=48
            ),
            "Strategic Reserve/Treasury": VestingSchedule(
                allocation=total_supply * 0.20,
                unlock_at_tge=0.0,
                lockup_duration=0,
                initial_cliff_unlock=0.0,
                unlock_duration=48
            ),
            "Investors": VestingSchedule(
                allocation=total_supply * 0.08,
                unlock_at_tge=0.20,
                lockup_duration=0,
                initial_cliff_unlock=0.20,
                unlock_duration=36,
                initial_cliff_delay=1
            ),
            "Rewards": VestingSchedule(
                allocation=total_supply * 0.05,
                unlock_at_tge=1.0,
                lockup_duration=0,
                initial_cliff_unlock=0.0,
                unlock_duration=0
            ),
            "Promotion": VestingSchedule(
                allocation=total_supply * 0.03,
                unlock_at_tge=0.20,
                lockup_duration=0,
                initial_cliff_unlock=0.20,
                unlock_duration=24,
                initial_cliff_delay=1
            ),
            "Advisors": VestingSchedule(
                allocation=total_supply * 0.02,
                unlock_at_tge=0.20,
                lockup_duration=12,
                initial_cliff_unlock=0.20,
                unlock_duration=36
            )
        }

    def get_unlocked_allocations(self, months_elapsed):
        unlocked = {}
        for group, schedule in self.schedules.items():
            unlocked[group] = schedule.get_unlocked_tokens(months_elapsed)
        return unlocked

# Example usage for vesting simulation:
if __name__ == '__main__':
    total_supply = 100_000_000  # e.g., 100 million tokens
    manager = PostTGERewardsManager(total_supply)
    
    for month in [0, 6, 12, 18, 24, 36, 48, 60]:
        allocations = manager.get_unlocked_allocations(month)
        print(f"At {month} months:")
        for group, tokens in allocations.items():
            print(f"  {group}: {tokens:.0f} tokens")
        print()
