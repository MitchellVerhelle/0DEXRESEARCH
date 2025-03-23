import numpy as np

class VestingSchedule:
    """
    Represents a vesting schedule for a token allocation.
    
    Parameters:
      - allocation: The total allocation for this group (absolute tokens or fraction of total supply).
      - unlock_at_tge: Fraction unlocked at TGE.
      - lockup_duration: Duration (in months) during which no additional tokens unlock beyond TGE.
      - initial_cliff_unlock: Fraction unlocked at the initial cliff after lockup.
      - unlock_duration: Duration (in months) over which the remaining tokens unlock linearly.
      - initial_cliff_delay: For groups with no lockup duration, the delay (in months) before the initial cliff unlock.
                           Defaults to 0 if lockup_duration > 0.
    """
    def __init__(self, allocation, unlock_at_tge, lockup_duration, initial_cliff_unlock, unlock_duration, initial_cliff_delay=0):
        self.allocation = allocation
        self.unlock_at_tge = unlock_at_tge
        self.lockup_duration = lockup_duration
        self.initial_cliff_unlock = initial_cliff_unlock
        self.unlock_duration = unlock_duration
        self.initial_cliff_delay = initial_cliff_delay

    def get_unlocked_fraction(self, months_elapsed):
        """
        Returns the fraction of allocation unlocked at a given time (in months since TGE).
        """
        if months_elapsed < 0:
            return 0.0

        # For groups with a lockup period.
        if self.lockup_duration > 0:
            if months_elapsed < self.lockup_duration:
                return self.unlock_at_tge
            elif months_elapsed < self.lockup_duration + self.unlock_duration:
                linear_progress = (months_elapsed - self.lockup_duration) / self.unlock_duration
                return self.unlock_at_tge + self.initial_cliff_unlock + (1.0 - self.unlock_at_tge - self.initial_cliff_unlock) * linear_progress
            else:
                return 1.0
        else:
            # For groups with no lockup, use initial cliff delay.
            if months_elapsed < self.initial_cliff_delay:
                return self.unlock_at_tge
            elif months_elapsed < self.initial_cliff_delay + self.unlock_duration:
                linear_progress = (months_elapsed - self.initial_cliff_delay) / self.unlock_duration
                return self.unlock_at_tge + self.initial_cliff_unlock + (1.0 - self.unlock_at_tge - self.initial_cliff_unlock) * linear_progress
            else:
                return 1.0

    def get_unlocked_tokens(self, months_elapsed):
        """
        Returns the absolute number of tokens unlocked given the time elapsed (in months since TGE).
        """
        fraction = self.get_unlocked_fraction(months_elapsed)
        return self.allocation * fraction

class PostTGERewardsManager:
    """
    Manages vesting schedules for all allocation groups.
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
        """
        Returns a dictionary mapping each group to the absolute number of tokens unlocked at the given time.
        """
        unlocked = {}
        for group, schedule in self.schedules.items():
            unlocked[group] = schedule.get_unlocked_tokens(months_elapsed)
        return unlocked

if __name__ == '__main__':
    total_supply = 100_000_000  # e.g., 100 million tokens
    manager = PostTGERewardsManager(total_supply)
    
    for month in [0, 6, 12, 18, 24, 36, 48, 60]:
        unlocked_allocations = manager.get_unlocked_allocations(month)
        print(f"At {month} months:")
        for group, tokens in unlocked_allocations.items():
            print(f"  {group}: {tokens:.0f} tokens")
        print()
