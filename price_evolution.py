import numpy as np
from math import isclose
from scipy.optimize import brentq  # A robust root-finding method for 1D.

class PriceEvolution:
    """
    Each month, we solve for p such that supply(t) = demand(t, p).
    This ensures we get an equilibrium price without blow-ups.
    """
    def __init__(self, monthly_supply, monthly_usd_demand, elasticity=1.0):
        """
        monthly_supply: array of length T+1 with total token supply in circulation each month.
        monthly_usd_demand: array of length T+1 with the nominal 'USD willingness-to-pay' that 
                            grows over time (like a sigmoid).
        elasticity: price elasticity (often 1.0 if you interpret demand as monthly_usd_demand / p).
        
        If elasticity != 1, we interpret demand as:
            Demand(t, p) = monthly_usd_demand[t] * p^(-elasticity).
        """
        self.monthly_supply = np.array(monthly_supply, dtype=float)
        self.monthly_usd_demand = np.array(monthly_usd_demand, dtype=float)
        self.elasticity = elasticity
        # Edge-case checks:
        if len(self.monthly_supply) != len(self.monthly_usd_demand):
            raise ValueError("monthly_supply and monthly_usd_demand must be same length.")

    def demand_function(self, month_idx, price):
        """
        Basic demand function: 
            Q_d = (USD_demand) * p^(-elasticity)
        Example:
          if elasticity=1, Q_d = (usd demand) / price
          if elasticity=0.5, Q_d = (usd demand) / p^0.5, etc.
        """
        if price <= 0.0:
            return 0.0
        usd_d = self.monthly_usd_demand[month_idx]
        return usd_d * (price ** (-self.elasticity))

    def find_equilibrium_price(self, month_idx):
        """
        Solve supply[month_idx] = demand_function(month_idx, p).
        Use brentq in [1e-12, 1e9] to avoid negative or insane blow-ups.
        Return 0 if supply=0 or demand=0.
        """
        supply_t = self.monthly_supply[month_idx]
        if supply_t <= 1e-12:
            return 0.0  # no tokens => price=0 or undefined. We'll pick 0.

        # We want f(p) = demand_function - supply = 0.
        def f(p):
            return self.demand_function(month_idx, p) - supply_t

        # If the demand is extremely large at very low prices, we might bracket carefully.
        # We'll do a basic bracket from (1e-12 to 1e9).
        low, high = 1e-12, 1e9
        # Check if f(low) > 0 => if so, the supply might exceed demand at min price => pick low
        # Or if f(high)<0 => supply might exceed demand at max price => pick high
        # We'll do some checks:
        f_low = f(low)
        f_high = f(high)
        if f_low < 0 and f_high < 0:
            # means demand < supply even at p=1e-12 => maybe no solution => pick near 0
            return 0.0
        if f_low > 0 and f_high > 0:
            # means demand > supply even at p=1e9 => means equilibrium price is huge, but let's cap.
            return high

        # Otherwise, there's a root in [low, high].
        try:
            p_star = brentq(f, low, high, maxiter=500)
            return p_star
        except:
            # If brentq fails (e.g. demand function is messed up, no bracket found), fallback
            return 0.0

    def simulate(self):
        """
        Return an array of equilibrium prices p[t] for each month t.
        """
        n = len(self.monthly_supply)
        prices = np.zeros(n)
        for t in range(n):
            prices[t] = self.find_equilibrium_price(t)
        return prices
