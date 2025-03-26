# DEX RESEARCH

#### We're researching the optimal pre-TGE rewards program, TGE airdrop policy, and post-TGE rewards program to maximize long-term token growth while minimizing token value depreciation.

### The Model: 3 Stages

- **Pre TGE**
- **Airdrop Event**
- **Post TGE**

_We assume the program has one large genesis airdrop event._

---

## New Files Overview

### activity_stats.py

This module encapsulates the logic for generating user activity statistics based on their attributes. For a **RegularUser**, it computes:
- **trading_volume**:  
  $$\text{trading\_volume} = \text{endowment} \times \text{volume\_multiplier}$$
- **maker_volume** and **taker_volume**: Derived from trading volume by splitting randomly (e.g., 30%–70% for maker orders).
- **qscore**: A random quality score in a range that depends on user size.
- **referral_points**: A random value in a range that depends on user size.

For users without a defined `user_size`, a default trading volume is returned.

_Source: Assumptions based on Vertex and dYdX pre-TGE incentive designs (e.g., [Mirror.xyz](https://mirror.xyz/vertexprotocol.eth) and [Cointelegraph](https://cointelegraph.com/news/dydx-airdrop-how-to-claim-310-to-9529-dydx-for-free))._

---

### vesting.py

This file defines the vesting schedules and the **PostTGERewardsManager** class. It sets the unlocking parameters (allocation, cliffs, linear unlock durations) for each stakeholder group. For example, a typical team vesting schedule might unlock 20% at TGE, then impose a 12‑month lockup, followed by a 36‑month linear unlock.

_Parameter sources include common vesting guidelines from [Messari](https://messari.io/asset/unlock-schedules) and [Solana documentation](https://docs.solana.com/economics)._

---

### postTGE_rewards_policy.py

This module defines post-TGE reward policies that apply engagement multipliers to users’ token rewards after TGE. For example, the **EngagementMultiplierPolicy** uses:

$$
\text{multiplier} = 1 + \gamma \left(\frac{\text{active\_days}}{T}\right)^\delta
$$

where:
- \(\gamma\) is the maximum additional multiplier (e.g., \(\gamma = 0.5\) implies full activity yields a 1.5× multiplier),
- \(T\) is the simulation horizon (in months),
- \(\delta\) is an exponent that emphasizes consistency (values \(>1\) accentuate differences).

The **GenericPostTGERewardPolicy** applies this multiplier to a user's token balance.

_Source: Inspired by Vertex’s VoVertex system ([Vertex Docs](https://vertexprotocol.com/docs/trade-and-earn)) and general DEX incentive designs._

---

### Price Calculation Changes

Price evolution now proceeds in discrete steps. At each time step \(t\):

1. **Baseline Supply/Demand Price** is computed as:

   $$
   price(t) = base_{price} \times \left(\frac{TGE_{total}}{combined_{supply}(t)}\right)^{elasticity}
   $$

   where

   $$
   combined_{supply}(t) = \alpha \times circulating_{supply}(t) + (1-\alpha) \times effective_{supply}(t)
   $$

   with

   $$
   circulating_{supply}(t) = TGE_{total} + \left(postTGE_{unlocked}(t)-postTGE_{unlocked}(0)\right) \times avg_{sell\_weight}
   $$
   $$
   effective_{supply}(t) = TGE_{total} + \left(postTGE_{unlocked}(t)-postTGE_{unlocked}(0)\right) \times \left(avg_{sell\_weight} \times (1 - buyback_{rate})\right)
   $$

2. **Dynamic Price Evolution** is modeled with a jump-diffusion process:

   - **Diffusion Component (GBM):**
     
     $$
     P_{diff}(t) = \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t}\, Z_t\right), \quad Z_t \sim N(0,1)
     $$
     
   - **Jump Component:**

     $$
     J_t = 
     \begin{cases}
     1 + jump_{size}, & \text{with probability } jump_{intensity} \Delta t, \\
     1, & \text{otherwise},
     \end{cases}
     $$
     
     where \(jump_{size} \sim N(jump_{mean}, jump_{std})\).

   - **Dynamic Multiplier:**

     $$
     P_{jump}(t) = P_{jump}(t-1) \times P_{diff}(t) \times J_t,\quad P_{jump}(0)=1
     $$

3. **Final Token Price:**

   $$
   Final\ Price(t) = price(t) \times P_{jump}(t)
   $$

In addition, at each time step the active user fraction \(A(t)\) is updated via:

$$
A_{t+1} = A_t + \left[\alpha (1-A_t) - \beta \left(1+\theta\frac{p(t)-p_0}{p_0}\right)A_t\right] dt + \epsilon_t,
$$

with \(\epsilon_t \sim N(0,\sigma_{de}\, dt)\).

_Sources: [Investopedia: Geometric Brownian Motion](https://www.investopedia.com/terms/g/geometric-brownian-motion.asp) and [ScienceDirect: Jump Diffusion](https://www.sciencedirect.com/topics/engineering/jump-diffusion)._

---

## airdrop_policy.py

The airdrop policies include:

- **Linear**  
  $$ tokens = factor \times airdrop_{points} $$
  ([Vertex Protocol](https://messari.io/project/vertex-protocol/token-unlocks))

- **Exponential**  
  $$ tokens = factor \times \left(e^{\frac{airdrop_{points}}{scaling}} - 1\right) $$
  ([Exponential Airdrop Models 2021](https://medium.com/@blockchaintokenomics/exponential-airdrop-models-2021))

- **Tiered Constant**  
  $$
  tokens =
  \begin{cases}
  0.1, & airdrop_{points} < 0.2 \\
  0.4, & 0.2 \le airdrop_{points} < 0.6 \\
  1.0, & airdrop_{points} \ge 0.6
  \end{cases}
  $$
  ([Tokenomics Lab](https://tokenomicslab.org/airdrop-mechanisms))

- **Tiered Linear**  
  $$
  tokens = 
  \begin{cases}
  1.0 \times airdrop_{points}, & airdrop_{points} \le 0.2 \\
  1.0 \times 0.2 + 1.5 \times (airdrop_{points}-0.2), & 0.2 < airdrop_{points} \le 0.6 \\
  1.0 \times 0.2 + 1.5 \times 0.4 + 2.0 \times (airdrop_{points}-0.6), & airdrop_{points} > 0.6
  \end{cases}
  $$
  ([Consensys](https://consensys.net/blog/blockchain-explained/token-distribution-models))

- **Tiered Exponential**  
  $$ tokens = \sum_{tiers} factor \times \left(e^{\frac{\Delta airdrop_{points}}{scaling}} - 1\right) $$
  ([Crypto Economics Handbook](https://www.cryptoeconomicshandbook.org/token-distribution))

---

## preTGE_rewards.py

The preTGE rewards policies remain largely as before for dYdX Retro, Vertex Maker/Taker, Jupiter Volume Tier, and Aevo Farm Boost. We have removed the standalone Helix and Game-like MMR policies and now incorporate Game-like MMR as our custom, generic reward policy using **GenericPreTGERewardPolicy**.

_Source: Discussion on crypto forums (e.g., [dYdX Community](https://forum.dydx.community)) and earlier research._

---

## postTGE_rewards.py & postTGE_rewards_policy.py

In **postTGE_rewards.py**, we now simulate price evolution step-by-step. At each time step \(t\):
1. The baseline price is computed from vesting (using the supply/demand model).
2. A jump-diffusion process updates the dynamic multiplier:
   $$
   P_{jump}(t) = P_{jump}(t-1) \times \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t}\, Z_t\right) \times J_t
   $$
   where
   $$
   J_t =
   \begin{cases}
   1 + jump_{size}, & \text{with probability } jump_{intensity}\Delta t, \\
   1, & \text{otherwise},
   \end{cases}
   $$
   with \(jump_{size} \sim N(jump_{mean}, jump_{std})\).
3. User activity is updated via a retention model:
   $$
   A_{t+1} = A_t + \left[\alpha (1-A_t) - \beta \left(1+\theta\frac{p(t)-p_0}{p_0}\right)A_t\right] dt + \epsilon_t,
   $$
   with \(\epsilon_t \sim N(0,\sigma_{de}\, dt)\).

In **postTGE_rewards_policy.py**, the **GenericPostTGERewardPolicy** applies an engagement multiplier to user tokens:

$$
\text{multiplier} = 1 + \gamma \left(\frac{\text{active\_days}}{T}\right)^\delta,
$$

where \(\gamma\) and \(\delta\) are user-defined parameters.  
_Source: [Vertex Docs](https://vertexprotocol.com/docs/trade-and-earn) and general DEX incentive research._

---

## Plotting Overview

The following plots are produced:
- **Histogram of TGE Token Distribution:**  
  Displays the distribution of TGE tokens for each pre-TGE and airdrop policy combination (ignoring post-TGE variations).
  
- **Vesting Schedule Plot:**  
  Shows a stackplot of unlocked tokens per group over time, with a dashed line overlay representing the total unlocked tokens.

- **Price Evolution Overlay Grid:**  
  For each pre-TGE + airdrop combination, all post-TGE price evolution curves (Baseline, High Volatility, Low Volatility, Aggressive Buyback) are overlayed on a single subplot. Grey bars indicate the Baseline active fraction over time. The grid layout is user-definable via maximum rows and columns per page.

---

## Instructions to Run

- To run the simulation, enter `python main.py` in the root directory.
- Adjust simulation parameters in **main.py** as needed.
- Graphs will appear sequentially; close each graph window to see the next.
- Save graphs as desired.

---

_This README.md provides a concise overview of the new file structure, price evolution model, and updated reward policies with supporting mathematical descriptions and citations. Replace your current README.md with the content above to integrate these changes._
