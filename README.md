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

$$
\text{trading\_volume} = \text{endowment} \times \text{volume\_multiplier}
$$

Maker and taker volumes are split randomly (for example, 30%–70%) from `trading_volume`. A random `qscore` depends on user size, and a random `referral_points` depends on user size.  
If a user lacks a defined `user_size`, a default volume is returned.

_Source: Assumptions based on Vertex and dYdX pre-TGE incentive designs ([Mirror.xyz](https://mirror.xyz/vertexprotocol.eth) and [Cointelegraph](https://cointelegraph.com/news/dydx-airdrop-how-to-claim-310-to-9529-dydx-for-free))._

---

### vesting.py

Defines vesting schedules and the **PostTGERewardsManager**. Each group’s tokens are subject to lockups, cliffs, and linear unlocks. For instance, a typical team allocation might unlock 20% at TGE, lock for 12 months, then linear‐unlock over 36 months.

_Sources: [Messari](https://messari.io/asset/unlock-schedules) and [Solana docs](https://docs.solana.com/economics)._

---

### postTGE_rewards_policy.py

Contains post-TGE engagement reward policies. For instance, **EngagementMultiplierPolicy** applies:

\[
\text{multiplier} = 1 + \gamma \left(\frac{\text{active\_days}}{T}\right)^\delta
\]

- $\gamma$ = max additional multiplier (e.g. 0.5 → full activity yields 1.5×),
- $T$ = simulation horizon (months),
- $\delta$ = exponent emphasizing consistency ($>1$ accentuates differences).

**GenericPostTGERewardPolicy** applies this multiplier to a user's token balance.

_Sources: Vertex’s VoVertex ([Vertex Docs](https://vertexprotocol.com/docs/trade-and-earn)) & general DEX incentives._

---

## Price Calculation Changes

We now step through time. Each step $t$:

1. **Baseline Supply/Demand Price**:

$$
\text{price}(t) = \text{base\_price} \times \Bigl(\frac{\text{TGE\_total}}{\text{combined\_supply}(t)}\Bigr)^\text{elasticity}
$$

with

$$
\text{combined\_supply}(t) = \alpha \times \text{circulating\_supply}(t) \;+\; (1-\alpha)\,\text{effective\_supply}(t)
$$

and

$$
\text{circulating\_supply}(t) = \text{TGE\_total} + \bigl(\text{postTGE\_unlocked}(t)-\text{postTGE\_unlocked}(0)\bigr)\times\text{avg\_sell\_weight},
$$

$$
\text{effective\_supply}(t) = \text{TGE\_total} + \bigl(\text{postTGE\_unlocked}(t)-\text{postTGE\_unlocked}(0)\bigr)\times\bigl(\text{avg\_sell\_weight}\times(1-\text{buyback\_rate})\bigr).
$$

2. **Dynamic Price Evolution** uses a jump-diffusion process:

- **Diffusion (GBM)**:
  
  $$
  P_{\text{diff}}(t) = \exp\Bigl((\mu-\tfrac{1}{2}\sigma^2)\,\Delta t \;+\;\sigma\sqrt{\Delta t}\,Z_t\Bigr),\quad Z_t\sim N(0,1).
  $$

- **Jump**:
  
  $$
  J_t =
  \begin{cases}
  1 + \text{jump\_size}, & \text{with prob } (\text{jump\_intensity}\times\Delta t),\\
  1, & \text{otherwise}.
  \end{cases}
  $$

  $\text{jump\_size} \sim N(\text{jump\_mean}, \text{jump\_std})$.

- **Dynamic Multiplier**:
  
  $$
  P_{\text{jump}}(t) = P_{\text{jump}}(t-1) \times P_{\text{diff}}(t)\times J_t,\quad P_{\text{jump}}(0)=1.
  $$

3. **Final Token Price**:

$$
\text{Final\_Price}(t) = \text{price}(t)\times P_{\text{jump}}(t).
$$

Meanwhile, user retention $A(t)$ updates each step:

$$
A_{t+1} = A_{t} + \Bigl[\alpha (1-A_{t}) \;-\;\beta\Bigl(1+\theta\,\frac{p(t)-p_0}{p_0}\Bigr)A_{t}\Bigr]\Delta t \;+\;\epsilon_t,
$$

where $\epsilon_t\sim N(0,\sigma_{de}\,\Delta t)$.

_Sources: [Investopedia (GBM)](https://www.investopedia.com/terms/g/geometric-brownian-motion.asp), [ScienceDirect (Jump Diffusion)](https://www.sciencedirect.com/topics/engineering/jump-diffusion)._

---

## airdrop_policy.py

We support:

- **Linear**  
  $$\text{tokens} = \text{factor}\times \text{airdrop\_points}$$

- **Exponential**  
  $$\text{tokens} = \text{factor}\times\Bigl(e^\frac{\text{airdrop\_points}}{\text{scaling}} -1\Bigr)$$

- **Tiered Constant**  
  \[
  \text{tokens}=
  \begin{cases}
  0.1,\; & \text{if } \text{airdrop\_points}<0.2\\
  0.4,\; & \text{if } 0.2\le \text{airdrop\_points}<0.6\\
  1.0,\; & \text{if } \text{airdrop\_points}\ge0.6
  \end{cases}
  \]

- **Tiered Linear**  
  \[
  \text{tokens} = 
  \begin{cases}
  1.0\times \text{airdrop\_points}, & \text{if } \text{airdrop\_points}\le 0.2\\
  1.0\times 0.2 + 1.5\times(\text{airdrop\_points}-0.2), & \text{if } 0.2<\text{airdrop\_points}\le 0.6\\
  1.0\times 0.2 + 1.5\times 0.4 + 2.0\times(\text{airdrop\_points}-0.6), & \text{if } \text{airdrop\_points}>0.6
  \end{cases}
  \]

- **Tiered Exponential**  
  $$\text{tokens} = \sum_{\text{tiers}} \text{factor}\times\Bigl(e^{\frac{\Delta \text{airdrop\_points}}{\text{scaling}}}-1\Bigr)$$

_References: Vertex’s [Token Unlocks](https://messari.io/project/vertex-protocol/token-unlocks), [Crypto Economics Handbook](https://www.cryptoeconomicshandbook.org/token-distribution)._

---

## preTGE_rewards.py

- **dYdX Retro**  
- **Vertex Maker/Taker**  
- **Jupiter Volume Tier**  
- **Aevo Farm Boost**  

We removed Helix and MMR standalones; **Game MMR** is now part of **GenericPreTGERewardPolicy** as our custom approach (see earlier references: [dYdX Community](https://forum.dydx.community)).

---

## postTGE_rewards.py & postTGE_rewards_policy.py

**postTGE_rewards.py** simulates vesting + price evolution in monthly steps. Each month:
1. Compute baseline price from supply/demand (see vesting).
2. Apply jump-diffusion multiplier.
3. Retention: update $A(t)$, the fraction of active users, feeding back into next step’s price.

**postTGE_rewards_policy.py** implements the engagement multiplier:

\[
\text{multiplier} = 1 + \gamma\Bigl(\frac{\text{active\_days}}{T}\Bigr)^\delta
\]

where \(\gamma,\delta\) are user-defined parameters.

_Sources: [GBM references from Investopedia](https://www.investopedia.com/terms/g/geometric-brownian-motion.asp), [Jump Diffusion from ScienceDirect](https://www.sciencedirect.com/topics/engineering/jump-diffusion), [Vertex Docs](https://vertexprotocol.com/docs/trade-and-earn)._

---

## Plotting Overview

- **Histogram of TGE Token Distribution**: Plots TGE tokens for each pre-TGE & airdrop combo.  
- **Vesting Schedule Plot**: Stackplot of unlocked tokens by group + dashed line for total unlocked.  
- **Price Evolution Overlay Grid**: For each pre-TGE + airdrop combo, we overlay post-TGE price curves (Baseline, High Vol, Low Vol, Agg. Buyback). Grey bars show Baseline’s active fraction. Grid layout is customizable (rows × columns).

---

## Instructions to Run

- `python main.py` from the root directory.
- Modify simulation params in **main.py** (like user count, supply, or jump intensities).
- Close each plot window to see the next.
- Save plots as you like.

_Replace your existing README.md with this content for a concise summary of the updated file structure and logic._ 
