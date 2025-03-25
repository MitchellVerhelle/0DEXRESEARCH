# DEX RESEARCH
#### We're researching the optimal pre-TGE rewards program, TGE airdrop policy, and post-TGE rewards program to maximize long-term token growth while minimizing chances of token value depreciation.

### The model is 3 stages:
- Pre TGE
- Airdrop Event
- Post TGE

##### We assume that the program has only one large genesis airdrop event.

### user.py

##### Users are currently split into two subcategories
- RegularUser
- SybilUser

##### Sybils will create multiple accounts, do little pre-TGE farming, and have minimal interaction with the platofrm before eventually selling their tokens, post Airdrop. We want to minimize tokens that go to Sybils.

##### RegularUsers are users who may or may not interact with the site. Unline Sybils, RegularUsers don't create accounts wiht the sole purpose of leaving, but instead will leave if they no longer see value in staying. We aim to maximize RegularUser engagement.
##### Design choice: I've used a simple differential equation to model RegularUser engagement pre-TGE for farming reward points. This model abstracts away the difficulties of modeling how exactly users gain rewards. First I assume RegularUsers are subject to 3 main sub-categories: "small", "medium", and "large". These refer to the user's capital. Small users have less capital so they will naturally interact less frequently with the testnet or dApps to farm tokens. Medium users have a medium amount of capital, and large users a large amount. The point is to differentiate how users interact with the site.

##### SybilUser endowment is low at lambda parameter 0.5 (e.g. [Airdrop Vulnerabilities Explained](https://blog.goodaudience.com/how-airdrops-can-be-sybil-attacked-8721480f3e5e)) whereas RegularUser endowment depends on user size is lambda parameter 1 for small, 3 for medium, 5 for large. All are poisson random variables to represent some users someitmes having larger endowments or smaller endowments depending on their respective lambdas.

### user_pool.py

##### A UserPool refers to the population of users. Backed by research, I split users into Sybil or Regular based on the estimated proportion of sybil wallet accounts, which is suprisingly high at 30%. All sybil users are assumed to be small, so they have a small poisson parameter for determining their wealth. I've also found that, for RegularUsers, 10% are "large", 30% "medium", and 60% "small". This allows me to generate a model population from which I initialize user wealth values.

##### Backed by resaerch, user wealth is assumed to follow a log-normal distribution, form which I use the following distributions: Small: LogNormal(6,1.5), Medium: LogNormal(7, 1.2), Large: LogNormal(8, 1.0). Then for SybilUsers, I generate from LogNormal(5, 1.0), since I am assuming that their accounts are proportionately smaller, as they will clearly have less time to interact with testnet/dApps, on average.

##### Retention and active_user model:
I model user retention as a differential‐equation model that factors in both a baseline activation/churn and an adjustment for token price (which proxies the opportunity cost)
The proportion of active users \( A(t) \) can be modeled like follows:
\[
    \frac{dA}{dt}(t) = \alpha(N-A(t))-\beta A(t)
\]
where
- $ A(t) $ is the number of active users at time \( t \),
- \( N \) is the total number of potential users, currently set to 10_000, but normalized to 1 as you'll read below,
- \( \alpha \) is the activation rate,
- \( \beta \) is the deactivation rate (churn rate).

Source: [Bass Diffusion Model](https://en.wikipedia.org/wiki/Bass_diffusion_model)

I work with a normalized version of this model and extend it to incorporate the effect of opportunity cost, which I assume is directly related to the token price \( p(t) \) on churn:

\[
    \frac{dA}{dt}(t) = \alpha(1-A(t))-\beta\left(1+\theta\frac{p(t)-p_0}{p_0}\right)A(t) + \sigma \eta(t)
\]
where
- \( A(t) \in [0,1] \) is the proportion of active users at time \( t \),
- \( p(t) \) is the price at time \( t \),
- \( p_0 \) is a "baseline" token price (like base price at TGE),
- \( \theta \) is a sensitivity parameter,
- \( \sigma \) scales random fluctuations,
- \( \eta(t) \) is a white-noise term.

Source: [Churn modeling in online services](https://www.sciencedirect.com/science/article/pii/S0167923606000470)

I use Euler-Maruyama steps to discretize this equation in my code. \( \epsilon_t \sim \mathcal{N}(0,\sigma dt) \)

\[
    A_{t+1} = A_t + \left[\alpha(1-A_t) - \beta\left(1+\theta\frac{p(t)-p_0}{p_0}\right)A_t \right]dt + \epsilon_t
\]

### simulation.py

##### Simulation.py runs in 3 main stages, as discussed above. Pre-TGE, TGE, Post-TGE.
##### From UserPool, I run a stepall function which steps each user in the user pool to interact or not interact with testnet.

#### Assumptions
##### Pre-TGE
- Larger users are more likely to interact with the testnet in larger proportion if the reward policy favors larger users. Larger users contributing to the platform is beneficial for both parties, but could be harmful if the large user decides to leave, so not too much wealth should be allocated to large users.
- Time steps are in months, to stay consistent with Forgd, which is mostly monthly vesting schedules.
- Users interact based on a differential equation which specifies some potential, an amount users are willing to spend, a rate of loss delta representing opportunity cost, and an interaction_factor which represents how frequently the user will contribute to the platform. larger users have larger endowments, interaction_factors, but also opportunity costs, since they will want larger rewards.
- Trading volume is estimated and monthly for each user size/type.
- Users are rewarded points depending on each preTGE_rewards.py policy.
##### TGE
- Users are rewarded tokens depending on each airdrop policy.

- **dYdX Retroactive (Tiered Fixed Reward) Policy**  
  Based on total trading volume (in USD), a fixed reward (points) is assigned. For example:
  - volume < 1,000 USD  ⇒ 310 points (base deposit bonus)
  - 1,000 ≤ volume < 10,000 USD  ⇒ 1,163 points
  - 10,000 ≤ volume < 100,000 USD  ⇒ 2,500 points
  - 100,000 ≤ volume < 1,000,000 USD  ⇒ 6,414 points
  - volume ≥ 1,000,000 USD  ⇒ 9,530 points

- **Vertex Maker/Taker Reward Policy**  
  Users earn points based on:
  - Maker volume weighted at 37.5%
  - Taker volume weighted at 37.5%
  - A Q-score (liquidity quality) weighted at 25%
  
  Additionally, a referral bonus (e.g., 25% of the referees’ points) is added.

- **Jupiter Volume Tier Reward Policy**  
  Implements a tiered, piecewise constant reward system based on swap volume. For example:
  - volume ≥ 1,000 USD  ⇒ 50 points
  - volume ≥ 29,000 USD  ⇒ 250 points
  - volume ≥ 500,000 USD  ⇒ 3,000 points
  - volume ≥ 3,000,000 USD  ⇒ 10,000 points
  - volume ≥ 14,000,000 USD  ⇒ 20,000 points

- **Aevo Boosted Volume Reward Policy**  
  Applies a base boost (scaling from 1× up to a maximum boost based on trailing volume) and a probabilistic “lucky boost” to each trade. Default lucky boost probabilities might be:
  - 10× boost with 10% chance
  - 50× boost with 2.5% chance
  - 100× boost with 1% chance

- **Helix (Injective) Loyalty Points Reward Policy**  
  Rewards users based on:
  - Trading volume (weighted linearly)
  - Diversity bonus (number of unique markets traded)
  - Loyalty bonus (active days multiplied by volume)

- **Game-like (MMR) Reward Policy**  
  Uses a game-like system inspired by Match-Making Rating (MMR), where rewards are determined by:
  - A base number of points
  - Bonus points proportional to the win rate (wins vs. losses)
  - Additional bonus for consistency (consecutive days of activity)

- **Custom / Generic Reward Policy**  
  A flexible reward policy that accepts a custom function.

##### Price
Price starts at a constnat value. [Forgd](https://app.forgd.com/academy/how-to-guides/build-tokenomics-and-protocol-value-flows/determining-your-valuation) does a good job explaining why these dynamics exist. (Venture backing, no community trading yet, investments, hype, ethos, etc. all contribute to pre-TGE price.) [This site called multiversx](https://multiversx.com/blog/maiar-dex-listings-price-discovery) gives reasonable parameter initialization for pre-TGE price at about $0.20-$1.00 USD.

Then price starts evolving at TGE, time 0. We model price as follows:

1. **Baseline Supply/Demand Price**

The baseline price at time *t* is given by:
\[
\text{price}(t) = \text{base\_price} \times \left(\frac{TGE\_total}{\text{combined\_supply}(t)}\right)^{\text{elasticity}}
\]
where:
- **TGE_total**: The fixed token amount distributed at the Token Generation Event (TGE).  
- **base_price**: The initial token price at TGE.  
- **elasticity**: A parameter representing how sensitive the token price is to changes in supply. A higher elasticity means that a small change in supply results in a larger change in price.  
- **combined_supply(t)**: A weighted combination of the circulating and effective supply at time *t*, defined as:
  \[
  \text{combined\_supply}(t) = \alpha \times \text{circulating\_supply}(t) + (1 - \alpha) \times \text{effective\_supply}(t)
  \]
  - **circulating_supply(t)**: Calculated as:
    \[
    \text{circulating\_supply}(t) = TGE\_total + \left(\text{postTGE\_unlocked}(t) - \text{postTGE\_unlocked}(0)\right) \times \text{avg\_sell\_weight}
    \]
  - **effective_supply(t)**: Adjusted for tokens removed from circulation (e.g., via buybacks), computed as:
    \[
    \text{effective\_supply}(t) = TGE\_total + \left(\text{postTGE\_unlocked}(t) - \text{postTGE\_unlocked}(0)\right) \times \left(\text{avg\_sell\_weight} \times (1 - \text{buyback\_rate})\right)
    \]
- **avg_sell_weight**: The average fraction of unlocked tokens that enter the market as sell orders. This can be computed from user data or provided via a distribution.  
- **buyback_rate**: The fraction of unlocked tokens effectively removed from circulation through buybacks or burns.  
- **alpha**: A weighting parameter between the raw circulating supply and the effective supply.

This supply/demand model operates under the idea that as more tokens become available (via vesting), the price adjusts inversely. ([Investopedia on Supply and Demand](https://www.investopedia.com/terms/s/supply-demand.asp)).

2. **Dynamic Price Evolution (Jump-Diffusion Process)**

To capture market volatility and sudden shocks, a jump-diffusion process is applied. The dynamic multiplier \( P_{\text{jump}}(t) \) is defined recursively:

- At TGE (time 0):
  \[
  P_{\text{jump}}(0) = 1.0
  \]
- For \( t \ge 1 \):
  \[
  P_{\text{jump}}(t) = P_{\text{jump}}(t-1) \times \exp\left((\mu - 0.5\sigma^2)\Delta t + \sigma \sqrt{\Delta t} \, Z_t\right) \times J_t
  \]
  where:
  - **\(\mu\)**: The drift rate, representing the average return per time unit.  
  - **\(\sigma\)**: The volatility parameter, indicating the randomness in price movements.  
  - **\(\Delta t\)**: The time step (e.g., one month).  
  - **\(Z_t\)**: A random variable drawn from a standard normal distribution (\( Z_t \sim N(0,1) \)).  
  - **\(J_t\)**: The jump component, defined as:
    \[
    J_t = 
    \begin{cases}
    1 + \text{jump\_size}, & \text{with probability } \text{jump\_intensity} \times \Delta t, \\
    1, & \text{otherwise},
    \end{cases}
    \]
    where **jump_size** is drawn from a normal distribution with mean **jump_mean** and standard deviation **jump_std**.

The jump-diffusion process combines a geometric Brownian motion ([Investopedia on Geometric Brownian Motion](https://www.investopedia.com/terms/g/geometric-brownian-motion.asp))—with a jump component that captures abrupt market events ([ScienceDirect on Jump-Diffusion Models](https://www.sciencedirect.com/topics/engineering/jump-diffusion)). I chose to abstract away more sophisticated market dynamics because I assumed it would suffice for the goal of discovering optimal PreTGE/TGE/PostTGE policies. However, it could make for an interesting extention to consider more advanced frameworks for price evolution depending on assumptions about user behavior.

3. **Final Token Price**

The final simulated token price at time *t* is:
\[
\text{Final Price}(t) = \text{Supply/Demand Price}(t) \times P_{\text{jump}}(t)
\]
This means the price is determined by both the fundamental supply/demand mechanics and the stochastic jump-diffusion dynamics.

**Parameter Summary**

- **TGE_total**: Total tokens available at the Token Generation Event.
- **total_unlocked_history**: Time series data showing how many tokens unlock over time due to vesting.
- **users**: User data used to compute average sell weight if a distribution is not provided.
- **base_price**: Initial token price at TGE.
- **elasticity**: Sensitivity of the token price to changes in supply.
- **buyback_rate**: Fraction of unlocked tokens removed from circulation via buybacks.
- **alpha**: Weighting between circulating supply and effective supply.
- **mu**: Drift rate of the diffusion component (average return per time step).
- **sigma**: Volatility in the diffusion process.
- **jump_intensity**: Likelihood (per time step) of a jump event occurring.
- **jump_mean**: Average jump size (expressed as a fraction, e.g., -0.05 for a 5% drop).
- **jump_std**: Standard deviation of the jump size.
- **distribution**: Optional dictionary with keys ("small", "medium", "large", "sybil") summing to 100, used to compute the average sell weight.

### airdrop_policy.py
##### The airdrop policy takes a few forms, of which we researched prior to writing this file.
- Linear (https://messari.io/project/vertex-protocol/token-unlocks)
\( \text{tokens} = \text{factor} \times \text{airdrop\_points} \)

- Exponential (https://medium.com/@blockchaintokenomics/exponential-airdrop-models-2021)
\( \text{tokens} = \text{factor} \times \Bigl(e^{\frac{\text{airdrop\_points}}{\text{scaling}}} - 1\Bigr) \)

- Tiered Constant (https://tokenomicslab.org/airdrop-mechanisms)
\[
\text{tokens} =
\begin{cases}
0.1, & \text{if } \text{airdrop\_points} < 0.2, \\
0.4, & \text{if } 0.2 \le \text{airdrop\_points} < 0.6, \\
1.0, & \text{if } \text{airdrop\_points} \ge 0.6.
\end{cases}
\]

- Tiered Linear (https://consensys.net/blog/blockchain-explained/token-distribution-models)
\[
\text{tokens} = 
\begin{cases}
1.0 \times \text{airdrop\_points}, & \text{if } \text{airdrop\_points} \le 0.2, \\
0.2 \times 1.0 + 1.5 \times (\text{airdrop\_points} - 0.2), & \text{if } 0.2 < \text{airdrop\_points} \le 0.6, \\
0.2 \times 1.0 + 0.4 \times 1.5 + 2.0 \times (\text{airdrop\_points} - 0.6), & \text{if } \text{airdrop\_points} > 0.6.
\end{cases}
\]

- Tiered Exponential (https://www.cryptoeconomicshandbook.org/token-distribution)
\[
\text{tokens} = \sum_{\text{tiers}} \text{factor} \times \Bigl(e^{\frac{\Delta \text{airdrop\_points}}{\text{scaling}}} - 1\Bigr)
\]

### postTGE-rewards.py
- This file currently just specifies the vesting schedules, cliffs, unlocks, and allocation percentages to each group. We were assuming constraints on non-community endowments, leaving about 45% of the allocation to community. We decided on this schedule and use the 20% in treasury/strategic reserve for demand drivers like token buybacks, as well as 5% to long-term rewards. The main purpose in terms of modeling here is to influence demand for the token during times of high expected sell-pressure to stabilize token price.
- There are plans to incorporate more advanced post-TGE rewards plans and integrate them into the user step behaviors.
- Currently, users interact with the post-TGE trading based on the vesting period unlock schedule.

### main.py
- This program runs the simulation with threads based on your computer's hardware capabilities.

**Parameters**
- **num_users**: 10_000 (This is the whole user-pool number of users. It could be made into a process in UserPool for "active" users).
    total_supply = 100_000_000
    preTGE_steps = 50
    simulation_horizon = 60  # months
    base_price = 10.0
    buyback_rate = 0.2  # 20% of additional unlocked tokens are removed. Should be same as or less than post-TGE token allocation to "Strategic Reserve/Treasury" for now.
    alpha = 0 # combined_supply = alpha * circulating_supply + (1 - alpha) * effective_supply
    elasticity = 1.0 # price(t) = base_price * (TGE_total / combined_supply(t))^elasticity

## Instructions
- **To run, enter "python main.py" in the root directory. Then wait for the program to run. As you see the graphs, click 'x' to see the next graph until the program finishes running. Save each graph as desired.**
- **Modify parameters in main.py.**
- **Current parameters:**
- - 
- **If it gives an error, try removing the threads or setting thread-count manually to 2 in main.py. If still errors, ensure your python environment is up to date.**