import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

from airdrop_policy import (
    LinearAirdropPolicy,
    ExponentialAirdropPolicy,
    TieredLinearAirdropPolicy,
    TieredConstantAirdropPolicy,
    TieredExponentialAirdropPolicy
)
from preTGE_rewards import (
    DydxRetroTieredRewardPolicy,
    VertexMakerTakerRewardPolicy,
    JupiterVolumeTierRewardPolicy,
    AevoFarmBoostRewardPolicy,
    GenericPreTGERewardPolicy
)
from postTGE_rewards_policy import (
    GenericPostTGERewardPolicy,
    EngagementMultiplierPolicy
)
# No longer use PostTGERewardsSimulator directly.
from plot_helper import (
    plot_airdrop_distribution_grid,
    plot_vesting_schedule,
    plot_price_evolution_overlay,
    plot_avg_price_heatmap,
    plot_avg_price_evolution_overlay
)
from simulation import MonteCarloSimulation
from users import RegularUser, SybilUser
from activity_stats import generate_stats

def run_simulation_for_combo(combo_name, num_users, total_supply, preTGE_steps, simulation_horizon,
                             ad_policy, pre_policy, post_policy, post_policy_config,
                             base_price, elasticity, buyback_rate, alpha=0.5,
                             airdrop_allocation_fraction=0.25):
    # Define the demand series (raw demand values) as provided from Forgd.
    demand_values = np.array([
        45, 25, 10, 12, 6, 7, 1,
        5, 2, 1, 1, 5, 15, 10,
        12, 14, 43, 50, 51, 43,
        66, 68, 70, 71, 77, 73, 73, 73,
        69, 63, 61, 63, 65, 62, 51, 52, 47,
        42, 34, 32, 38, 32, 31, 32, 37, 33,
        28, 25, 22, 23, 21, 20, 18, 19, 17,
        19, 21, 12, 10, 13
    ], dtype=float)
    
    # Instantiate MonteCarloSimulation. All phases (pre-TGE, TGE, and dynamic post-TGE evolution)
    # are now computed inside run().
    sim = MonteCarloSimulation(
        num_users=num_users,
        total_supply=total_supply,
        preTGE_steps=preTGE_steps,
        simulation_horizon=simulation_horizon,
        airdrop_policy=ad_policy,
        preTGE_rewards_policy=pre_policy,
        postTGE_rewards_policy=post_policy,
        airdrop_allocation_fraction=airdrop_allocation_fraction,
        initial_price=base_price,
        buyback_rate=buyback_rate,
        elasticity=elasticity,
        demand_series=demand_values
    )
    # Run the full simulation.
    sim_results = sim.run()  # This returns a dictionary with dynamic price evolution.
    
    # Apply a post-TGE reward policy (engagement multiplier) to each RegularUser.
    post_reward_policy = GenericPostTGERewardPolicy()
    active_days = sim_results["active_fraction_history"][-1] * simulation_horizon \
                  if len(sim_results["active_fraction_history"]) > 0 else simulation_horizon
    for user in sim.user_pool.users:
        if isinstance(user, RegularUser):
            post_reward_policy.apply_rewards(user, active_days)
    
    # IMPORTANT: change the key for prices to "prices" so that plot_helper works correctly.
    return combo_name, {
        "TGE_total": sim_results["scaled_TGE_total"],
        "months": sim_results["months"],
        "prices": sim_results["dynamic_prices"],  # rename to "prices"
        "active_fraction_history": sim_results["active_fraction_history"],
        "TGE_tokens": [user.tokens for user in sim.user_pool.users],
        "unlocked_history": sim_results["unlocked_history"],
        "total_unlocked_history": sim_results["total_unlocked_history"],
        "distribution": sim_results["distribution"],
        "combo_label": combo_name
    }

if __name__ == '__main__':
    # Define airdrop conversion policies.
    airdrop_policies = [
        ("Linear", LinearAirdropPolicy()),
        ("Exponential", ExponentialAirdropPolicy()),
        ("Tiered Linear", TieredLinearAirdropPolicy()),
        ("Tiered Constant", TieredConstantAirdropPolicy()),
        ("Tiered Exponential", TieredExponentialAirdropPolicy())
    ]
    
    # Define pre-TGE rewards policies.
    preTGE_policies = [
        ("dYdX Retro", DydxRetroTieredRewardPolicy()),
        ("Vertex Maker/Taker", VertexMakerTakerRewardPolicy()),
        ("Jupiter Volume Tier", JupiterVolumeTierRewardPolicy()),
        ("Aevo Farm Boost", AevoFarmBoostRewardPolicy()),
        ("Game-like MMR", GenericPreTGERewardPolicy())
    ]

    # Define post-TGE rewards policies.
    postTGE_policies = [
        ("Generic", GenericPostTGERewardPolicy()),
        ("Engagement Multiplier", GenericPostTGERewardPolicy(engagement_policy=EngagementMultiplierPolicy()))
    ]
    
    # Define post-TGE rewards configurations.
    postTGE_scenarios = [
        ("Baseline", {"sigma": 0.05, "jump_intensity": 0.1, "jump_mean": -0.05, "jump_std": 0.1}),
        ("High Volatility", {"sigma": 0.1, "jump_intensity": 0.15, "jump_mean": -0.1, "jump_std": 0.15}),
        ("Low Volatility", {"sigma": 0.03, "jump_intensity": 0.05, "jump_mean": -0.03, "jump_std": 0.05}),
        ("Aggressive Buyback", {"sigma": 0.05, "jump_intensity": 0.1, "jump_mean": -0.05, "jump_std": 0.1, "buyback_rate": 0.3})
    ]
    
    # Simulation parameters.
    num_users = 40_000
    total_supply = 1_000_000_000
    airdrop_allocation_percentage = 0.25
    preTGE_steps = 50
    simulation_horizon = 60  # months
    base_price = 0.2
    buyback_rate = 0.2
    alpha = 0.1
    elasticity = 0.5
    
    results = {}
    tasks = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for pre_name, pre_policy in preTGE_policies:
            for ad_name, ad_policy in airdrop_policies:
                for reward_name, reward_policy in postTGE_policies:
                    for scenario_name, scenario_config in postTGE_scenarios:
                        combo_name = f"{pre_name} + {ad_name} + {reward_name} + {scenario_name}"
                        print(f"Submitting simulation for: {combo_name}")
                        # Use the buyback_rate from config if provided.
                        post_buyback_rate = scenario_config.get("buyback_rate", buyback_rate)
                        task = executor.submit(
                            run_simulation_for_combo,
                            combo_name,
                            num_users, total_supply, preTGE_steps, simulation_horizon,
                            ad_policy, pre_policy, reward_policy, scenario_config,
                            base_price, elasticity, post_buyback_rate,
                            alpha=alpha,
                            airdrop_allocation_fraction=airdrop_allocation_percentage
                        )
                        tasks.append(task)
        
        for future in concurrent.futures.as_completed(tasks):
            combo_name, res = future.result()
            results[combo_name] = res
            print(f"Completed simulation for: {combo_name}")
    
    # (Plotting code below remains the same.)
    
    # Plot Histogram of TGE Token Distribution
    baseline_results = {}
    for combo_name, data in results.items():
        parts = combo_name.split(" + ")
        if len(parts) == 4 and parts[2] == "Generic" and parts[3] == "Baseline":
            key = " + ".join(parts[:2])
            if key not in baseline_results:
                baseline_results[key] = data
    plt.figure(figsize=(12, 8))
    for combo_name, data in baseline_results.items():
        tokens = np.array(data["TGE_tokens"])
        finite_tokens = tokens[np.isfinite(tokens)]
        if finite_tokens.size > 0:
            plt.hist(finite_tokens, bins=30, alpha=0.5, label=combo_name,
                     histtype='step', linewidth=1.5)
    plt.xlabel("Tokens Assigned at TGE")
    plt.ylabel("Number of Users")
    plt.title("Histogram of TGE Tokens Distribution (Pre-TGE & Airdrop Policies)")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True)
    plt.show(block=True)

    # Plot Vesting Schedule and Total Supply Over Time
    chosen = "dYdX Retro + Linear + Generic + Baseline"
    if chosen in results:
        chosen_result = results[chosen]
        from plot_helper import plot_vesting_schedule
        plot_vesting_schedule(chosen_result["months"],
                              chosen_result["unlocked_history"],
                              chosen_result["total_unlocked_history"])
    else:
        print(f"Chosen simulation '{chosen}' not found.")
    
    # TGE Distribution Grid Plot
    tge_results = {}
    for combo_name, data in results.items():
        parts = combo_name.split(" + ")
        if len(parts) == 4 and parts[2] == "Generic" and parts[3] == "Baseline":
            key = f"{parts[0]} + {parts[1]}"
            tge_results[key] = data
    pre_labels = [p[0] for p in preTGE_policies]
    ad_labels = [a[0] for a in airdrop_policies]
    from plot_helper import plot_airdrop_distribution_grid
    plot_airdrop_distribution_grid(tge_results, pre_labels, ad_labels)
    
    # Price Evolution Heatmaps
    post_reward_labels = [p[0] for p in postTGE_policies]
    scenario_labels = [s[0] for s in postTGE_scenarios]
    from plot_helper import plot_avg_price_heatmap, plot_price_evolution_overlay, plot_avg_price_evolution_overlay
    for reward_label in post_reward_labels:
        for scenario_label in scenario_labels:
            heatmap = np.zeros((len(pre_labels), len(ad_labels)))
            for i, pre_name in enumerate(pre_labels):
                for j, ad_name in enumerate(ad_labels):
                    combo_name = f"{pre_name} + {ad_name} + {reward_label} + {scenario_label}"
                    if combo_name in results:
                        final_price = results[combo_name]["prices"][-1]
                        heatmap[i, j] = final_price
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap="viridis", aspect="auto")
            plt.colorbar(label="Final Token Price (USD)")
            plt.xticks(ticks=np.arange(len(ad_labels)), labels=ad_labels, rotation=45)
            plt.yticks(ticks=np.arange(len(pre_labels)), labels=pre_labels)
            plt.title(f"Heatmap: Final Price (Month 60)\n{reward_label} + {scenario_label}", fontsize=12)
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    plt.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="w")
            plt.tight_layout()
            plt.show()
    plot_avg_price_heatmap(results, pre_labels, ad_labels, post_reward_labels, scenario_labels)

    plot_price_evolution_overlay(results, pre_labels, ad_labels, post_reward_labels, scenario_labels,
                                 max_rows_per_fig=2, max_cols_per_fig=3)
    plot_avg_price_evolution_overlay(results, pre_labels, ad_labels, post_reward_labels, scenario_labels,
                                     max_rows_per_fig=5, max_cols_per_fig=5)
