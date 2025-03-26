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
from postTGE_rewards import PostTGERewardsSimulator
from postTGE_rewards_policy import GenericPostTGERewardPolicy
from plot_helper import plot_airdrop_distribution_grid, plot_vesting_schedule, plot_price_evolution_overlay
from simulation import MonteCarloSimulation
from users import RegularUser, SybilUser
from activity_stats import generate_stats

def run_simulation_for_combo(combo_name, num_users, total_supply, preTGE_steps, simulation_horizon,
                             ad_policy, pre_policy, post_policy_config, base_price, elasticity, buyback_rate, alpha=0.5,
                             airdrop_allocation_fraction=0.25):
    sim = MonteCarloSimulation(
        num_users=num_users,
        total_supply=total_supply,
        preTGE_steps=preTGE_steps,
        simulation_horizon=simulation_horizon,
        airdrop_policy=ad_policy,
        preTGE_rewards_policy=pre_policy,
        airdrop_allocation_fraction=airdrop_allocation_fraction,
        initial_price=base_price
    )
    sim_results = sim.run()  # Returns a dictionary.
    TGE_total = sim_results["scaled_TGE_total"]
    months = sim_results["months"]
    total_unlocked_history = sim_results["total_unlocked_history"]
    unlocked_history = sim_results["unlocked_history"]
    dist = sim_results["distribution"]
    active_fraction_history = sim_results["active_fraction_history"]

    # Use the PostTGERewardsSimulator for dynamic price evolution.
    simulator = PostTGERewardsSimulator(
        TGE_total=TGE_total,
        total_unlocked_history=total_unlocked_history,
        users=sim.user_pool.users,
        base_price=base_price,
        elasticity=elasticity,
        buyback_rate=buyback_rate,
        alpha=alpha,
        mu=post_policy_config.get("mu", 0.0),
        sigma=post_policy_config.get("sigma", 0.05),
        jump_intensity=post_policy_config.get("jump_intensity", 0.1),
        jump_mean=post_policy_config.get("jump_mean", -0.05),
        jump_std=post_policy_config.get("jump_std", 0.1),
        distribution=dist
    )
    prices = simulator.simulate_price_evolution()

    # Apply a post-TGE reward policy (engagement multiplier) to each RegularUser.
    post_reward_policy = GenericPostTGERewardPolicy()
    active_days = active_fraction_history[-1] * simulation_horizon if len(active_fraction_history) > 0 else simulation_horizon
    for user in sim.user_pool.users:
        if isinstance(user, RegularUser):
            post_reward_policy.apply_rewards(user, active_days)
    
    return combo_name, {
        "TGE_total": TGE_total,
        "months": months,
        "total_unlocked_history": total_unlocked_history,
        "unlocked_history": unlocked_history,
        "prices": prices,
        "TGE_tokens": [user.tokens for user in sim.user_pool.users],
        "distribution": dist,
        "active_fraction_history": active_fraction_history,
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
    
    # Define post-TGE rewards configurations.
    postTGE_configs = [
        ("Baseline", {"mu": 0.0, "sigma": 0.05, "jump_intensity": 0.1, "jump_mean": -0.05, "jump_std": 0.1}),
        ("High Volatility", {"mu": 0.0, "sigma": 0.1, "jump_intensity": 0.15, "jump_mean": -0.1, "jump_std": 0.15}),
        ("Low Volatility", {"mu": 0.0, "sigma": 0.03, "jump_intensity": 0.05, "jump_mean": -0.03, "jump_std": 0.05}),
        ("Aggressive Buyback", {"mu": 0.0, "sigma": 0.05, "jump_intensity": 0.1, "jump_mean": -0.05, "jump_std": 0.1, "buyback_rate": 0.3})
    ]
    
    # Simulation parameters.
    num_users = 10_000
    total_supply = 100_000_000
    airdrop_allocation_percentage = 0.15
    preTGE_steps = 50
    simulation_horizon = 60  # months
    base_price = 10.0
    buyback_rate = 0.2
    alpha = 0.1
    elasticity = 1.0
    
    results = {}
    tasks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for pre_name, pre_policy in preTGE_policies:
            for ad_name, ad_policy in airdrop_policies:
                for post_name, post_config in postTGE_configs:
                    combo_name = f"{pre_name} + {ad_name} + {post_name}"
                    print(f"Submitting simulation for: {combo_name}")
                    post_buyback_rate = post_config.get("buyback_rate", buyback_rate)
                    task = executor.submit(
                        run_simulation_for_combo,
                        combo_name,
                        num_users, total_supply, preTGE_steps, simulation_horizon,
                        ad_policy, pre_policy, post_config, base_price, elasticity, post_buyback_rate,
                        alpha=alpha,
                        airdrop_allocation_fraction=airdrop_allocation_percentage
                    )
                    tasks.append(task)
        
        for future in concurrent.futures.as_completed(tasks):
            combo_name, res = future.result()
            results[combo_name] = res
            print(f"Completed simulation for: {combo_name}")
    
    # ---------------------------
    # Plot Histogram of TGE Token Distribution
    # ---------------------------
    # Use only the pre-TGE + airdrop combinations (ignoring post-TGE variations).
    baseline_results = {}
    for combo_name, data in results.items():
        parts = combo_name.split(" + ")
        if len(parts) == 3:
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

    # ---------------------------
    # Plot Vesting Schedule and Total Supply Over Time
    # ---------------------------
    # For one chosen simulation (e.g., "dYdX Retro + Linear + Baseline"), plot the vesting schedule.
    chosen = "dYdX Retro + Linear + Baseline"
    if chosen in results:
        chosen_result = results[chosen]
        from plot_helper import plot_vesting_schedule
        plot_vesting_schedule(chosen_result["months"], chosen_result["unlocked_history"],
                                chosen_result["total_unlocked_history"])
    else:
        print(f"Chosen simulation '{chosen}' not found.")
    
    # ---------------------------
    # TGE Distribution Grid Plot
    # ---------------------------
    # For TGE distribution, use only the preTGE + airdrop policies.
    # Create a dictionary keyed by "<pre_policy> + <ad_policy>" from results that have "Baseline" post config.
    tge_results = {}
    for combo_name, data in results.items():
        parts = combo_name.split(" + ")
        if len(parts) == 3 and parts[2] == "Baseline":
            key = f"{parts[0]} + {parts[1]}"
            # We assume all postTGE variants yield the same TGE distribution.
            tge_results[key] = data

    pre_labels = [p[0] for p in preTGE_policies]
    ad_labels = [a[0] for a in airdrop_policies]
    from plot_helper import plot_airdrop_distribution_grid
    plot_airdrop_distribution_grid(tge_results, pre_labels, ad_labels)
    
    # ---------------------------
    # Price Evolution Heatmaps
    # ---------------------------
    # For each postTGE config, build a separate heatmap (rows: preTGE policies; columns: airdrop policies)
    for post_name, _ in postTGE_configs:
        heatmap = np.zeros((len(pre_labels), len(ad_labels)))
        for i, pre_name in enumerate(pre_labels):
            for j, ad_name in enumerate(ad_labels):
                combo_name = f"{pre_name} + {ad_name} + {post_name}"
                if combo_name in results:
                    final_price = results[combo_name]["prices"][-1]
                    heatmap[i, j] = final_price
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, cmap="viridis", aspect="auto")
        plt.colorbar(label="Final Token Price (USD)")
        plt.xticks(ticks=np.arange(len(ad_labels)), labels=ad_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(pre_labels)), labels=pre_labels)
        plt.title(f"Heatmap: Final Token Price (Month 60) - {post_name}", fontsize=12)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                plt.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="w")
        plt.tight_layout()
        plt.show()
    
    # ---------------------------
    # Plot Price Evolution Overlay Grid
    # ---------------------------
    # For each pre-TGE + airdrop combination, overlay the price evolution curves for all post-TGE configurations.
    # Define the lists of policy names.
    # pre_labels, ad_labels, and post_labels should be lists of names (strings).
    pre_labels = [p[0] for p in preTGE_policies]
    ad_labels = [a[0] for a in airdrop_policies]
    post_labels = [p[0] for p in postTGE_configs]

    plot_price_evolution_overlay(results, pre_labels, ad_labels, post_labels, max_rows_per_fig=3, max_cols_per_fig=4)