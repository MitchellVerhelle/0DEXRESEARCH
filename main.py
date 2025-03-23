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
    AevoBoostedVolumeRewardPolicy,
    HelixLoyaltyPointsRewardPolicy,
    GameLikeMMRRewardPolicy
)
from plot_helper import (plot_price_evolution_grid, plot_airdrop_distribution_grid)
from simulation import MonteCarloSimulation, simulate_price_evolution_dynamic

# Helper function to run one simulation combination.
def run_simulation_for_combo(combo_name, num_users, total_supply, preTGE_steps, simulation_horizon,
                             ad_policy, pre_policy, base_price, elasticity, buyback_rate):
    sim = MonteCarloSimulation(
        num_users=num_users,
        total_supply=total_supply,
        preTGE_steps=preTGE_steps,
        simulation_horizon=simulation_horizon,
        airdrop_policy=ad_policy,
        preTGE_rewards_policy=pre_policy
    )
    TGE_total, months, total_unlocked_history, unlocked_history, dist = sim.run()
    # Call dynamic price evolution; note: distribution parameter is passed last.
    prices = simulate_price_evolution_dynamic(
        TGE_total, total_unlocked_history, sim.user_pool.users,
        base_price=base_price, elasticity=elasticity, buyback_rate=buyback_rate,
        alpha=0.5, mu=0.0, sigma=0.05, jump_intensity=0.1, jump_mean=-0.05, jump_std=0.1,
        distribution=dist
    )
    return combo_name, {
        "TGE_total": TGE_total,
        "months": months,
        "total_unlocked_history": total_unlocked_history,
        "unlocked_history": unlocked_history,
        "prices": prices,
        "TGE_tokens": [user.tokens for user in sim.user_pool.users],
        "distribution": dist
    }

if __name__ == '__main__':
    # Define lists of airdrop conversion policies and pre-TGE rewards policies.
    airdrop_policies = [
        ("Linear", LinearAirdropPolicy()),
        ("Exponential", ExponentialAirdropPolicy()),
        ("Tiered Linear", TieredLinearAirdropPolicy()),
        ("Tiered Constant", TieredConstantAirdropPolicy()),
        ("Tiered Exponential", TieredExponentialAirdropPolicy())
    ]
    
    preTGE_policies = [
        ("dYdX Retro", DydxRetroTieredRewardPolicy()),
        ("Vertex Maker/Taker", VertexMakerTakerRewardPolicy()),
        ("Jupiter Volume Tier", JupiterVolumeTierRewardPolicy()),
        ("Aevo Boosted Volume", AevoBoostedVolumeRewardPolicy()),
        ("Helix Loyalty", HelixLoyaltyPointsRewardPolicy()),
        ("Game-like MMR", GameLikeMMRRewardPolicy())
    ]
    
    # Simulation parameters.
    num_users = 100000
    total_supply = 1_000_000_000
    preTGE_steps = 50
    simulation_horizon = 60  # months
    base_price = 10.0
    elasticity = 1.0
    buyback_rate = 0.2  # 20% of additional unlocked tokens are removed.
    
    results = {}
    tasks = []
    
    # Run simulations concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for pre_name, pre_policy in preTGE_policies:
            for ad_name, ad_policy in airdrop_policies:
                combo_name = f"{pre_name} + {ad_name}"
                print(f"Submitting simulation for: {combo_name}")
                task = executor.submit(
                    run_simulation_for_combo,
                    combo_name,
                    num_users, total_supply, preTGE_steps, simulation_horizon,
                    ad_policy, pre_policy, base_price, elasticity, buyback_rate
                )
                tasks.append(task)
        
        # Collect results.
        for future in concurrent.futures.as_completed(tasks):
            combo_name, res = future.result()
            results[combo_name] = res
            print(f"Completed simulation for: {combo_name}")
    
    # Plot histograms of TGE tokens distribution.
    plt.figure(figsize=(12, 8))
    for combo_name, data in results.items():
        tokens = np.array(data["TGE_tokens"])
        finite_tokens = tokens[np.isfinite(tokens)]
        if finite_tokens.size > 0:
            plt.hist(finite_tokens, bins=30, alpha=0.5, label=combo_name,
                     histtype='step', linewidth=1.5)
    plt.xlabel("Tokens Assigned at TGE")
    plt.ylabel("Number of Users")
    plt.title("Histogram of TGE Tokens Distribution for All Combinations")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True)
    plt.show(block=True)
    
    # For one chosen combination, plot vesting graphs.
    chosen = "dYdX Retro + Linear"
    chosen_result = results[chosen]
    
    plt.figure(figsize=(10, 6))
    plt.plot(chosen_result["months"], chosen_result["total_unlocked_history"],
             label=f"Total Unlocked Tokens ({chosen})", color='blue')
    plt.xlabel("Months since TGE")
    plt.ylabel("Total Unlocked Tokens")
    plt.title("Post-TGE Vesting: Total Unlocked Tokens Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    groups = list(chosen_result["unlocked_history"].keys())
    data_stack = np.vstack([chosen_result["unlocked_history"][group] for group in groups])
    plt.stackplot(chosen_result["months"], data_stack, labels=groups)
    plt.xlabel("Months since TGE")
    plt.ylabel("Tokens Unlocked")
    plt.title("Post-TGE Vesting: Unlocked Tokens per Group Over Time")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    # Build a heatmap of final token prices (month 60) for each combination.
    pre_labels = [p[0] for p in preTGE_policies]
    ad_labels = [a[0] for a in airdrop_policies]
    heatmap = np.zeros((len(pre_labels), len(ad_labels)))
    for i, pre_name in enumerate(pre_labels):
        for j, ad_name in enumerate(ad_labels):
            combo_name = f"{pre_name} + {ad_name}"
            final_price = results[combo_name]["prices"][-1]
            heatmap[i, j] = final_price
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="viridis", aspect="auto")
    plt.colorbar(label="Final Token Price (USD)")
    plt.xticks(ticks=np.arange(len(ad_labels)), labels=ad_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(pre_labels)), labels=pre_labels)
    plt.title("Heatmap: Final Token Price (Month 60) for Each Combination")
    for i in range(len(pre_labels)):
        for j in range(len(ad_labels)):
            plt.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="w")
    plt.tight_layout()
    plt.show()
    
    # Plot a grid of token price evolution curves (excluding t=0) for each combination.
    plot_price_evolution_grid(results, pre_labels, ad_labels, max_rows_per_fig=6)

    # Plot the grid of TGE token distribution.
    plot_airdrop_distribution_grid(results, pre_labels, ad_labels)