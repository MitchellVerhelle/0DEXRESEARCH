import numpy as np
import matplotlib.pyplot as plt
import math

def plot_airdrop_distribution_grid(results, pre_labels, ad_labels):
    """
    Plots a grid of bar charts showing the TGE token distribution by user type
    for each combination of pre-TGE rewards policy (rows) and airdrop conversion policy (columns).
    """
    nrows = len(pre_labels)
    ncols = len(ad_labels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows),
                            sharex=True, sharey=True)
    if nrows == 1:
        axs = np.array([axs])
    if ncols == 1:
        axs = np.expand_dims(axs, axis=1)
    
    for i, pre_name in enumerate(pre_labels):
        for j, ad_name in enumerate(ad_labels):
            combo_name = f"{pre_name} + {ad_name}"
            ax = axs[i, j]
            if combo_name in results and "distribution" in results[combo_name]:
                dist_data = results[combo_name]["distribution"]
                user_types = list(dist_data.keys())
                percentages = list(dist_data.values())
                total = sum(percentages)
                if total > 0 and total != 100:
                    percentages = [100 * p / total for p in percentages]
                ax.bar(user_types, percentages, color=['blue', 'orange', 'green', 'red'])
                ax.set_title(combo_name, fontsize=8)
                ax.set_ylim([0, 100])
            else:
                ax.text(0.5, 0.5, "No data",
                        horizontalalignment='center',
                        verticalalignment='center')
            if i == nrows - 1:
                ax.set_xlabel("User Type")
            if j == 0:
                ax.set_ylabel("Percent of TGE Tokens")
            ax.grid(True, axis='y')
    fig.suptitle("TGE Distribution by User Type for All Combinations", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_price_evolution_overlay(results, pre_labels, ad_labels, post_reward_labels, scenario_labels, 
                                 max_rows_per_fig=4, max_cols_per_fig=4):
    """
    Plots a grid of subplots. Each subplot corresponds to a unique combination
    of pre-TGE rewards policy and airdrop conversion policy.
    
    On each subplot, the price evolution curves for all combinations of 
    post-TGE reward policies and scenario configurations are overlaid with a legend.
    The legend label for each curve is formatted as "RewardPolicy | ScenarioConfig".
    
    Additionally, if baseline active fraction data is available for that
    pre+ad combination under the "Generic" reward policy and "Baseline" scenario,
    it is overlaid as grey bars on a twin y-axis.
    
    Parameters:
      - results: Dictionary containing simulation results keyed as 
          "pre_policy + ad_policy + post_reward_policy + scenario_config".
      - pre_labels: List of pre-TGE rewards policy names.
      - ad_labels: List of airdrop conversion policy names.
      - post_reward_labels: List of post-TGE reward policy names (e.g. ["Generic", "Engagement"]).
      - scenario_labels: List of post-TGE scenario configuration names (e.g. ["Baseline", "HighVol", ...]).
      - max_rows_per_fig: Maximum number of rows per page.
      - max_cols_per_fig: Maximum number of columns per page.
    """
    # Create list of keys for each pre + ad combination.
    combined_keys = [f"{pre} + {ad}" for pre in pre_labels for ad in ad_labels]
    total_plots = len(combined_keys)
    max_plots_per_page = max_rows_per_fig * max_cols_per_fig
    n_pages = math.ceil(total_plots / max_plots_per_page)
    
    page_index = 1
    for page in range(n_pages):
        start = page * max_plots_per_page
        end = start + max_plots_per_page
        current_keys = combined_keys[start:end]
        n_plots = len(current_keys)
        # Determine grid dimensions for this page.
        ncols = min(max_cols_per_fig, n_plots)
        nrows = math.ceil(n_plots / ncols)
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), sharex=True)
        # Ensure axs is a 2D array.
        if nrows == 1:
            axs = np.array([axs])
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
        
        for idx, key in enumerate(current_keys):
            i = idx // ncols
            j = idx % ncols
            ax = axs[i, j]
            # For each combination of post reward policy and scenario configuration, plot the curve.
            for reward_label in post_reward_labels:
                for scenario_label in scenario_labels:
                    combo_name = f"{key} + {reward_label} + {scenario_label}"
                    if combo_name in results:
                        months = results[combo_name]["months"]
                        prices = results[combo_name]["prices"]
                        label = f"{reward_label} | {scenario_label}"
                        ax.plot(months, prices, marker='o', label=label, zorder=2)
            # Overlay grey bars from the baseline active fraction if available.
            # We assume baseline is under reward policy "Generic" and scenario "Baseline".
            baseline_key = f"{key} + Generic + Baseline"
            if baseline_key in results and "active_fraction_history" in results[baseline_key]:
                active_frac = results[baseline_key]["active_fraction_history"]
                ax2 = ax.twinx()
                ax2.bar(months, active_frac, color='grey', alpha=0.3, width=0.8, zorder=0)
                ax2.set_ylim(0, 1)
                ax2.tick_params(axis='y', labelsize=8, colors='grey')
                ax2.set_ylabel("Active Fraction", fontsize=8, color='grey')
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("Months", fontsize=8)
            ax.set_ylabel("Token Price (USD)", fontsize=8)
            ax.grid(True)
            ax.legend(fontsize=8)
        
        # Hide any unused subplot axes.
        total_slots = nrows * ncols
        for idx in range(n_plots, total_slots):
            i = idx // ncols
            j = idx % ncols
            axs[i, j].axis('off')
        
        fig.suptitle(f"Token Price Evolution (Overlay) - Page {page_index}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        page_index += 1

def plot_vesting_schedule(months, unlocked_history, total_unlocked_history):
    """
    Plots the vesting schedule allocations per group (stackplot) along with the total unlocked tokens (line plot)
    over the simulation horizon.
    """
    plt.figure(figsize=(10,6))
    groups = list(unlocked_history.keys())
    data_stack = np.vstack([unlocked_history[group] for group in groups])
    plt.stackplot(months, data_stack, labels=groups, alpha=0.7)
    plt.plot(months, total_unlocked_history, 'k--', linewidth=2, label="Total Unlocked")
    plt.xlabel("Months since TGE")
    plt.ylabel("Tokens Unlocked")
    plt.title("Post-TGE Vesting: Unlocked Tokens by Group & Total Supply Over Time")
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True)
    plt.show()

def plot_avg_price_heatmap(results, pre_labels, ad_labels, post_reward_labels, scenario_labels):
    """
    For each post-TGE reward policy, average the final token prices over all post-TGE scenarios,
    then plot a heatmap (rows: pre-TGE policies, columns: airdrop policies) of these averaged final prices.
    
    Parameters:
      - results: Dictionary with keys "pre_policy + ad_policy + post_reward_policy + scenario".
      - pre_labels: List of pre-TGE rewards policy names.
      - ad_labels: List of airdrop conversion policy names.
      - post_reward_labels: List of post-TGE reward policy names.
      - scenario_labels: List of scenario configuration names.
    """
    for reward in post_reward_labels:
        heatmap = np.zeros((len(pre_labels), len(ad_labels)))
        for i, pre in enumerate(pre_labels):
            for j, ad in enumerate(ad_labels):
                prices = []
                for scenario in scenario_labels:
                    combo = f"{pre} + {ad} + {reward} + {scenario}"
                    if combo in results:
                        prices.append(results[combo]["prices"][-1])
                if prices:
                    avg_price = np.mean(prices)
                else:
                    avg_price = 0
                heatmap[i, j] = avg_price
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, cmap="viridis", aspect="auto")
        plt.colorbar(label="Avg Final Token Price (USD)")
        plt.xticks(ticks=np.arange(len(ad_labels)), labels=ad_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(pre_labels)), labels=pre_labels)
        plt.title(f"Avg Final Price (Over Scenarios) - Reward Policy: {reward}", fontsize=12)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                plt.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", color="w")
        plt.tight_layout()
        plt.show()


def plot_avg_price_evolution_overlay(results, pre_labels, ad_labels, post_reward_labels, scenario_labels, 
                                     max_rows_per_fig=4, max_cols_per_fig=4):
    """
    For each pre-TGE + airdrop combination, overlay averaged price evolution curves for each post-TGE reward policy.
    For each (pre, ad, reward), the price evolution time series is averaged element-wise over all scenarios.
    
    The result is a grid of subplots with one averaged curve per reward policy.
    
    Parameters:
      - results: Dictionary with keys "pre_policy + ad_policy + post_reward_policy + scenario".
      - pre_labels: List of pre-TGE rewards policy names.
      - ad_labels: List of airdrop conversion policy names.
      - post_reward_labels: List of post-TGE reward policy names.
      - scenario_labels: List of scenario configuration names.
      - max_rows_per_fig, max_cols_per_fig: Layout parameters.
    """
    combined_keys = [f"{pre} + {ad}" for pre in pre_labels for ad in ad_labels]
    total_plots = len(combined_keys)
    max_plots_per_page = max_rows_per_fig * max_cols_per_fig
    n_pages = math.ceil(total_plots / max_plots_per_page)
    
    page_index = 1
    for page in range(n_pages):
        start = page * max_plots_per_page
        end = start + max_plots_per_page
        current_keys = combined_keys[start:end]
        n_plots = len(current_keys)
        ncols = min(max_cols_per_fig, n_plots)
        nrows = math.ceil(n_plots / ncols)
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), sharex=True)
        if nrows == 1:
            axs = np.array([axs])
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
        
        for idx, key in enumerate(current_keys):
            i = idx // ncols
            j = idx % ncols
            ax = axs[i, j]
            for reward in post_reward_labels:
                price_series_list = []
                for scenario in scenario_labels:
                    combo = f"{key} + {reward} + {scenario}"
                    if combo in results:
                        price_series_list.append(np.array(results[combo]["prices"]))
                if price_series_list:
                    avg_series = np.mean(np.vstack(price_series_list), axis=0)
                    # Use months from one of the combos (assumed identical across scenarios)
                    months = results[f"{key} + {reward} + {scenario_labels[0]}"]["months"]
                    ax.plot(months, avg_series, marker='o', label=reward, zorder=2)
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("Months", fontsize=8)
            ax.set_ylabel("Avg Token Price (USD)", fontsize=8)
            ax.grid(True)
            ax.legend(fontsize=8)
        total_slots = nrows * ncols
        for idx in range(n_plots, total_slots):
            i = idx // ncols
            j = idx % ncols
            axs[i, j].axis('off')
        fig.suptitle(f"Averaged Token Price Evolution (Over Scenarios) - Page {page_index}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        page_index += 1
