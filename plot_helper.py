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

def plot_price_evolution_overlay(results, pre_labels, ad_labels, post_labels, 
                                 max_rows_per_fig=4, max_cols_per_fig=4):
    """
    Plots a grid of subplots. Each subplot corresponds to a unique pre-TGE + airdrop combination.
    On each subplot, the price evolution curves for all post-TGE configurations (e.g. Baseline,
    High Volatility, Low Volatility, Aggressive Buyback) are overlaid with a legend.
    
    Additionally, if the Baseline active fraction data is available for that combination, it is
    overlaid as grey bars on a twin y-axis.
    
    Parameters:
      - results: Dictionary containing simulation results keyed as "pre_policy + ad_policy + post_config".
      - pre_labels: List of pre-TGE rewards policy names.
      - ad_labels: List of airdrop conversion policy names.
      - post_labels: List of post-TGE rewards configuration names.
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
        # Ensure axs is 2D.
        if nrows == 1:
            axs = np.array([axs])
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
        
        for idx, key in enumerate(current_keys):
            i = idx // ncols
            j = idx % ncols
            ax = axs[i, j]
            # For each post configuration, plot the corresponding curve.
            for post_config in post_labels:
                combo_name = f"{key} + {post_config}"
                if combo_name in results:
                    months = results[combo_name]["months"]
                    prices = results[combo_name]["prices"]
                    ax.plot(months, prices, marker='o', label=post_config, zorder=2)
            # Overlay grey bars from the Baseline active fraction if available.
            baseline_key = f"{key} + Baseline"
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
        
        # For any empty subplot slots, hide the axes.
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
