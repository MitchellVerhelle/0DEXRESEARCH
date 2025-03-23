import matplotlib.pyplot as plt

def plot_price_evolution_grid(results, pre_labels, ad_labels, max_rows_per_fig=6):
    import numpy as np
    import matplotlib.pyplot as plt
    
    nrows = len(pre_labels)
    ncols = len(ad_labels)
    
    pages = [pre_labels[i:i+max_rows_per_fig] for i in range(0, nrows, max_rows_per_fig)]
    
    page_index = 1
    for page in pages:
        current_nrows = len(page)
        fig, axs = plt.subplots(current_nrows, ncols, 
                                figsize=(3 * ncols, 2.5 * current_nrows), 
                                sharex=True, sharey=True)
        if current_nrows == 1:
            axs = np.array([axs])
        for i, pre_name in enumerate(page):
            orig_i = pre_labels.index(pre_name)
            for j, ad_name in enumerate(ad_labels):
                combo_name = f"{pre_name} + {ad_name}"
                if combo_name in results:
                    months = results[combo_name]["months"]
                    prices = results[combo_name]["prices"]
                    axs[i, j].plot(months[1:], prices[1:], marker='o')
                    axs[i, j].set_title(combo_name, fontsize=8)
                else:
                    axs[i, j].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
                if i == current_nrows - 1:
                    axs[i, j].set_xlabel("Months")
                if j == 0:
                    axs[i, j].set_ylabel("Price (USD)")
                axs[i, j].grid(True)
        fig.suptitle(f"Token Price Evolution (t > 0) - Page {page_index}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        page_index += 1

def plot_airdrop_distribution_grid(results, pre_labels, ad_labels):
    """
    Plots a grid of bar charts showing the TGE token distribution by user type
    for each combination of pre-TGE rewards policy (rows) and airdrop conversion policy (columns).

    Parameters:
      - results: A dictionary containing simulation results for each combination,
                 keyed as "<preTGE policy> + <airdrop policy>".
                 Each result should have a "distribution" key, e.g.:
                 {"small": 40.0, "medium": 30.0, "large": 20.0, "sybil": 10.0}.
      - pre_labels: List of pre-TGE policy names.
      - ad_labels: List of airdrop policy names.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    nrows = len(pre_labels)
    ncols = len(ad_labels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows),
                            sharex=True, sharey=True)
    
    # Ensure axs is a 2D array.
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
                    # Normalize so the values sum to 100.
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