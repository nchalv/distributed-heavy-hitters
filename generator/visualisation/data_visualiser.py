
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
from pathlib import Path



def plot_frequency_distribution_with_hh(
    freq_dist,
    n,
    save_info,
    title="Frequency Distribution of Keys",
    max_keys_head=100,
    max_keys_show_all=100,
    percent_step=10
):


    # Set Roboto as the default font (if available)
    roboto_path = None

    # Try to find Roboto in system fonts
    if any('Roboto' in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto'
    else:
        # Optionally download Roboto (requires internet)
        try:
            roboto_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
            roboto_path = fm.urlretrieve(roboto_url, "Roboto-Regular.ttf")[0]
            roboto_prop = fm.FontProperties(fname=roboto_path)
            plt.rcParams['font.family'] = roboto_prop.get_name()
        except:
            print("Roboto not found - falling back to default font")

    # Always use the full distribution for totals and threshold, even when the
    # plot only renders the head of a large key set.
    global_total = sum(freq_dist.values())
    all_items = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
    total_keys = len(all_items)
    sorted_items = all_items
    pruned = False
    if len(sorted_items) > max_keys_show_all:
        sorted_items = sorted_items[:max_keys_head]
        pruned = True
    if not sorted_items:
        print(f"Skipping empty frequency plot: {title}")
        return
    keys, freqs = zip(*sorted_items)
    visible_total = sum(freqs)
    threshold = 1 / n
    threshold_value = threshold * global_total
    rel_freqs = [freq / global_total for freq in freqs]
    colors = ['#e74c3c' if rf > threshold else '#95a5a6' for rf in rel_freqs]

    plt.figure(figsize=(14, 7), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#f5f6f7')
    bars = ax.bar(range(len(freqs)), freqs, color=colors, width=0.6,
                  edgecolor=colors, linewidth=0.5, alpha=0.9)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Dynamic y-axis limit: 10% above max freq for visual focus
    y_max = max(freqs)
    y_lim_top = int((y_max * 1.15) // 10 + 1) * 10  # round up to nearest 10 for neatness
    ax.set_ylim(0, y_lim_top)

    ax.set_ylabel('Frequency (count)', fontsize=10, labelpad=10)
    if pruned:
        ax.set_xlabel(
            f'Keys sorted by frequency (top {len(keys)} of {total_keys})',
            fontsize=10,
            labelpad=10,
        )
    else:
        ax.set_xlabel('Keys sorted by frequency', fontsize=10, labelpad=10)
    title_extra = " (showing only top keys)" if pruned else ""
    ax.set_title(title + title_extra, fontsize=14, pad=20, fontweight='bold')
    plt.xticks(range(len(keys)), keys, rotation=90, fontsize=8, ha='center')

    # Format y-ticks as plain integers
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    hh_patch = mpatches.Patch(color='#e74c3c', label=f'Heavy Hitters (freq > 1/{n} ≈ {int(threshold_value):,})')
    other_patch = mpatches.Patch(color='#95a5a6', label='Other Keys')
    ax.legend(handles=[hh_patch, other_patch], loc="upper right", framealpha=1, facecolor='white')

    if pruned:
        ax.text(
            0.01,
            0.98,
            f"Rendered mass: {visible_total:,} / {global_total:,} "
            f"({visible_total / max(global_total, 1):.1%})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#bdc3c7", "alpha": 0.9},
        )

    # Draw threshold line if visible; otherwise, annotate below axes inside plot
    threshold_in_view = 0 < threshold_value <= y_lim_top
    if threshold_in_view:
        ax.axhline(y=threshold_value, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(
            len(keys) * 0.98, threshold_value * 1.01,
            f'1/{n} threshold = {int(threshold_value):,}',
            color='#e74c3c', ha='right', va='bottom'
        )
    else:
        # Place annotation below the lowest y-tick, stretching across width
        ax.text(
            0.5, -0.18, f'Note: HH threshold (1/{n} ≈ {int(threshold_value):,}) is not visible in the current plot window.',
            color='#e74c3c', ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.transAxes
        )

    plt.tight_layout()
    #plt.show()
    directory, file_format = save_info
    if directory is None:
        plt.show()
    else:
        # Validate file format
        if file_format.lower() not in ("pdf", "png"):
            raise ValueError("File format must be either 'pdf' or 'png'")

        # Create directory if it doesn't exist
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save the plot
        filename = path / f"{title.lower().replace('window', 'w').replace(': ', '_').replace(' ', '_')[:15]}.{file_format.lower()}"
        plt.savefig(filename, format=file_format.lower())
        plt.close()  # Close the figure to free memory

        print(f"Plot saved to: {filename}")
    # plt.savefig('./plots/'+title.lower().replace("window", "w").replace(": ", "_").replace(" ", "_")[:15])
    # plt.close()


def plot_key_partition_distribution(key_id, partitioned_window, num_partitions, window_num, save_info):
    # Set Roboto as the default font (if available)
    try:
        if any('Roboto' in f.name for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = 'Roboto'
        else:
            roboto_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
            roboto_path = fm.urlretrieve(roboto_url, "Roboto-Regular.ttf")[0]
            roboto_prop = fm.FontProperties(fname=roboto_path)
            plt.rcParams['font.family'] = roboto_prop.get_name()
    except:
        print("Roboto not found - falling back to default font")

    freqs = [partitioned_window[p].get(key_id, 0) for p in range(num_partitions)]

    plt.figure(figsize=(14, 7), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#f5f6f7')

    # Modern orange color with thinner bars (width=0.6) and edge
    bars = ax.bar(range(num_partitions), freqs, color='#e67e22', width=0.6,
                 edgecolor='#d35400', linewidth=0.5, alpha=0.9)

    # Grid and styling
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Dynamic y-axis limit
    y_max = max(freqs) if freqs else 1
    y_lim_top = int((y_max * 1.15) // 10 + 1) * 10
    ax.set_ylim(0, y_lim_top)

    # Labels and title with modern styling
    ax.set_xlabel("Partition ID", fontsize=10, labelpad=10)
    ax.set_ylabel(f"Frequency of {key_id}", fontsize=10, labelpad=10)
    ax.set_title(f"{window_num} : Partition-wise Frequency for {key_id}", fontsize=14, pad=20, fontweight='bold')

    # Format y-ticks with thousands separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))

    # X-ticks - rotate if many partitions
    if num_partitions > 20:
        plt.xticks(range(num_partitions), rotation=45, fontsize=8, ha='right')
    else:
        plt.xticks(range(num_partitions), fontsize=9)

    # Add data labels if not too many partitions
    if num_partitions <= 30:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # plt.show()
    directory, file_format = save_info
    if directory is None:
        plt.show()
    else:
        # Validate file format
        if file_format.lower() not in ("pdf", "png"):
            raise ValueError("File format must be either 'pdf' or 'png'")

        # Create directory if it doesn't exist
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save the plot
        filename = path / (f"{window_num} {key_id}".lower().replace("window", "w").replace(" ", "_")[:15]+f".{file_format.lower()}")
        plt.savefig(filename, format=file_format.lower())
        plt.close()  # Close the figure to free memory

        print(f"Plot saved to: {filename}")
    # plt.savefig('./plots/'+f"{window_num} {key_id}".lower().replace("window", "w").replace(" ", "_")[:15])
    # plt.close()

def plot_partition_skew(partitioned_window, window_num, save_info,):
    # Set Roboto as the default font (if available)
    try:
        if any('Roboto' in f.name for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = 'Roboto'
        else:
            roboto_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
            roboto_path = fm.urlretrieve(roboto_url, "Roboto-Regular.ttf")[0]
            roboto_prop = fm.FontProperties(fname=roboto_path)
            plt.rcParams['font.family'] = roboto_prop.get_name()
    except:
        print("Roboto not found - falling back to default font")

    totals = {pid: sum(freqs.values()) for pid, freqs in partitioned_window.items()}
    pids = sorted(totals)
    vals = [totals[p] for p in pids]

    mean_val = sum(vals)/len(vals) if vals else 0

    plt.figure(figsize=(14, 7), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#f5f6f7')

    # MODIFIED: Much thinner bars (width=0.3) and tighter spacing (align='edge')
    bars = ax.bar(pids, vals, color='#3498db', width=0.4, align='edge',
                 edgecolor='#2980b9', linewidth=0.5, alpha=0.9)

    # Mean reference line
    ax.axhline(y=mean_val, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(pids)*0.98, mean_val*1.05, f'Mean: {mean_val:,.1f}',
           color='#e74c3c', ha='right', va='bottom')

    # Grid and styling
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Dynamic y-axis limit
    y_max = max(vals) if vals else 1
    y_lim_top = int((y_max * 1.15) // 10 + 1) * 10
    ax.set_ylim(0, y_lim_top)

    # Labels and title
    title = f"{window_num} : Partition Skew"
    ax.set_xlabel("Partition ID", fontsize=10, labelpad=10)
    ax.set_ylabel("Total Assigned Frequency", fontsize=10, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')

    # Format y-ticks
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))

    # X-ticks - modified to account for thinner bars
    if len(pids) > 20:
        ax.set_xticks([x + 0.15 for x in pids])  # Center ticks with the bars
        ax.set_xticklabels(pids, rotation=45, fontsize=8, ha='right')
    else:
        ax.set_xticks([x + 0.15 for x in pids])  # Center ticks with the bars
        ax.set_xticklabels(pids, fontsize=9)

    # Add data labels if few enough partitions
    if len(pids) <= 30:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # plt.show()
    directory, file_format = save_info
    if directory is None:
        plt.show()
    else:
        # Validate file format
        if file_format.lower() not in ("pdf", "png"):
            raise ValueError("File format must be either 'pdf' or 'png'")

        # Create directory if it doesn't exist
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save the plot
        filename = path / (title.lower().replace("window", "w").replace("partition skew", "part_sk").replace(" : ", "_").replace(" ", "_")[:15]+f".{file_format.lower()}")
        plt.savefig(filename, format=file_format.lower())
        plt.close()  # Close the figure to free memory

        print(f"Plot saved to: {filename}")
    # plt.savefig('./plots/'+title.lower().replace("window", "w").replace("partition skew", "part_sk").replace(" : ", "_").replace(" ", "_")[:15])
    # plt.close()
