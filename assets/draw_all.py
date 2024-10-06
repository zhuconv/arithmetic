import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def check_and_merge_2d_lists(json_files):
    merged_list = None
    for file in json_files:
        data = load_json(file)[0]
        data_np = np.array(data)

        if merged_list is None:
            merged_list = np.zeros_like(data_np)

        conflict = (merged_list > 0) & (data_np > 0)
        if np.any(conflict):
            raise ValueError(f"Conflict detected between files at {file}")

        merged_list = np.where(data_np > 0, data_np, merged_list)

    return merged_list.tolist()

def grid_plotter(ax, data, title, legend=True):
    """Plot a 2D accuracy grid on the given axes."""
    data = np.array(data) * 100
    df = pd.DataFrame(data)
    df = df.iloc[:40, :40]

    sns.heatmap(df, ax=ax, cmap="YlGnBu", fmt=".1f", annot_kws={'size': 8, 'rotation': 0}, vmin=0, vmax=100, cbar=False, rasterized=True)
    if title is not None:
        # title += f' (avg. {round(df.mean(axis=None), 2)})'
        ax.set_title(title, pad=10, fontsize=20)
    print(round(df.mean(axis=None), 2))
    size = data.shape[0]
    list_1 = [10, 20, 30, 40]
    ax.set_xticks(list_1)
    ax.set_xticklabels(list_1, fontsize=20)
    if legend:
        ax.set_yticks(list_1)
        ax.set_yticklabels(list_1, fontsize=20)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # if not legend:
        # ax.legend([])

def plot_results(names):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration
    # plt.subplots_adjust(top=0.82)

    for i, name in enumerate(names):
        path = f'/scratch/gpfs/pw4811/arithmetic/cramming-data/{name}/downstream'
        json_files = [f'{path}/accs_{j}.json' for j in range(1, 9)]
        legend = True if i == 0 else False
        title = f"{name_mapping.get(name, None)}"

        try:
            merged_result = check_and_merge_2d_lists(json_files)
            grid_plotter(axs[i], merged_result, title=title, legend=legend)
        except ValueError as e:
            print(str(e))
            exit(-1)
            # axs[i].text(0.5, 0.5, str(e), ha='center', va='center')
            # axs[i].set_title(name)
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1), bbox_transform=fig.transFigure)
    norm = plt.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])
    # 在全局的图上添加颜色条 (可以通过 shrink 调整颜色条大小)
    # 创建分隔器
    # divider = make_axes_locatable(fig)

    # 在右侧添加颜色条的 Axes，减少 pad 以使颜色条更靠近主图
    cax_position = [0.92, 0.12, 0.016, 0.76]  # 根据需要调整这些值
    cax = fig.add_axes(cax_position)

# 添加颜色条
    # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar = fig.colorbar(sm, ax=axs, cax=cax, orientation='vertical', shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    # plt.tight_layout()
    # import rand
    # plt.pcolormesh(rand(10,10); rasterized=true)
    plt.savefig(f"arithmetic.pdf", dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":
    names = ["add_ronope", "add_randomized", "add_fire", "add_adayarn"] #, "add_ropecus", "add_fire_abacus", "add_adarope_new", "add_adayarn_cus"]  # Replace with actual names
    name_mapping = {"add_ronope": "RoPE", "add_fire": "FIRE", "add_adarope_ppp": "AdeRoPE", 'add_adayarn': "TAPE", "add_randomized": "RandPE"}
    plot_results(names)