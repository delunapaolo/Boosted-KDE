import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


def adjust_spines(ax, spines, offset=3, smart_bounds=False):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', offset))
            spine.set_smart_bounds(smart_bounds)
        else:
            spine.set_color('None')  # don't draw spine

    # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # No y-axis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # No x-axis ticks
        ax.xaxis.set_ticks([])


def compare_results(data, x, y, hue, col, group_labels, dodge=True,
                    n_rows=None, n_cols=None):
    # Compute offsets
    width = 0.8
    n_levels = np.unique(data[hue]).shape[0]
    if dodge:
        each_width = width / n_levels
        offsets = np.linspace(0, width - each_width, n_levels)
        offsets -= offsets.mean()
    else:
        offsets = np.zeros(n_levels)

    # Set order of x groups
    if x == 'classifier type':
        order = ['SVM', 'ocSVM']
    else:
        order = None

    # Open figure and plot groups
    n_groups = np.unique(data[col]).shape[0]
    if n_rows is None and n_cols is None:
        n_rows = 1
        n_cols = n_groups
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=True, figsize=(18, 5))
    for i_group in range(n_groups):
        # Select data
        data_this_group = data[data[col] == i_group]

        # Make strip plot
        p = sns.categorical._StripPlotter(x=x, y=y, hue=hue,
                                          data=data_this_group,
                                          order=order, jitter=True,
                                          dodge=dodge, palette='hls',
                                          hue_order=None, orient='v',
                                          color=None)
        p.draw_stripplot(ax[i_group], dict(s=10 ** 2, edgecolor='w',
                                           linewidth=1, zorder=1))
        # Add legend and keep also left axes for first group only
        if i_group == 0:
            p.add_legend_data(ax[i_group])
            keep_axes = ['bottom', 'left']
        else:
            p.hue_names = None
            keep_axes = ['bottom']
        # Add labels
        p.annotate_axes(ax[i_group])

        # Add means of each group
        sns.categorical._PointPlotter(x=x, y=y, hue=hue, data=data_this_group,
                                      order=order, estimator=np.mean,
                                      ci=95, n_boot=1000, markers='o',
                                      linestyles='-', dodge=offsets[-1],
                                      join=False, scale=1, color='k',
                                      errwidth=None, capsize=None,
                                      hue_order=None, units=None, orient='v',
                                      palette=None).draw_points(ax[i_group])

        # Adjust axes appearance
        adjust_spines(ax[i_group], spines=keep_axes, offset=0,
                      smart_bounds=False)
        ax[i_group].set_title(group_labels[i_group])
        if i_group > 0:
            ax[i_group].set_ylabel('')
        ax[i_group].set_xlabel('')
    # Set same y-limits
    y_limits = np.vstack([i.get_ylim() for i in ax])
    y_limits = [y_limits.min(), y_limits.max()]
    [i.set_ylim(*y_limits) for i in ax]
    # Adjust subplot layout
    fig.tight_layout()
    fig.subplots_adjust(wspace=.2)
