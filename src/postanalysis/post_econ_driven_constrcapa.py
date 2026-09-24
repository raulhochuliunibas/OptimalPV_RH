import os
import glob
import json
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch


class EconDrivenConstrcapa:
    pass
        # def NPVhist(self, 
        #             scen = 'pvalloc_29nbfs_LRG2_max',
        #                                 # export_name = None,
        #                                 # npv_hist_width = 3.85,
        #                                 # npv_hist_height= 3.4,
        #                                 # # npv_hist_xrange = (-1e5, 4.75e5),
        #                                 # title = 'NPV Distribution',
        #                                 # x_label = 'NPV (CHF)',
        #                                 # y_label = 'Count',
        #                                 # negative_rgb = (214, 39, 40),
        #                                 # positive_rgb = (31, 119, 180),
        #                                 # plot_width_func = None,
        #                                 # plot_height_func = None,
        #                                 ):
        #     npv_df
        #     # npv_df = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'npv_df.parquet'))
        #     npv_df = pd.read_parquet(os.path.join(self.dir_path, f'npv_df_1_{scen}.parquet'))

        #     if 'NPV_uid_before_subsidy' not in npv_df.columns:
        #         print("Column 'NPV_uid_before_subsidy' not found in npv_df.")
        #         return

        #     df_plot = npv_df.loc[npv_df['NPV_uid_before_subsidy'].notna(), ['NPV_uid_before_subsidy']].copy()
        #     if df_plot.empty:
        #         print("No values found in 'NPV_uid_before_subsidy' for histogram.")
        #         return

        #     mean_val = df_plot['NPV_uid_before_subsidy'].mean()
        #     median_val = df_plot['NPV_uid_before_subsidy'].median()

        #     def rgb_to_mpl_color(rgb_value):
        #         if not isinstance(rgb_value, (list, tuple, np.ndarray)) or len(rgb_value) != 3:
        #             raise ValueError('Expected an RGB tuple/list with exactly 3 values.')
        #         rgb_array = np.asarray(rgb_value, dtype=float)
        #         if np.nanmax(rgb_array) > 1.0:
        #             rgb_array = rgb_array / 255.0
        #         return tuple(rgb_array.tolist())

        #     negative_color = rgb_to_mpl_color(negative_rgb)
        #     positive_color = rgb_to_mpl_color(positive_rgb)

        #     plot_width = self.plot_width if plot_width_func is None else plot_width_func
        #     plot_height = self.plot_height if plot_height_func is None else plot_height_func


        #     values = df_plot['NPV_uid_before_subsidy'].to_numpy()
        #     data_min = float(np.nanmin(values))
        #     data_max = float(np.nanmax(values))
        #     hist_range = (data_min, data_max)
        #     if npv_hist_xrange is not None and len(npv_hist_xrange) == 2:
        #         input_min = float(npv_hist_xrange[0])
        #         input_max = float(npv_hist_xrange[1])
        #         hist_min = max(input_min, data_min)
        #         hist_max = min(input_max, data_max)
        #         if hist_min < hist_max:
        #             hist_range = (hist_min, hist_max)

        #     bins = np.linspace(hist_range[0], hist_range[1], 41)
        #     counts, bin_edges = np.histogram(values, bins=bins)
        #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        #     bar_colors = [negative_color if center < 0 else positive_color for center in bin_centers]

        #     plt.figure(figsize=(plot_width, plot_height))
        #     ax = plt.gca()
        #     ax.bar(
        #         bin_edges[:-1],
        #         counts,
        #         width=np.diff(bin_edges),
        #         align='edge',
        #         color=bar_colors,
        #         edgecolor='white',
        #         linewidth=0.2,
        #         alpha=0.75,
        #     )
        #     ax.axvline(0, color='black', linestyle=':', linewidth=1.0, label='Zero')
        #     summary_line_color = 'black'
        #     ax.axvline(mean_val, color=summary_line_color, linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:,.0f}')
        #     ax.axvline(median_val, color=summary_line_color, linestyle='-', linewidth=1.5, label=f'Median: {median_val:,.0f}')
        #     ax.set_xlim(hist_range[0], hist_range[1])
        #     plt.xlabel(x_label)
        #     plt.ylabel(y_label)
        #     plt.title(title)
        #     plt.legend(title=None)
        #     plt.tight_layout()
        #     # plt.show()
        #     if export_name is None:
        #         export_file = f'{scen}_npv_df_hist.png'
        #     else:
        #         export_file = export_name if os.path.splitext(export_name)[1] else f'{export_name}.png'
        #     self._save_figure(os.path.join(self.dir_path_export, export_file), plot_width, plot_height)
