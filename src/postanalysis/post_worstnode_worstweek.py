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


class _PlotExportMixin:
    """
    Shared save/export helpers used by every postanalysis topic class in this file.
    """
    def _write_latex_from_template(self,
                                template_file,
                                export_file,
                                replacements):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = template_file if os.path.isabs(template_file) else os.path.join(script_dir, template_file)
        export_path = export_file if os.path.isabs(export_file) else os.path.join(self.dir_path_export, export_file)

        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.read()

        filled_text = template_text.format(**replacements)

        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(filled_text)

        print(f'LaTeX file written: {export_path}')

    def _save_figure(self, export_path, plot_width=None, plot_height=None):
        fig = plt.gcf()
        if plot_width is not None and plot_height is not None:
            fig.set_size_inches(plot_width, plot_height, forward=True)
        plt.savefig(export_path, dpi=self.plot_dpi)


class WorstNodes(_PlotExportMixin):
    """
    Class to analyze and visualize the worst nodes in terms of excess feed-in over time and specific nodes.
    """
    def _rgb_to_mpl(self, rgb):
        return tuple(c / 255.0 if max(rgb) > 1.0 else c for c in rgb)

    def _resolve_iter_list(self, scen, iter_list):
        if iter_list is not None:
            return list(iter_list)

        iter_dir = os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'pred_gridprem_node_by_M')
        iter_nums = []
        for f in glob.glob(os.path.join(iter_dir, 'gridnode_df_*.parquet')):
            suffix = os.path.splitext(os.path.basename(f))[0].rsplit('_', 1)[-1]
            if suffix.isdigit():
                iter_nums.append(int(suffix))

        return [max(iter_nums)] if iter_nums else [None]

    def _load_gridnode_iter_df(self, scen, iter_val):
        if iter_val is not None:
            iter_path = os.path.join(
                self.data_path, 'pvalloc', scen, 'zMC1', 'pred_gridprem_node_by_M', f'gridnode_df_{iter_val}.parquet'
            )
            if os.path.exists(iter_path):
                return pl.read_parquet(iter_path), str(iter_val)

        final_path = os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet')
        return pl.read_parquet(final_path), ('final' if iter_val is None else str(iter_val))

    def _topo_egid_counts(self, scen):
        topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
        topo_rows = [{'EGID': k, 'grid_node': v['grid_node']} for k, v in topo.items()]
        return pl.DataFrame(topo_rows).group_by('grid_node').agg(pl.len().alias('n_egid'))

    def _build_threshold_segments(self, x_vals, y_vals, threshold):
        below_segments = []
        above_segments = []

        for idx in range(len(x_vals) - 1):
            x0, x1 = float(x_vals[idx]), float(x_vals[idx + 1])
            y0, y1 = float(y_vals[idx]), float(y_vals[idx + 1])

            if y0 <= threshold and y1 <= threshold:
                below_segments.append([(x0, y0), (x1, y1)])
                continue
            if y0 >= threshold and y1 >= threshold:
                above_segments.append([(x0, y0), (x1, y1)])
                continue

            x_cross = x0 if y1 == y0 else x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            crossing_point = (x_cross, threshold)
            if y0 < threshold < y1:
                below_segments.append([(x0, y0), crossing_point])
                above_segments.append([crossing_point, (x1, y1)])
            else:
                above_segments.append([(x0, y0), crossing_point])
                below_segments.append([crossing_point, (x1, y1)])

        return below_segments, above_segments



    def excessfeedin_annualTS(self,
                            scen = 'pvalloc_LRG3_max',
                            title = 'Excess Feed-in - Annual Time Series',
                            export_name = 'excessfeedin_annualTS',
                            gridnode_list: list = None,
                            n_worstnodes: int = None,
                            iter_list: list = None,
                            x_label = 'Hour of year',
                            y_label = 'Feed-in at node (kW)',
                            y_scaling = 1.0,
                            linewidth = 0.6,
                            excess_rgb_line = (214, 39, 40),
                            production_rgb_line = (31, 119, 180),
                            legend_loc = 'upper left',
                            plot_width_func = None,
                            plot_height_func = None,):

        iter_vals = self._resolve_iter_list(scen, iter_list)

        for iter_val in iter_vals:
            gridnode_df, iter_label = self._load_gridnode_iter_df(scen, iter_val)

            if gridnode_list is not None:
                nodes_to_plot = list(gridnode_list)
            elif n_worstnodes is not None:
                nodes_to_plot = (
                    gridnode_df
                    .group_by('grid_node')
                    .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
                    .sort('total_loss_kW', descending=True)
                    .head(n_worstnodes)
                    .get_column('grid_node')
                    .to_list()
                )
            else:
                nodes_to_plot = gridnode_df.get_column('grid_node').unique().to_list()

            below_segments_all = []
            above_segments_all = []
            y_min_global, y_max_global = 0.0, 0.0

            for node in nodes_to_plot:
                node_df = gridnode_df.filter(pl.col('grid_node') == node).sort('t_int')
                if node_df.is_empty():
                    continue

                threshold_value = float(node_df.get_column('kW_threshold')[0]) * y_scaling
                x_values = node_df.get_column('t_int').to_numpy()
                y_values = (node_df.get_column('feedin_atnode_kW') * y_scaling).to_numpy()

                below_segments, above_segments = self._build_threshold_segments(x_values, y_values, threshold_value)
                below_segments_all.extend(below_segments)
                above_segments_all.extend(above_segments)

                y_min_global = min(y_min_global, float(y_values.min()))
                y_max_global = max(y_max_global, float(y_values.max()))

            plot_width = self.plot_width if plot_width_func is None else plot_width_func
            plot_height = self.plot_height if plot_height_func is None else plot_height_func
            plt.figure(figsize=(plot_width, plot_height))
            ax = plt.gca()

            if below_segments_all:
                ax.add_collection(LineCollection(
                    below_segments_all,
                    colors=[self._rgb_to_mpl(production_rgb_line)],
                    linewidths=linewidth,
                    label='accommodated feed-in',
                ))
            if above_segments_all:
                ax.add_collection(LineCollection(
                    above_segments_all,
                    colors=[self._rgb_to_mpl(excess_rgb_line)],
                    linewidths=linewidth,
                    label='excess feed-in',
                ))

            ax.set_xlim(1, 8760)
            y_pad = max(1.0, 0.05 * (y_max_global - y_min_global if y_max_global > y_min_global else 1.0))
            ax.set_ylim(y_min_global - y_pad, y_max_global + y_pad)

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'{title} (iter {iter_label})')
            plt.legend(loc=legend_loc)
            plt.tight_layout()
            self._save_figure(
                os.path.join(self.dir_path_export, f'{export_name}_{scen}_iter{iter_label}.png'),
                plot_width,
                plot_height,
            )
            plt.close()

    def excessfeedin_worstnodes_bars(self,
                                    scen = 'pvalloc_LRG3_max',
                                    iter_list: list = None,
                                    n_bars: int | None = None,
                                    export_name = 'excessfeedin_worstnodes_bars',
                                    title = 'Nodes with the Most Excess Feed-in',
                                    bar_rgb = (214, 39, 40),
                                    n_highlight: int = 5,
                                    highlight_rgb_list = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189), (140, 86, 75)),
                                    plot_size_func = None,
                                    plot_width_func = None,
                                    plot_height_func = None,):

        iter_vals = self._resolve_iter_list(scen, iter_list)
        topo_counts = self._topo_egid_counts(scen)

        for iter_val in iter_vals:
            gridnode_df, iter_label = self._load_gridnode_iter_df(scen, iter_val)

            node_totals = (
                gridnode_df
                .group_by('grid_node')
                .agg([
                    pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'),
                    pl.col('feedin_atnode_kW').sum().alias('total_production_kW'),
                ])
                .join(topo_counts, on='grid_node', how='left')
                .with_columns([
                    (pl.col('total_loss_kW') / pl.col('total_production_kW')).alias('loss_per_production'),
                    (pl.col('total_loss_kW') / pl.col('n_egid')).alias('loss_per_house'),
                ])
            )

            bar_specs = [
                ('total_loss_kW', 'Total excess feed-in (kWh)'),
                ('loss_per_production', 'Excess / production (share)'),
                ('loss_per_house', 'Excess per house (kWh)'),
            ]

            # any node that ranks in the top n_highlight of *any* panel gets a
            # unique, persistent color so it can be spotted across all three
            # panels, regardless of which metric each one is sorted by
            top_nodes = []
            for col, _ in bar_specs:
                for node in (
                    node_totals.sort(col, descending=True)
                    .head(n_highlight)
                    .get_column('grid_node')
                    .to_list()
                ):
                    if node not in top_nodes:
                        top_nodes.append(node)

            highlight_colors = list(highlight_rgb_list)
            if len(top_nodes) > len(highlight_colors):
                highlight_colors += sns.color_palette('husl', n_colors=len(top_nodes) - len(highlight_colors))

            default_color = self._rgb_to_mpl(bar_rgb)
            node_color_map = {
                node: self._rgb_to_mpl(rgb)
                for node, rgb in zip(top_nodes, highlight_colors)
            }

            plot_width = self.plot_width if plot_width_func is None else plot_width_func
            plot_height = self.plot_height if plot_height_func is None else plot_height_func
            fig, axes = plt.subplots(1, 3, figsize=(plot_width * 3, plot_height))

            for ax, (col, sub_label) in zip(axes, bar_specs):
                top_df = node_totals.sort(col, descending=True)
                if n_bars is not None:
                    top_df = top_df.head(n_bars)
                nodes_in_plot = top_df.get_column('grid_node').to_list()
                node_labels = [str(v) for v in nodes_in_plot]
                values = top_df.get_column(col).to_list()
                bar_colors = [node_color_map.get(node, default_color) for node in nodes_in_plot]
                ax.barh(node_labels, values, color=bar_colors)
                ax.invert_yaxis()
                ax.set_xlabel(sub_label)
                ax.set_title(sub_label)

            if node_color_map:
                legend_handles = [
                    Patch(facecolor=node_color_map[node], label=f'node {node}')
                    for node in top_nodes
                ]
                fig.legend(
                    handles=legend_handles,
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.92),
                    ncol=min(len(legend_handles), 5),
                    frameon=False,
                    fontsize=9,
                    title=f'Nodes in the top {n_highlight} of any panel',
                )

            fig.suptitle(f'{title} (iter {iter_label})', y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.82])
            self._save_figure(
                os.path.join(self.dir_path_export, f'{export_name}_{scen}_iter{iter_label}.png'),
                plot_width * 3,
                plot_height,
            )
            plt.close()

        return top_nodes

    def excfeedin_freq_TS(self,
                        scen = 'pvalloc_LRG3_max',
                        freq = 'weekly',
                        title = 'Excess Feed-in - Frequency Time Series',
                        export_name = 'excfeedin_freq_TS',
                        gridnode_list: list = None,
                        n_worstnodes: int = None,
                        iter_list: list = None,
                        y_label_excess = 'Excess feed-in (kWh)',
                        y_label_share = 'Share (%)',
                        excess_rgb_line = (214, 39, 40),
                        nodes_share_rgb_line = (31, 119, 180),
                        houses_share_rgb_line = (44, 160, 44),
                        linewidth = 1.5,
                        opacity = 0.6,
                        legend_loc = 'upper left',
                        plot_width_func = None,
                        plot_height_func = None,):

        if freq not in ('daily', 'weekly'):
            raise ValueError("freq must be 'daily' or 'weekly'")
        period_hours = 24 if freq == 'daily' else 24 * 7
        x_label = 'Day of year' if freq == 'daily' else 'Week of year'

        iter_vals = self._resolve_iter_list(scen, iter_list)
        topo_counts = self._topo_egid_counts(scen)
        total_houses = topo_counts.get_column('n_egid').sum()

        for iter_val in iter_vals:
            gridnode_df, iter_label = self._load_gridnode_iter_df(scen, iter_val)
            gridnode_df = gridnode_df.with_columns(
                (((pl.col('t_int') - 1) // period_hours) + 1).alias('period')
            )
            total_nodes = gridnode_df.get_column('grid_node').n_unique()

            # excess feed-in trace, restricted to the selected node subset
            if gridnode_list is not None:
                nodes_for_excess = list(gridnode_list)
            elif n_worstnodes is not None:
                nodes_for_excess = (
                    gridnode_df
                    .group_by('grid_node')
                    .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
                    .sort('total_loss_kW', descending=True)
                    .head(n_worstnodes)
                    .get_column('grid_node')
                    .to_list()
                )
            else:
                nodes_for_excess = None

            excess_df = gridnode_df if nodes_for_excess is None else gridnode_df.filter(pl.col('grid_node').is_in(nodes_for_excess))
            period_excess = (
                excess_df
                .group_by('period')
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('excess_kWh'))
            )

            # node / house prevalence, always computed across all nodes in the scenario
            node_period_loss = (
                gridnode_df
                .group_by(['period', 'grid_node'])
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('node_period_loss_kWh'))
            )
            nodes_with_excess = (
                node_period_loss
                .filter(pl.col('node_period_loss_kWh') > 0)
                .join(topo_counts, on='grid_node', how='left')
                .group_by('period')
                .agg([
                    pl.len().alias('n_nodes_excess'),
                    pl.col('n_egid').sum().alias('n_houses_excess'),
                ])
            )

            period_stats = (
                gridnode_df
                .select('period')
                .unique()
                .join(period_excess, on='period', how='left')
                .join(nodes_with_excess, on='period', how='left')
                .fill_null(0)
                .with_columns([
                    (pl.col('n_nodes_excess') / total_nodes * 100).alias('node_share_pct'),
                    (pl.col('n_houses_excess') / total_houses * 100).alias('house_share_pct'),
                ])
                .sort('period')
            )

            periods = period_stats.get_column('period').to_numpy()
            excess_vals = period_stats.get_column('excess_kWh').to_numpy()
            node_share_vals = period_stats.get_column('node_share_pct').to_numpy()
            house_share_vals = period_stats.get_column('house_share_pct').to_numpy()

            plot_width = self.plot_width if plot_width_func is None else plot_width_func
            plot_height = self.plot_height if plot_height_func is None else plot_height_func
            _, ax1 = plt.subplots(figsize=(plot_width, plot_height))
            ax2 = ax1.twinx()

            line1, = ax1.plot(periods, excess_vals, color=self._rgb_to_mpl(excess_rgb_line), linewidth=linewidth, alpha=opacity, label='excess feed-in')
            line2, = ax2.plot(periods, node_share_vals, color=self._rgb_to_mpl(nodes_share_rgb_line), linewidth=linewidth, linestyle='--', alpha=opacity, label='nodes with excess (%)')
            line3, = ax2.plot(periods, house_share_vals, color=self._rgb_to_mpl(houses_share_rgb_line), linewidth=linewidth, linestyle=':', alpha=opacity, label='houses in excess nodes (%)')

            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label_excess)
            ax2.set_ylabel(y_label_share)
            ax2.set_ylim(0, 100)
            ax1.set_title(f'{title} (iter {iter_label})')
            ax1.legend(handles=[line1, line2, line3], loc=legend_loc)
            plt.tight_layout()
            self._save_figure(
                os.path.join(self.dir_path_export, f'{export_name}_{freq}_{scen}_iter{iter_label}.png'),
                plot_width,
                plot_height,
            )
            plt.close()


class BatteryBuffer_WorstWeek(_PlotExportMixin):
    # def worstnode_worstweek(self,
    #                         scen = 'pvalloc_LRG3_max',
    #                         title = 'Excess Feedin ("Worst Week")',
    #                         export_name = 'worstnode_worstweek',
    #                         gridnode = None,
    #                         excess_feedin_pegid = True,
    #                         x_label = 'Hour of year',
    #                         y_label = 'Excess Feed-in (kW)',
    #                         y_scaling = 1.0,
    #                         rgb_line = (200, 50, 50),
    #                         legend_loc = 'upper left',
    #                         plot_width_func = None,
    #                         plot_height_func = None,):

    #     topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
    #     gridnode_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))

    #     topo_rows = []
    #     for k, v in topo.items():
    #         topo_rows.append({
    #             'EGID': k,
    #             'grid_node': v['grid_node'],
    #         })
    #     topo_df = pl.DataFrame(topo_rows)

    #     if gridnode is None:
    #         worst_node = (
    #             gridnode_df
    #             .group_by('grid_node')
    #             .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
    #             .sort('total_loss_kW', descending=True)
    #             .get_column('grid_node')
    #             .item(0)
    #         )
    #     else:
    #         worst_node = gridnode

    #     worst_node_df = (
    #         gridnode_df
    #         .filter(pl.col('grid_node') == worst_node)
    #         .sort('t_int')
    #         .to_pandas()
    #         .reset_index(drop=True)
    #     )

    #     worst_node_df['loss_7d_kW'] = worst_node_df['feedin_atnode_loss_kW'].rolling(168).sum()
    #     worst_idx = worst_node_df['loss_7d_kW'].idxmax()

    #     worst_start = worst_node_df.loc[worst_idx, 't_int'] - 167
    #     worst_end   = worst_node_df.loc[worst_idx, 't_int']

    #     negid_worstnode = topo_df.filter(pl.col('grid_node') == worst_node).get_column('EGID').count()
    #     worst_week_df = worst_node_df.loc[worst_node_df['t_int'].between(worst_start, worst_end)].copy()
    #     if negid_worstnode <= 0:
    #         raise ValueError(f'No EGIDs mapped to grid node {worst_node}.')

    #     if excess_feedin_pegid is True:
    #         worst_week_df['feedin_atnode_loss_kW'] = (
    #             worst_week_df['feedin_atnode_loss_kW'] / negid_worstnode * y_scaling
    #         )

    #     # find the actual worst day (max daily loss) within the worst week
    #     _ww = worst_node_df.loc[worst_node_df['t_int'].between(worst_start, worst_end)].copy()
    #     _ww['day'] = (_ww['t_int'] - 1) // 24 + 1
    #     _worst_day = _ww.groupby('day')['feedin_atnode_loss_kW'].sum().idxmax()
    #     worst_peakweek1_start = (_worst_day - 1) * 24 + 1
    #     worst_peakweek1_end   = _worst_day * 24

    #     def _fmt_ch(val, d=1):
    #         return f'{val:,.{d}f}'.replace(',', "'")

    #     def _node_peak_stats(node, node_df_pl, topo_df_pl):
    #         ndf = (
    #             node_df_pl
    #             .filter(pl.col('grid_node') == node)
    #             .sort('t_int')
    #             .to_pandas()
    #             .reset_index(drop=True)
    #         )
    #         ndf['loss_7d_kW'] = ndf['feedin_atnode_loss_kW'].rolling(168).sum()
    #         idx = ndf['loss_7d_kW'].idxmax()
    #         w_start = ndf.loc[idx, 't_int'] - 167
    #         w_end   = ndf.loc[idx, 't_int']
    #         ww = ndf.loc[ndf['t_int'].between(w_start, w_end)].copy()
    #         ww['day'] = (ww['t_int'] - 1) // 24 + 1
    #         worst_day = ww.groupby('day')['feedin_atnode_loss_kW'].sum().idxmax()
    #         t_peak_start = (worst_day - 1) * 24 + 1
    #         t_peak_end   = worst_day * 24
    #         peak_date = pd.Timestamp('2025-01-01') + pd.to_timedelta(worst_day - 1, unit='D')
    #         peak_df = ndf.loc[ndf['t_int'].between(t_peak_start, t_peak_end)].copy()
    #         n_egid = topo_df_pl.filter(pl.col('grid_node') == node).get_column('EGID').count()
    #         return {
    #             'peak_loss':      peak_df['feedin_atnode_loss_kW'].sum(),
    #             'peak_netdemand': peak_df['netdemand_kW'].sum(),
    #             'n_egid':         n_egid,
    #             'peak_day_str':   peak_date.strftime('%d.%m.'),
    #         }

    #     top5_nodes = (
    #         gridnode_df
    #         .group_by('grid_node')
    #         .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
    #         .sort('total_loss_kW', descending=True)
    #         .head(5)
    #         .get_column('grid_node')
    #         .to_list()
    #     )

    #     replacements = {
    #         'figure_filename':      f'{export_name}_{scen}.png',
    #         'hist_figure_filename': f'hist_avgloss_pEGID_{scen}.png',
    #         'node1_color_rgb':      f'{rgb_line[0]},{rgb_line[1]},{rgb_line[2]}',
    #     }
    #     for i, node in enumerate(top5_nodes, start=1):
    #         s = _node_peak_stats(node, gridnode_df, topo_df)
    #         replacements[f'node_{i}']                = node
    #         replacements[f'peak_day_{i}']            = s['peak_day_str']
    #         replacements[f'total_excess_feedin_{i}'] = _fmt_ch(s['peak_loss'])
    #         replacements[f'n_houses_{i}']            = s['n_egid']
    #         replacements[f'avg_feedin_p_house_{i}']  = _fmt_ch(s['peak_loss'] / s['n_egid'])
    #         replacements[f'avg_demand_p_house_{i}']  = _fmt_ch(s['peak_netdemand'] / s['n_egid'])

    #     # also keep the original single-node keys (node 1 = worst node) for backwards compat
    #     replacements['worst_node'] = top5_nodes[0]

    #     self._write_latex_from_template(
    #         template_file='latex_table_template__worstnode_worstweek.txt',
    #         export_file='worstweek_node_peak.txt',
    #         replacements=replacements,
    #     )

    #     scen_color = (rgb_line[0] / 255, rgb_line[1] / 255, rgb_line[2] / 255)
    #     plot_width = self.plot_width if plot_width_func is None else plot_width_func
    #     plot_height = self.plot_height if plot_height_func is None else plot_height_func

    #     plt.figure(figsize=(plot_width, plot_height))
    #     sns.lineplot(
    #         data=worst_week_df,
    #         x='t_int',
    #         y='feedin_atnode_loss_kW',
    #         marker='',
    #         color=scen_color,
    #         linewidth=1.5,
    #         alpha=self.line_opacity,
    #         label=f'grid node {worst_node}',
    #     )

    #     # highlight the actual worst day with a red band behind the line
    #     ax = plt.gca()
    #     try:
    #         ax.axvspan(worst_peakweek1_start - 0.5, worst_peakweek1_end + 0.5, color='red', alpha=0.15, zorder=0)
    #     except Exception:
    #         pass
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     plt.title(title)
    #     plt.legend(title=None, loc=legend_loc)
    #     plt.tight_layout()
    #     self._save_figure(
    #         os.path.join(self.dir_path_export, f'{export_name}_{scen}.png'),
    #         plot_width,
    #         plot_height,
    #     )
    #     plt.close()

    def maxbatterycap_to_buffer(self,
                                scen = 'pvalloc_LRG3_max',
                                nodes: list = None,
                                start_date = '2025-01-01',
                                n_days_buffer: int = 14,
                                title = 'Battery Buffer Capacity',
                                export_name = 'maxbatterycap_to_buffer',
                                x_label_capacity = 'Buffer duration (days)',
                                x_label_empty = 'Days until battery empties',
                                y_label = 'Battery capacity needed (kWh)',
                                linewidth = 1.5,
                                marker = 'o',
                                opacity = 0.6,
                                legend_loc = 'upper left',
                                plot_width_func = None,
                                plot_height_func = None,):

        gridnode_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))
        topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
        topo_df_list = []
        for k,v in topo.items():
            rows = {
                'EGID':         k,
                'grid_node':    v.get('grid_node'),
                'inst_TF':      v['pv_inst']['inst_TF']
                }
            topo_df_list.append(rows)
        topo_df = pl.DataFrame(topo_df_list)


        if nodes is None:
            nodes = (
                gridnode_df
                .group_by('grid_node')
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
                .sort('total_loss_kW', descending=True)
                .head(5)
                .get_column('grid_node')
                .to_list()
            )

        t_max = int(gridnode_df.get_column('t_int').max())
        start_ts = pd.Timestamp(start_date)
        year_start = pd.Timestamp(year=start_ts.year, month=1, day=1)
        t_start = int((start_ts - year_start) / pd.Timedelta(hours=1)) + 1
        if not (1 <= t_start <= t_max):
            raise ValueError(f'start_date {start_date} falls outside the available t_int range (1..{t_max}) for scen {scen}.')

        days_range = list(range(1, n_days_buffer + 1))
        node_colors = dict(zip(nodes, sns.color_palette(n_colors=len(nodes))))

        capacities_by_node = {}
        empty_days_by_node = {}

        for node in nodes:
            node_df = (
                gridnode_df
                .filter(pl.col('grid_node') == node)
                .sort('t_int')
                .to_pandas()
                .set_index('t_int')
                .reindex(range(1, t_max + 1), fill_value=0.0)
            )
            loss_hourly = node_df['feedin_atnode_loss_kW'].to_numpy()
            # netdemand_kW already reflects demand net of self-consumption; only the
            # positive part draws down the battery, negative hours (still exporting) don't.
            drain_cumsum = np.cumsum(node_df['netdemand_kW'].clip(lower=0).to_numpy())

            n_pv_houses_node = topo_df.filter(
                (pl.col('grid_node') == node) & 
                (pl.col('inst_TF') == True)
            ).get_column('EGID').count()

            capacities = []
            empty_days = []
            for d in days_range:
                window_end = t_start + d * 24 - 1
                capacity = float(loss_hourly[t_start - 1: window_end].sum())

                capacities.append(capacity / n_pv_houses_node if n_pv_houses_node > 0 else np.nan)

                window_end_idx = min(window_end, t_max)
                base_cum = drain_cumsum[window_end_idx - 1] if window_end_idx >= 1 else 0.0
                future_cum = drain_cumsum[window_end_idx:] - base_cum
                idx = np.searchsorted(future_cum, capacity, side='left')

                if capacity <= 0:
                    empty_days.append(0.0)
                elif idx >= len(future_cum):
                    empty_days.append(np.nan)
                else:
                    prev_cum = future_cum[idx - 1] if idx > 0 else 0.0
                    step_drain = future_cum[idx] - prev_cum
                    frac = (capacity - prev_cum) / step_drain if step_drain > 0 else 1.0
                    empty_days.append((idx + frac) / 24.0)

            capacities_by_node[node] = capacities
            empty_days_by_node[node] = empty_days

        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(plot_width * 2, plot_height), sharey=True)

        for node in nodes:
            color = node_colors[node]
            ax_left.plot(days_range, capacities_by_node[node], color=color, marker=marker, linewidth=linewidth, alpha=opacity, label=f'node {node}')
            ax_right.plot(empty_days_by_node[node], capacities_by_node[node], color=color, marker=marker, linewidth=linewidth, alpha=opacity)

        ax_left.set_xlabel(x_label_capacity)
        ax_left.set_ylabel(y_label)
        ax_left.set_title('Capacity to buffer excess feed-in')
        ax_left.set_xlim(1, n_days_buffer)
        ax_left.legend(loc=legend_loc)

        ax_right.set_xlabel(x_label_empty)
        ax_right.set_title('Time to fully discharge')

        fig.suptitle(f'{title} ({scen}, start {start_ts.date()})')
        plt.tight_layout()
        self._save_figure(
            os.path.join(self.dir_path_export, f'{export_name}_{scen}.png'),
            plot_width * 2,
            plot_height,
        )
        plt.close()
