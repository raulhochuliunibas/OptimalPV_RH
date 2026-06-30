import sys
import os as os
import shutil
import copy
import numpy as np
import pandas as pd
import polars as pl
import glob
import  sqlite3

import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns


# GENERAL SETTINGS
class static_plotter_class:
    def __init__(self, export_dir_list=None):
        if export_dir_list is None:
            export_dir_list = ['data', 'visualization_static_wpaper']
        self.data_path  = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data')
        self.paper_path = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'paper_publication')
        self.dir_path        = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'visualization_static_wpaper')
        self.dir_path_export = os.path.join('C:', os.sep, 'Models', 'OptimalPV_RH', *export_dir_list)
        os.makedirs(self.dir_path_export, exist_ok=True)

        self.scen_default_color_map = {
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'scenario2': (50, 200, 50),
            'scenario3': (50, 50, 200),
            'scenario4': (200, 200, 50),
        }
        self.simple_scen_name_mapping = {
            'pvalloc_29nbfs_LRG2_max': 'default scenario',
            'scenario2': 'Scenario 2',
            'scenario3': 'Scenario 3',
            'scenario4': 'Scenario 4',
        }
        self.scen_default_linedash_marker_map = {}
        self.line_opacity = 0.8
        self.plot_width = 8
        self.plot_height = 4
        self.plot_dpi = 500
        self.show_plt_TF = False


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

    def _copy_csv_to_export(self, csv_source_path):
        if not str(csv_source_path).lower().endswith('.csv'):
            return
        if not os.path.isfile(csv_source_path):
            return
        target_path = os.path.join(self.dir_path_export, os.path.basename(csv_source_path))
        if os.path.abspath(csv_source_path) == os.path.abspath(target_path):
            return
        shutil.copy2(csv_source_path, target_path)

    def _get_scenario_label(self, scen):
        return self.simple_scen_name_mapping.get(scen, scen)

    def _get_scenario_linedash_marker(self, scen):
        if scen in self.scen_default_linedash_marker_map:
            linedash = self.scen_default_linedash_marker_map[scen][0]
            marker   = self.scen_default_linedash_marker_map[scen][1]
        else: 
            linedash = 'solid'
            marker = ''
        return linedash, marker


    def copy(self):
        # Return an isolated clone so wrapper-level mutations do not affect the caller instance.
        return copy.deepcopy(self)

    def _save_figure(self, export_path, plot_width=None, plot_height=None):
        fig = plt.gcf()
        if plot_width is not None and plot_height is not None:
            fig.set_size_inches(plot_width, plot_height, forward=True)
        plt.savefig(export_path, dpi=self.plot_dpi)
    

    def copy_standalone_graphs_to_presentation_dir(self, 
                                                  graphs_list =[
                                                      'bfe_ep2050_netz_MNA_notitle.PNG', 
                                                      'calib_rfr2_predvsactual.png', 
                                                      'EGID_solkat_example.png', 
                                                      'eth_summerschool_scenario_agg_notitle.png', 
                                                      'grid-levels 1.png', 
                                                      'grid-levels.png', 
                                                      'newspaper_header2_optimalpv.png', 
                                                      'newspaper_header3.png',
                                                      'pvinstcost_table.png', 
                                                      'wp_topo_DSO_example_1_3.png', 
                                                      'wp_topo_DSO_grid_large_example_1_3.png', 
                                                      'wp_topo_roof_solkat_1_3.png', 
                                                      'EGID_HOY_profile.png',
                                                      ]
                                                    ):
        for graph in graphs_list:
            src_path = os.path.join(self.paper_path, '0_standalone_graphs', graph)
            dst_path = os.path.join(self.dir_path_export, graph)
            if not os.path.exists(src_path):
                print(f'SKIP (not found): {src_path}')
                continue
            if os.path.isdir(dst_path):
                print(f'SKIP (dst is a directory): {dst_path}')
                continue
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy(src_path, dst_path)
                print(f'Copied {graph} to {dst_path}')
            except OSError as e:
                print(f'ERROR copying {graph}: {e}')
                print(f'  src exists={os.path.exists(src_path)}, dst_dir exists={os.path.isdir(os.path.dirname(dst_path))}')
                print(f'  src={src_path}')
                print(f'  dst={dst_path}')                                              



    # SCENARIO - based plots, tables + data ====================================================================== 
    def plot_EGID_pvprod_demand_HOY(self, 
                                export_name = 'plot_EGID_pvprod_demand_HOY',
                                hours_incl_list=list(range(4920 + 3*24, 4920 + 5*24)),
                                scen = 'pvalloc_LRG3_max',
                                egid_plot = None, 
                                plot_cols_incl_list = None,
                                title = 'Profiles Single House',
                                x_label = 't (hours of year)',
                                y_label = 'kW',
                                prod_rgb = (185, 150, 215),
                                feedin_rgb = (120, 170, 235),
                                demand_rgb = (204, 85, 0),
                                y_lower_limit = -2,
                                y_upper_padding = 1.5,
                                export_plots = True,
                                daynightbands = True,
                                plot_width_func=None,
                                plot_height_func=None,
    ):

        topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'topo_egid.json')))
        egids = list(topo.keys())          

        if egid_plot is None:
            for egid in egids:
                if topo[egid]['pv_inst']['info_source'] == 'pv_df':
                    if len(topo[egid]['solkat_partitions']) > 1:
                        egid_plot = egid
                        break
        
        subdf_paths = glob.glob(f'{self.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')

        for path in subdf_paths:
            subdf = pl.read_parquet(path)
            if egid_plot in subdf['EGID'].unique():
                egid_subdf = subdf.filter(pl.col('EGID') == egid_plot)
                break
        
        egid_subdf.shape[0]/8760
        topo[egid_plot]['solkat_partitions']

        egid_subdf.columns
        egid_agg = egid_subdf.group_by('EGID', 't','t_int').agg([
            pl.col('df_uid').count().alias('n_dfuid'),
            pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
            pl.col('demand_kW').first().alias('demand_kW'),
            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
        ])

        # calc selfconsumption
        egid_agg = egid_agg.sort(['EGID', 't_int'], descending = [False, False])

        selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * 1.0

        egid_agg = egid_agg.with_columns([        
            selfconsum_expr.alias("selfconsum_kW"),
            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
        ])

        # PLOT -------------------------------
        egid_plot_df = (
            egid_agg
            .filter(pl.col('t_int').is_in(hours_incl_list))
            .to_pandas()
            .sort_values('t_int')
        )

        if egid_plot_df.empty:
            print(f'No data found for EGID {egid_plot} in the selected hours.')
            return

        if plot_cols_incl_list is None:
            plot_cols_incl_list = ['demand_kW', 'pvprod_kW', 'netfeedin_kW']

        allowed_cols = {'demand_kW', 'pvprod_kW', 'netfeedin_kW'}
        plot_cols_incl_list = [col for col in plot_cols_incl_list if col in allowed_cols]
        if not plot_cols_incl_list:
            raise ValueError('plot_cols_incl_list must include at least one of: demand_kW, pvprod_kW, netfeedin_kW')

        def rgb_to_mpl_color(rgb_value):
            if not isinstance(rgb_value, (list, tuple, np.ndarray)) or len(rgb_value) != 3:
                raise ValueError('Expected an RGB tuple/list with exactly 3 values.')
            rgb_array = np.asarray(rgb_value, dtype=float)
            if np.nanmax(rgb_array) > 1.0:
                rgb_array = rgb_array / 255.0
            return tuple(rgb_array.tolist())

        def build_line_segments(x_vals, y_vals):
            segments = []
            for idx in range(len(x_vals) - 1):
                x0 = float(x_vals[idx])
                x1 = float(x_vals[idx + 1])
                y0 = float(y_vals[idx])
                y1 = float(y_vals[idx + 1])
                segments.append([(x0, y0), (x1, y1)])
            return segments

        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        plt.figure(figsize=(plot_width, plot_height))

        ax = plt.gca()
        x_values = egid_plot_df['t_int'].to_numpy()

        series_specs = [
            ('demand_kW', demand_rgb, 'Demand', 1.6, '-'),
            ('pvprod_kW', prod_rgb, 'PV production', 1.6, '-'),
            ('netfeedin_kW', feedin_rgb, 'Net feed-in', 1.8, '-'),
        ]

        for col_name, rgb_value, label, linewidth, linestyle in series_specs:
            if col_name not in plot_cols_incl_list:
                continue
            if col_name not in egid_plot_df.columns:
                continue

            y_values = egid_plot_df[col_name].to_numpy()
            segments = build_line_segments(x_values, y_values)
            if not segments:
                continue

            collection = LineCollection(
                segments,
                colors=[rgb_to_mpl_color(rgb_value)],
                linewidths=linewidth,
                linestyles=linestyle,
                label=label,
                zorder=3,
            )
            ax.add_collection(collection)

        y_columns = [col for col in plot_cols_incl_list if col in egid_plot_df.columns]
        y_min = float(np.nanmin(egid_plot_df[y_columns].to_numpy()))
        y_max = float(np.nanmax(egid_plot_df[y_columns].to_numpy()))
        ax.set_xlim(float(x_values.min()) - 0.5, float(x_values.max()) + 0.5)
        ax.set_ylim(min(float(y_lower_limit), y_min), y_max + float(y_upper_padding))

        if daynightbands is True:
            daynightbands = {'day_start_hour': 7, 'day_end_hour': 19}

        if isinstance(daynightbands, dict):
            day_start_hour = int(daynightbands.get('day_start_hour', 7))
            day_end_hour = int(daynightbands.get('day_end_hour', 19))
            day_color = daynightbands.get('day_color', '#fff7cc')
            night_color = daynightbands.get('night_color', '#e6f0ff')
            band_alpha = float(daynightbands.get('alpha', 0.25))

            t_min = int(egid_plot_df['t_int'].min())
            t_max = int(egid_plot_df['t_int'].max())

            def is_day_hour(t_val):
                hour_of_day = ((int(t_val) - 1) % 24) + 1
                return day_start_hour <= hour_of_day < day_end_hour

            segment_start = t_min
            prev_is_day = is_day_hour(t_min)

            for t_val in range(t_min + 1, t_max + 1):
                curr_is_day = is_day_hour(t_val)
                if curr_is_day != prev_is_day:
                    segment_color = day_color if prev_is_day else night_color
                    ax.axvspan(segment_start - 0.5, t_val - 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)
                    segment_start = t_val
                    prev_is_day = curr_is_day

            segment_color = day_color if prev_is_day else night_color
            ax.axvspan(segment_start - 0.5, t_max + 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)

        for line in ax.lines:
            line.set_zorder(4)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title if title is not None else f'EGID {egid_plot} PV production, demand and net feed-in')
        plt.legend(title=None)
        plt.tight_layout()
        # plt.show() 

        if export_plots:
            self._save_figure(os.path.join(self.dir_path_export, f'{export_name}.png'), plot_width, plot_height)
        plt.close()


    def plot_ind_line_demand(self,
                             name_dir_export ,
                             hours_incl_list,
                             export_name,
                             select_egids = None,
                             n_egids_by_group = {
                                 'sfh_rur_hpF': (0, 'SFH', 'Rural',     'no_heatpump'),
                                 'sfh_rur_hpT': (0, 'SFH', 'Rural',     'heatpump'),
                                 'sfh_sub_hpF': (0, 'SFH', 'Suburban',  'no_heatpump'),
                                 'sfh_sub_hpT': (0, 'SFH', 'Suburban',  'heatpump'),
                                 'sfh_urb_hpF': (0, 'SFH', 'Urban',     'no_heatpump'),
                                 'sfh_urb_hpT': (0, 'SFH', 'Urban',     'heatpump'),
                                 'mfh_rur_hpF': (0, 'MFH', 'Rural',     'no_heatpump'),
                                 'mfh_rur_hpT': (0, 'MFH', 'Rural',     'heatpump'),
                                 'mfh_sub_hpF': (0, 'MFH', 'Suburban',  'no_heatpump'),
                                 'mfh_sub_hpT': (0, 'MFH', 'Suburban',  'heatpump'),
                                 'mfh_urb_hpF': (0, 'MFH', 'Urban',     'no_heatpump'),
                                 'mfh_urb_hpT': (0, 'MFH', 'Urban',     'heatpump'),
                             },
                             export_plots = True, 
                             plot_width_func=None,
                             plot_height_func=None):
        
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        file_path = os.path.join(self.data_path, 'pvalloc', name_dir_export)

        topo    = json.load(open(os.path.join(file_path, 'topo_egid.json')))
        npv_df  = pd.read_parquet(os.path.join(file_path, 'zMC1', 'npv_df.parquet'))
        topo_subdf_paths    = glob.glob(f'{self.data_path}/pvalloc/{name_dir_export}/topo_time_subdf/topo_subdf_*.parquet')

        sfhmfh_map_list = []
        for k,v in topo.items():
            sfhmfh_map_list.append({
                'EGID': k, 
                'sfhmfh_typ': v['gwr_info']['sfhmfh_typ'], 
                'are_typ': v['gwr_info']['are_typ'],
                'gwaerzh1': v['gwr_info']['gwaerzh1'],
                'genh1': v['gwr_info']['genh1'],
            })
        sfhmfh_map_df = pd.DataFrame(sfhmfh_map_list)
        sfhmfh_map_df['heatpump_TF'] = np.where(sfhmfh_map_df['gwaerzh1'].isin(['7410', '7411']), 'heatpump', 'no_heatpump')

        npv_df_info = npv_df.merge(sfhmfh_map_df, on='EGID', how='left')

        def get_n_egids_filtered_df(df, n, sfhmfh, are, heatpump):
            df_filt = df[
                (df['sfhmfh_typ'] == sfhmfh) &
                (df['are_typ'] == are) &
                (df['heatpump_TF'] == heatpump)
            ]
            egid_list = df_filt['EGID'].unique().tolist()[:n]
            return list(df.loc[df['EGID'].isin(egid_list), 'EGID'])
        n_egids_list = []

        for k,v in n_egids_by_group.items():
            n_egids, sfhmfh, are, heatpump = v
            if n_egids > 0:
                get_egids = get_n_egids_filtered_df(npv_df_info, n_egids, sfhmfh, are, heatpump)
                n_egids_list.extend(get_egids)

        if select_egids is not None and len(select_egids) > 0:
            filter_egids_subdf = select_egids
        else:
            filter_egids_subdf = n_egids_list

        # sfh_sub_hpT = get_n_egids_filtered_df(npv_df_info, 10, 'SFH', 'Suburban', 'heatpump') 
        # sfh_sub_hpF = get_n_egids_filtered_df(npv_df_info, 0, 'SFH', 'Suburban', 'no_heatpump')         
        # filter_egids_subdf = sfh_sub_hpT + sfh_sub_hpF

        topo_subdf_list = []
        for path in topo_subdf_paths:
            topo_subdf = pl.read_parquet(path)
            topo_filtr = topo_subdf.filter(pl.col('EGID').is_in(filter_egids_subdf))
            if topo_filtr.shape[0] > 0:
                topo_subdf_list.append(topo_filtr)
                topo_subdf_list.append(topo_filtr)


        topo_subdf = pl.concat(topo_subdf_list)

        # --- first df_uid per EGID ---
        topo_subdf_first = topo_subdf.group_by('EGID').agg([
            pl.first('df_uid').alias('df_uid')
        ]).to_pandas()

        # --- convert full topo_subdf to pandas for seaborn plotting ---
        topo_subdf_pd = topo_subdf.to_pandas()

        # --- prepare two plot variants: week and full year ---
        plot_variants = [
            {"hours": hours_incl_list, "suffix": "_week"},
            {"hours": None, "suffix": "_year"}  # all hours
        ]

        for variant in plot_variants:
            hours = variant["hours"]
            suffix = variant["suffix"]

            plt.figure(figsize=(plot_width, plot_height))
            np.random.seed(42)
            n_pairs = topo_subdf_first.shape[0]
            random_colors = [tuple(np.random.rand(3)) for _ in range(n_pairs)]

            single_values_list = []

            for i, row in topo_subdf_first.iterrows():
                egid = row['EGID']
                df_uid = row['df_uid']
                sfhmfh = topo[str(egid)]['gwr_info']['sfhmfh_typ']
                are_typ = topo[str(egid)]['gwr_info']['are_typ']
                heatpump_TF = 'heatpump' if topo[str(egid)]['gwr_info']['gwaerzh1'] in ['7410', '7411'] else 'no_heatpump'

                df_plot = topo_subdf_pd.loc[
                    (topo_subdf_pd['EGID'] == egid) &
                    (topo_subdf_pd['df_uid'] == df_uid)
                ].copy()

                # optional filtering for week hours
                if hours is not None and 't_int' in df_plot.columns:
                    df_plot = df_plot.loc[df_plot['t_int'].isin(hours)]

                if df_plot.shape[0] == 0:
                    continue  # skip empty

                if export_plots: 
                    sns.lineplot(
                        data=df_plot,
                        x='t_int' if 't_int' in df_plot.columns else np.arange(len(df_plot)),
                        y='demand_kW',
                        # color=color,
                        # label=f"EGID {egid} ({sfhmfh}, {are_typ}, {heatpump_TF})",
                        label=f"{sfhmfh}, {are_typ}, {heatpump_TF}",
                        alpha=self.line_opacity,
                        linewidth=1.5,
                    )

                # collect single values
                egid = egid
                garea = topo[str(egid)]['gwr_info']['garea']
                TotalPower = npv_df.loc[npv_df['EGID'] == egid, 'TotalPower'].values[0]
                NPV = npv_df.loc[npv_df['EGID'] == egid, 'NPV_uid'].values[0]
                total_demand_kWh = df_plot['demand_kW'].sum()
                total_pvprod_kWh = df_plot['pvprod_kW'].sum() if 'pvprod_kW' in df_plot.columns else 0

                single_values_list.append({
                    'EGID': egid,
                    'GAREA': garea,
                    'TotalPower': TotalPower,
                    'NPV': NPV,
                    'Total_Demand_kWh': total_demand_kWh,
                    'Total_PVProd_kWh': total_pvprod_kWh,
                    'sfhmfh_typ': sfhmfh,
                    'are_typ': are_typ,
                    'heatpump_TF': heatpump_TF,
                })

            if export_plots:
                # plot export
                plt.xlabel('Hour (t_int)' if hours is not None else 'Index')
                plt.ylabel('Demand (kW)')
                plt.title(f'Individual Demand Profiles ({suffix.strip("_")})')
                plt.legend()
                plt.tight_layout()
                self._save_figure(os.path.join(self.dir_path_export, f"{export_name}{suffix}.png"), plot_width, plot_height)
                plt.close() 

        # export single values
        single_values_df = pd.DataFrame(single_values_list)
        single_values_df.to_csv(os.path.join(self.dir_path_export, f"{export_name}_single_values.csv"), index=False)


    def get_single_values(self, 
                          csv_file,
                          scen_incl_list,
                          n_iter_range_list,
            ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)



        df_comp = df.loc[
            (df['scen'].isin(scen_incl_list
                )) & 
            (df['n_iter'].isin(n_iter_range_list
                )),
            :].copy()
        
        for scen in scen_incl_list:

            # print losses
            scen_loss = df_comp.loc[df_comp['scen'] == scen, 'feedin_atnode_loss_kW'].values[0]
            print_str = f'{scen} loss: {round(scen_loss,1)} kWh/y'
            print(f'\n{print_str}\n')
            
            with open (os.path.join(self.dir_path_export, "loss_comparison_single_values.txt"), 'a') as f:
                f.write(f'{print_str}\n')

            # # print topo egid count
            # topo_path = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'pvalloc', scen , 'topo_egid.json')
            # topo = json.load(open(topo_path))

            # print_str = f'{scen} topo_egid_count: {len(topo)}'
            # print(f'\n{print_str}\n')
            # with open (os.path.join(self.dir_path, "loss_comparison_single_values.txt"), 'a') as f:
            #     f.write(f'{print_str}\n')


    # def NPVhist_DataSampleSummary(self, 
    def NPVhist(self, 
                                        scen = 'pvalloc_29nbfs_LRG2_max',
                                        export_name = None,
                                        npv_hist_width = 3.85,
                                        npv_hist_height= 3.4,
                                        npv_hist_xrange = (-1e5, 4.75e5),
                                        title = 'NPV Distribution',
                                        x_label = 'NPV (CHF)',
                                        y_label = 'Count',
                                        negative_rgb = (214, 39, 40),
                                        positive_rgb = (31, 119, 180),
                                        plot_width_func = None,
                                        plot_height_func = None,
                                        ):

            # npv_df = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'npv_df.parquet'))
            npv_df = pd.read_parquet(os.path.join(self.dir_path, f'npv_df_1_{scen}.parquet'))

            if 'NPV_uid_before_subsidy' not in npv_df.columns:
                print("Column 'NPV_uid_before_subsidy' not found in npv_df.")
                return

            df_plot = npv_df.loc[npv_df['NPV_uid_before_subsidy'].notna(), ['NPV_uid_before_subsidy']].copy()
            if df_plot.empty:
                print("No values found in 'NPV_uid_before_subsidy' for histogram.")
                return

            mean_val = df_plot['NPV_uid_before_subsidy'].mean()
            median_val = df_plot['NPV_uid_before_subsidy'].median()

            def rgb_to_mpl_color(rgb_value):
                if not isinstance(rgb_value, (list, tuple, np.ndarray)) or len(rgb_value) != 3:
                    raise ValueError('Expected an RGB tuple/list with exactly 3 values.')
                rgb_array = np.asarray(rgb_value, dtype=float)
                if np.nanmax(rgb_array) > 1.0:
                    rgb_array = rgb_array / 255.0
                return tuple(rgb_array.tolist())

            negative_color = rgb_to_mpl_color(negative_rgb)
            positive_color = rgb_to_mpl_color(positive_rgb)

            plot_width = self.plot_width if plot_width_func is None else plot_width_func
            plot_height = self.plot_height if plot_height_func is None else plot_height_func


            values = df_plot['NPV_uid_before_subsidy'].to_numpy()
            data_min = float(np.nanmin(values))
            data_max = float(np.nanmax(values))
            hist_range = (data_min, data_max)
            if npv_hist_xrange is not None and len(npv_hist_xrange) == 2:
                input_min = float(npv_hist_xrange[0])
                input_max = float(npv_hist_xrange[1])
                hist_min = max(input_min, data_min)
                hist_max = min(input_max, data_max)
                if hist_min < hist_max:
                    hist_range = (hist_min, hist_max)

            bins = np.linspace(hist_range[0], hist_range[1], 41)
            counts, bin_edges = np.histogram(values, bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bar_colors = [negative_color if center < 0 else positive_color for center in bin_centers]

            plt.figure(figsize=(plot_width, plot_height))
            ax = plt.gca()
            ax.bar(
                bin_edges[:-1],
                counts,
                width=np.diff(bin_edges),
                align='edge',
                color=bar_colors,
                edgecolor='white',
                linewidth=0.2,
                alpha=0.75,
            )
            ax.axvline(0, color='black', linestyle=':', linewidth=1.0, label='Zero')
            summary_line_color = 'black'
            ax.axvline(mean_val, color=summary_line_color, linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:,.0f}')
            ax.axvline(median_val, color=summary_line_color, linestyle='-', linewidth=1.5, label=f'Median: {median_val:,.0f}')
            ax.set_xlim(hist_range[0], hist_range[1])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend(title=None)
            plt.tight_layout()
            # plt.show()
            if export_name is None:
                export_file = f'{scen}_npv_df_hist.png'
            else:
                export_file = export_name if os.path.splitext(export_name)[1] else f'{export_name}.png'
            self._save_figure(os.path.join(self.dir_path_export, export_file), plot_width, plot_height)
            plt.close()
        
    def DataSampleSummary(self, 
                                        scen = 'pvalloc_LRG3_max',
                                        ):
            # get table for data summary
            # SAMPLE ----------------------------------------
            pvalloc_sett = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'pvalloc_sett.json')))
            bfs_numbers_sample = pvalloc_sett.get('bfs_numbers')

            topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'topo_egid.json')))
            gridnode_df = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))
            
            topo_gwr_summary_list = []
            for k, v in topo.items():
                topo_gwr_summary_list.append({
                    'EGID':           k,
                    'grid_node':      v['grid_node'],
                    'bfs':            v['gwr_info']['bfs'],
                    'garea':          v['gwr_info']['garea'],
                    'heating_system': v['gwr_info']['heating_system'],
                    'info_source':    v['pv_inst']['info_source'],
                    'TotalPower':    v['pv_inst']['TotalPower'],

                })
            topo_gwr_summary_df = pd.DataFrame(topo_gwr_summary_list)
            
            topo_solkat_summary_list = []
            for k, v in topo.items():
                for k_solkat, v_solkat in v.get('solkat_partitions', {}).items():
                    topo_solkat_summary_list.append({
                       'EGID': k,
                       'duid': k_solkat,
                       'FLAECHE': v_solkat['FLAECHE'],
                       'STROMERTRAG': v_solkat['STROMERTRAG'],
                   }) 
            topo_solkat_summary_df = pd.DataFrame(topo_solkat_summary_list)

            constrcapa = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'constrcapa.parquet'))



            # ALL CH ------------------------------------------

            # import GWR
            GWR_GKLAS = [ '1110' , '1121', '1122', ]
            GWR_STAT = ['1004']
            # get ALL BUILDING data
            # select cols
            query_columns = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE',
                                            'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ',
                                            'GANZWHG',
                                            'GWAERZH1', 'GENH1',# 'GWAERSCEH1', 'GWAERDATH1',
                                            'GWAERZH2', 'GENH2',# 'GWAERSCEH2', 'GWAERDATH2',
                                            'GEBF', 'GAREA']
            query_columns_str = ', '.join(query_columns)

            conn = sqlite3.connect(f'{self.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM building')
            sqlrows = cur.fetchall()
            conn.close()

            gwr_allch_raw = pl.DataFrame(sqlrows, schema = query_columns)
            gwr_allch_raw = gwr_allch_raw.with_columns([
                pl.col('GGDENR').cast(pl.Int32),
            ])
            gwr_allch_houses = gwr_allch_raw.filter(
                (pl.col('GSTAT').is_in(GWR_STAT)) 
            )
            gwr_allch_resid = gwr_allch_raw.filter(
                (pl.col('GSTAT').is_in(GWR_STAT)) &
                (pl.col('GKLAS').is_in(GWR_GKLAS))
            )

            # import PV
            pv_pl = pl.read_parquet(os.path.join(self.data_path, 'input_split_data_geometry', 'pv_pq.parquet'))
            pv_allch = pv_pl.filter(pl.col('TotalPower') < 30)
            pv_pl = pv_pl.with_columns([
                pl.col('BeginningOfOperation').str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            ])            
            result = (
                pv_pl
                .with_columns(pl.col('BeginningOfOperation').dt.year().alias('year'))
                .filter(pl.col('year').is_between(2021, 2025))
                .group_by('year')
                .agg((pl.col('TotalPower').sum() / 1000).alias('TotalPower_sum'))
                .sort('year')
            )

            # summary stats
            allch_n_municipalities = pd.Series(gwr_allch_raw['GGDENR'].n_unique())
            allch_n_houses = pd.Series(gwr_allch_houses['EGID'].n_unique()).values[0]
            allch_n_egids = pd.Series(gwr_allch_resid['EGID'].n_unique()).values[0]
            # allch_grid_nodes = na
            allch_pvcapacity_mw_2024 = pv_allch['TotalPower'].sum() / 1000
            allch_pvcapacity_mw_2050 = 37.5 * 1000 * 0.4617 # 37.5 GW target, 46.17% on class1 buildings



            # ALLDSO sample -------------------------------
            ALLDSO_bfs_list = [
                # RURAL 
                2612, 2889, 2883, 2621, 2622,
                2620, 2615, 2614, 2616, 2480,
                2617, 2611, 2788, 2619, 2783, 2477, 
                2790, 2426, 2428, 2893, 
                2885, 2852, 2491, 2502, 2499,
                2585,
                # SUBURBAN
                2613, 2782, 2618, 2786, 2785, 
                2772, 2761, 2743, 2476, 2768,
                2471, 2481, 2775, 2764, 2771, 
                2763, 2473, 2475, 2474, 2472, 
                2478, 2830, 2766, 2767, 2774, 
                2422, 2792, 2787, 2789, 2479, 
                2834, 2833, 
                2579, 2582, 2586, 2500, 2501, 
                2493, 2497, 2495, 2584, 2573, 
                2572, 2576, 2583,
                # URBAN
                2773, 2769, 2770,
                2762, 2765, 
                2829, 2831, 
                2581,
            ]

            preprep_gwr_all_buildilngs = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr_all_building_df.parquet")
            preprep_gwr                = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr.parquet")
            solkat                     = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\solkat.parquet")
            
            preprep_gwr['GGDENR'] = preprep_gwr['GGDENR'].astype(int)
            solkat['BFS_NUMMER'] = solkat['BFS_NUMMER'].astype(int)

            # ALLDSO specific


            # ALLDSO from CH GWR import               
            dsonodes_df                = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\dsonodes_df.parquet")
            gwr_alldso_houses = gwr_allch_raw.filter(
                (pl.col('GSTAT').is_in(GWR_STAT)) & 
                (pl.col('GGDENR').is_in(ALLDSO_bfs_list))
            )
            gwr_alldso_resid = gwr_allch_raw.filter(
                (pl.col('GSTAT').is_in(GWR_STAT)) &
                (pl.col('GKLAS').is_in(GWR_GKLAS)) &
                (pl.col('GGDENR').is_in(ALLDSO_bfs_list))
            )
            pv_alldso = pv_allch.filter(pl.col('BFS_NUMMER').is_in([str(bfs) for bfs in ALLDSO_bfs_list]))

            # summary stats
            alldso_n_municipalities = pd.Series(gwr_alldso_houses['GGDENR']).nunique()
            alldso_n_houses = preprep_gwr_all_buildilngs.loc[preprep_gwr_all_buildilngs['BFS_NO'].isin(ALLDSO_bfs_list),'EGID'].nunique()
            alldso_n_egids = preprep_gwr.loc[preprep_gwr['GGDENR'].isin(ALLDSO_bfs_list),'EGID'].nunique()
            alldso_n_grid_nodes = dsonodes_df['grid_node'].nunique()
            alldso_pvcapacity_mw_2024= pv_alldso['TotalPower'].sum()/1000
            alldso_pvcapacity_mw_2050 = 37.5 * 1000 * 0.4617 * (alldso_n_houses / allch_n_houses) # 37.5 GW target, 46.17% on class1 buildings, scaled by share of ALLDSO EGIDs in all CH EGIDs
            
            recalc_alldso_n_houses = pd.Series(gwr_alldso_houses['EGID']).nunique()
            recalc_alldso_n_egids = pd.Series(gwr_alldso_resid['EGID']).nunique()

            # CHECKS
            print(f'alldso: alldso_n_houses {alldso_n_houses}, \trecalc_alldso_n_houses {recalc_alldso_n_houses}')
            print(f'alldso: alldso_n_egids  {alldso_n_egids}, \t\trecalc_alldso_n_egids  {recalc_alldso_n_egids}')


            # SAMPLE --------------------------------------------
            LRG_bfs_list = [
                # RURAL 
                2612, 2889, 2883, 2621, 2622,
                2620, 2615, 2614, 2616, 2480,
                2617, 2611, 2788, 2619, 2783, 2477, 
                # SUBURBAN
                2613, 2782, 2618, 2786, 2785, 
                2772, 2761, 2743, 2476, 2768,
                # URBAN
                2773, 2769, 2770,
                ]
            
            gwr_n_houses = gwr_allch_raw.filter(
                (pl.col('GSTAT').is_in(GWR_STAT)) & 
                (pl.col('GGDENR').is_in(LRG_bfs_list))
            )
            gwr_n_resid = gwr_allch_raw.filter(
                (pl.col('GSTAT').is_in(GWR_STAT)) &
                (pl.col('GKLAS').is_in(GWR_GKLAS)) &
                (pl.col('GGDENR').is_in(LRG_bfs_list))
            )
            pv_nsample = pv_allch.filter(pl.col('BFS_NUMMER').is_in([str(bfs) for bfs in LRG_bfs_list]))



            recalc_n_municipalities    = topo_gwr_summary_df['bfs'].nunique()
            recalc_n_houses =      pd.Series(gwr_n_houses['EGID']).nunique()
            recalc_n_egids =      pd.Series(gwr_n_resid['EGID']).nunique()
            recalc_pvcapacity =     pv_nsample['TotalPower'].sum() / 1000
            recalc_capacity_mw_2050 = 37.5 * 1000 * 0.4617 * (recalc_n_houses / allch_n_houses) # 37.5 GW target, 46.17% on class1 buildings, scaled by share of ALLDSO EGIDs in all CH EGIDs

            n_municipalities    = topo_gwr_summary_df['bfs'].nunique()
            n_houses            = preprep_gwr_all_buildilngs.loc[preprep_gwr_all_buildilngs['BFS_NO'].isin(LRG_bfs_list),'EGID'].nunique()
            n_egids             = topo_gwr_summary_df.shape[0]
            n_grid_nodes        = topo_gwr_summary_df['grid_node'].nunique()
            pvcapacity_mw_2024  = topo_gwr_summary_df.loc[topo_gwr_summary_df['info_source'] == 'pv_df', 'TotalPower'].sum() / 1000
            pvcapacity_mw_2050    = constrcapa.loc[constrcapa['year'] == 2050, 'constr_capacity_kw'].sum() / 1000
            


            

            # FORMAT LATEX TABLE
            def fmt(n):
                return f"{n:,}".replace(",", "'")
            replacements = {
                "n_municipalities":         fmt(n_municipalities),
                "n_houses":                 fmt(n_houses),
                "n_egids":                  fmt(n_egids),
                "n_grid_nodes":             fmt(n_grid_nodes),
                "pvcapacity_mw_2024":       fmt(round(pvcapacity_mw_2024, 1)),
                "pvcapacity_mw_2050":       fmt(round(pvcapacity_mw_2050, 1)),

                "alldso_n_municipalities":    fmt(len(ALLDSO_bfs_list)),
                "alldso_n_houses":            fmt(alldso_n_houses),
                "alldso_n_egids":             fmt(alldso_n_egids),
                "alldso_n_grid_nodes":        fmt(alldso_n_grid_nodes),
                "alldso_pvcapacity_mw_2024": fmt(round(alldso_pvcapacity_mw_2024, 1)),
                "alldso_pvcapacity_mw_2050": fmt(round(alldso_pvcapacity_mw_2050, 1)),

                "allch_n_municipalities":    fmt(int(allch_n_municipalities)),
                "allch_n_houses":            fmt(allch_n_houses),
                "allch_n_egids":             fmt(allch_n_egids),
                # "allch_n_grid_nodes":        fmt(allch_n_grid_nodes),
                "allch_pvcapacity_mw_2024": fmt(round(allch_pvcapacity_mw_2024, 1)),
                "allch_pvcapacity_mw_2050": fmt(round(allch_pvcapacity_mw_2050, 1)),
                }

            self._write_latex_from_template(
                template_file="latex_table_template__datasummary.txt",
                export_file="summary_stats.txt",
                replacements=replacements,
            )

    def plot_gridnode_HOY(self, 
                          scen      = 'pvalloc_29nbfs_LRG2_max', 
                          gridnode  = None,
                          iter = 8, 
                          hours_incl_list=list(range(4920 + 3*24, 4920 + 6*24)),
                          daynightbands = True,
                          below_threshold_rgb = (31, 119, 180),
                          above_threshold_rgb = (214, 39, 40),
                          threshold_line_rgb = (0, 0, 0),
                          threshold_line_style = '--',

                          plot_width_func = None,
                          plot_height_func = None,
                          ):
        # import df if avialable
        gridnode_iter_path = os.path.join(
            self.data_path,
            'pvalloc',
            scen,
            'zMC1',
            'pred_gridprem_node_by_M',
            f'gridnode_df_{iter}.parquet',
        )
        if os.path.exists(gridnode_iter_path):
            gridnode_iter_df = pl.read_parquet(gridnode_iter_path)
        else:
            gridnode_iter_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))

        # select node with max loss by default
        if gridnode is None:
            gridnode_to_plot = (
                gridnode_iter_df
                .sort('feedin_atnode_loss_kW', descending=True)
                .get_column('grid_node')
                .item(0)
            )
            
        else: 
            gridnode_to_plot = gridnode    

        df_plot = (
            gridnode_iter_df
            .filter(
                (pl.col('grid_node') == gridnode_to_plot) &
                (pl.col('t_int').is_in(hours_incl_list))
            )
            .to_pandas()
            .sort_values('t_int')
        )
        if df_plot.empty:
            print(f'No data found for grid node {gridnode_to_plot}.')
            return

        # check for required columns
        threshold_col = 'kW_threshold'
        if threshold_col not in df_plot.columns:
            raise KeyError(f"Missing required column '{threshold_col}'")

        threshold_value = float(df_plot[threshold_col].iloc[0])
        values = df_plot['max_demand_feedin_atnode_kW'].to_numpy()
        x_values = df_plot['t_int'].to_numpy()

        def rgb_to_mpl_color(rgb_value):
            if not isinstance(rgb_value, (list, tuple, np.ndarray)) or len(rgb_value) != 3:
                raise ValueError('Expected an RGB tuple/list with exactly 3 values.')
            rgb_array = np.asarray(rgb_value, dtype=float)
            if np.nanmax(rgb_array) > 1.0:
                rgb_array = rgb_array / 255.0
            return tuple(rgb_array.tolist())

        def build_threshold_segments(x_vals, y_vals, threshold):
            below_segments = []
            above_segments = []

            for idx in range(len(x_vals) - 1):
                x0 = float(x_vals[idx])
                x1 = float(x_vals[idx + 1])
                y0 = float(y_vals[idx])
                y1 = float(y_vals[idx + 1])

                if y0 == threshold and y1 == threshold:
                    below_segments.append([(x0, y0), (x1, y1)])
                    continue

                if (y0 <= threshold and y1 <= threshold) or (y0 >= threshold and y1 >= threshold):
                    target_segments = below_segments if y0 <= threshold and y1 <= threshold else above_segments
                    target_segments.append([(x0, y0), (x1, y1)])
                    continue

                if y1 != y0:
                    x_cross = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
                else:
                    x_cross = x0
                crossing_point = (x_cross, threshold)

                if y0 < threshold < y1:
                    below_segments.append([(x0, y0), crossing_point])
                    above_segments.append([crossing_point, (x1, y1)])
                else:
                    above_segments.append([(x0, y0), crossing_point])
                    below_segments.append([crossing_point, (x1, y1)])

            return below_segments, above_segments

        below_segments, above_segments = build_threshold_segments(x_values, values, threshold_value)

        # PLOT
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        plt.figure(figsize=(plot_width, plot_height))

        ax = plt.gca()
        if below_segments:
            below_collection = LineCollection(below_segments, colors=[rgb_to_mpl_color(below_threshold_rgb)], linewidths=1.8, label='below threshold')
            ax.add_collection(below_collection)
        if above_segments:
            above_collection = LineCollection(above_segments, colors=[rgb_to_mpl_color(above_threshold_rgb)], linewidths=1.8, label='above threshold')
            ax.add_collection(above_collection)
        ax.axhline(threshold_value, color=rgb_to_mpl_color(threshold_line_rgb), linestyle=threshold_line_style, linewidth=1.0, label=f'kW_threshold = {threshold_value:.0f}')

        ax.set_xlim(float(x_values.min()) - 0.5, float(x_values.max()) + 0.5)
        y_min = float(np.nanmin(values))
        y_max = float(np.nanmax(values))
        y_pad = max(10.0, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
        ax.set_ylim(min(y_min, threshold_value) - y_pad, max(y_max, threshold_value) + y_pad)

        # add day/night bands if specified
        if daynightbands is True:
            daynightbands = {'day_start_hour': 7, 'day_end_hour': 19}

        if isinstance(daynightbands, dict):
            day_start_hour = int(daynightbands.get('day_start_hour', 7))
            day_end_hour = int(daynightbands.get('day_end_hour', 19))
            day_color = daynightbands.get('day_color', '#fff7cc')
            night_color = daynightbands.get('night_color', '#e6f0ff')
            band_alpha = float(daynightbands.get('alpha', 0.25))

            t_min = int(df_plot['t_int'].min())
            t_max = int(df_plot['t_int'].max())

            def is_day_hour(t_val):
                hour_of_day = ((int(t_val) - 1) % 24) + 1
                return day_start_hour <= hour_of_day < day_end_hour

            segment_start = t_min
            prev_is_day = is_day_hour(t_min)

            for t_val in range(t_min + 1, t_max + 1):
                curr_is_day = is_day_hour(t_val)
                if curr_is_day != prev_is_day:
                    segment_color = day_color if prev_is_day else night_color
                    ax.axvspan(segment_start - 0.5, t_val - 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)
                    segment_start = t_val
                    prev_is_day = curr_is_day

            segment_color = day_color if prev_is_day else night_color
            ax.axvspan(segment_start - 0.5, t_max + 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)

            for line in ax.lines:
                line.set_zorder(3)

        plt.xlabel('t (hours of year)')
        plt.ylabel('Feed-in at node (kW)')
        plt.title(f'Grid node {gridnode_to_plot}')
        plt.tight_layout()
        # plt.show()
        self._save_figure(os.path.join(self.dir_path_export, 'gridnode_feedin_threshold_HOY.png'), plot_width, plot_height)
        plt.close()

    def plot_constrcapa_comparison(self,
                                   scen = 'pvalloc_LRG3_max',
                                   capa_col_list = [
                                    #    'constr_capacity_kw_HIST',
                                    'constr_capacity_kw_EP2050_rescale1.0',
                                    'constr_capacity_kw_EP2050_rescale0.25', 
                                    'constr_capacity_kw_AdjHist_refact0.2',
                                   ],
                                   year_range=(2016, 2024),
                                   year_col='date',
                                   pv_time_col='BeginningOfOperation',
                                   pv_capacity_col='TotalPower',
                                   capa_col_label_map=None,
                                   pv_label='Installed capacity from pv_df',
                                   export_name = 'constrcapa_comparison.png',
                                   title='Installed Capacity Comparison',
                                   x_label='Year',
                                   y_label='Installed capacity (MW)',
                                   plot_width=None,
                                   plot_height=None,
                                   pv_line_color=(31, 119, 180),
                                   pv_line_style='-',
                                   pv_line_width=2.0,
                                   constr_line_width=1.8,
                                   legend_title=None,
                                   ):
        constrcapa_comparison = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'constrcapa_comparison.parquet'))
        pv_df = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'pv_df.parquet'))

        def rgb_to_mpl_color(rgb_value):
            if not isinstance(rgb_value, (list, tuple, np.ndarray)) or len(rgb_value) != 3:
                raise ValueError('Expected an RGB tuple/list with exactly 3 values.')
            rgb_array = np.asarray(rgb_value, dtype=float)
            if np.nanmax(rgb_array) > 1.0:
                rgb_array = rgb_array / 255.0
            return tuple(rgb_array.tolist())

        def format_capacity_label(col_name):
            label = col_name.replace('constr_capacity_kw_', '')
            label = label.replace('EP2050_', 'EP2050 ')
            label = label.replace('AdjHist_', 'Adjusted historical ')
            label = label.replace('rescale', 'rescale ')
            label = label.replace('_', ' ')
            return label.strip()

        def resolve_year_series(df, year_column, preferred_time_column=None):
            if preferred_time_column is not None and preferred_time_column in df.columns:
                parsed = pd.to_datetime(df[preferred_time_column], errors='coerce')
                if parsed.notna().any():
                    return parsed.dt.year
            if year_column in df.columns:
                if year_column == 'date':
                    parsed = pd.to_datetime(df[year_column], errors='coerce')
                    if parsed.notna().any():
                        return parsed.dt.year
                return pd.to_numeric(df[year_column], errors='coerce')
            if 'BeginningOfOperation' in df.columns:
                parsed = pd.to_datetime(df['BeginningOfOperation'], errors='coerce')
                return parsed.dt.year
            if 'BeginOp' in df.columns:
                parsed = pd.to_datetime(df['BeginOp'], errors='coerce')
                return parsed.dt.year
            if df.index.name == year_column:
                return pd.to_numeric(df.index, errors='coerce')
            raise KeyError(f"Could not resolve a year column for '{year_column}' or a date column such as 'BeginningOfOperation'.")

        if year_range is not None and len(year_range) == 2:
            year_start = int(year_range[0])
            year_end = int(year_range[1])
        else:
            year_start = None
            year_end = None

        # Prepare PV installed capacity time series from pv_df.
        if pv_capacity_col not in pv_df.columns:
            raise KeyError(f"Missing required column '{pv_capacity_col}' in pv_df.")

        pv_years = resolve_year_series(pv_df, year_col, preferred_time_column=pv_time_col)
        pv_series_df = pv_df.loc[pv_years.notna(), [pv_capacity_col]].copy()
        pv_series_df['year'] = pv_years[pv_years.notna()].astype(int).to_numpy()
        if year_start is not None and year_end is not None:
            pv_series_df = pv_series_df.loc[pv_series_df['year'].between(year_start, year_end)]

        pv_yearly = (
            pv_series_df
            .groupby('year', as_index=False)[pv_capacity_col]
            .sum()
            .sort_values('year')
        )
        pv_yearly['installed_capacity_mw'] = pv_yearly[pv_capacity_col] / 1000.0
        pv_yearly['installed_capacity_mw'] = pv_yearly['installed_capacity_mw'].cumsum()

        # Prepare comparison dataframe.
        constr_years = resolve_year_series(constrcapa_comparison, year_col)
        constr_plot_df = constrcapa_comparison.loc[constr_years.notna()].copy()
        constr_plot_df['year'] = constr_years[constr_years.notna()].astype(int).to_numpy()
        if year_start is not None and year_end is not None:
            constr_plot_df = constr_plot_df.loc[constr_plot_df['year'].between(year_start, year_end)]

        if capa_col_label_map is None:
            capa_col_label_map = {}

        plot_width = self.plot_width if plot_width is None else plot_width
        plot_height = self.plot_height if plot_height is None else plot_height

        plt.figure(figsize=(plot_width, plot_height))
        ax = plt.gca()

        ax.plot(
            pv_yearly['year'],
            pv_yearly['installed_capacity_mw'],
            color=rgb_to_mpl_color(pv_line_color),
            linestyle=pv_line_style,
            linewidth=pv_line_width,
            marker='o',
            label=pv_label,
            zorder=3,
        )

        for col_name in capa_col_list:
            if col_name not in constr_plot_df.columns:
                continue
            label = capa_col_label_map.get(col_name, format_capacity_label(col_name))
            plot_series = constr_plot_df.groupby('year', as_index=False)[col_name].last().sort_values('year')
            ax.plot(
                plot_series['year'],
                plot_series[col_name] / 1000.0,
                linewidth=constr_line_width,
                marker='o',
                label=label,
                zorder=2,
            )

        if year_start is not None and year_end is not None:
            ax.set_xlim(year_start, year_end)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(title=legend_title)
        plt.tight_layout()

        if export_name is None:
            export_file = f'{scen}_constrcapa_comparison.png'
        else:
            export_file = export_name if os.path.splitext(export_name)[1] else f'{export_name}.png'
        self._save_figure(os.path.join(self.dir_path_export, export_file), plot_width, plot_height)
        plt.close()

    def worstnode_worstweek(self,
                            scen = 'pvalloc_LRG3_max',
                            title = 'Excess Feedin ("Worst Week")', 
                            export_name = 'worstnode_worstweek',
                            gridnode = None,
                            excess_feedin_pegid = True,
                            x_label = 'Hour of year',
                            y_label = 'Excess Feed-in (kW)',
                            y_scaling = 1.0,
                            rgb_line = (200, 50, 50),
                            legend_loc = 'upper left',
                            plot_width_func = None,
                            plot_height_func = None,):
        
        topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
        gridnode_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))

        topo_rows = []
        for k, v in topo.items():
            topo_rows.append({
                'EGID': k,
                'grid_node': v['grid_node'],
            })
        topo_df = pl.DataFrame(topo_rows)

        if gridnode is None:
            worst_node = (
                gridnode_df
                .group_by('grid_node')
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
                .sort('total_loss_kW', descending=True)
                .get_column('grid_node')
                .item(0)
            )
        else:
            worst_node = gridnode

        worst_node_df = (
            gridnode_df
            .filter(pl.col('grid_node') == worst_node)
            .sort('t_int')
            .to_pandas()
            .reset_index(drop=True)
        )

        worst_node_df['loss_7d_kW'] = worst_node_df['feedin_atnode_loss_kW'].rolling(168).sum()
        worst_idx = worst_node_df['loss_7d_kW'].idxmax()

        worst_start = worst_node_df.loc[worst_idx, 't_int'] - 167
        worst_end   = worst_node_df.loc[worst_idx, 't_int']

        negid_worstnode = topo_df.filter(pl.col('grid_node') == worst_node).get_column('EGID').count()
        worst_week_df = worst_node_df.loc[worst_node_df['t_int'].between(worst_start, worst_end)].copy()
        if negid_worstnode <= 0:
            raise ValueError(f'No EGIDs mapped to grid node {worst_node}.')
        
        if excess_feedin_pegid is True:
            worst_week_df['feedin_atnode_loss_kW'] = (
                worst_week_df['feedin_atnode_loss_kW'] / negid_worstnode * y_scaling
            )

        # print(f'Worst week: t_int\t\t{worst_start} to {worst_end}')
        # print(f'Date start:\t\t\t{worst_start_date}')
        # print(f'Date end:\t\t\t{worst_end_date}')
        # print(f'total loss:\t\t\t{worst_loss:.1f} kWh')
        # print(f'Average loss p House:\t\t{worst_loss / negid_worstnode} kWh')
        # find the actual worst day (max daily loss) within the worst week
        _ww = worst_node_df.loc[worst_node_df['t_int'].between(worst_start, worst_end)].copy()
        _ww['day'] = (_ww['t_int'] - 1) // 24 + 1
        _worst_day = _ww.groupby('day')['feedin_atnode_loss_kW'].sum().idxmax()
        worst_peakweek1_start = (_worst_day - 1) * 24 + 1
        worst_peakweek1_end   = _worst_day * 24
        
        def _fmt_ch(val, d=1):
            return f'{val:,.{d}f}'.replace(',', "'")

        def _node_peak_stats(node, node_df_pl, topo_df_pl):
            ndf = (
                node_df_pl
                .filter(pl.col('grid_node') == node)
                .sort('t_int')
                .to_pandas()
                .reset_index(drop=True)
            )
            ndf['loss_7d_kW'] = ndf['feedin_atnode_loss_kW'].rolling(168).sum()
            idx = ndf['loss_7d_kW'].idxmax()
            w_start = ndf.loc[idx, 't_int'] - 167
            w_end   = ndf.loc[idx, 't_int']
            # find the day with maximum total loss within the worst week
            ww = ndf.loc[ndf['t_int'].between(w_start, w_end)].copy()
            ww['day'] = (ww['t_int'] - 1) // 24 + 1
            worst_day = ww.groupby('day')['feedin_atnode_loss_kW'].sum().idxmax()
            t_peak_start = (worst_day - 1) * 24 + 1
            t_peak_end   = worst_day * 24
            peak_date = pd.Timestamp('2025-01-01') + pd.to_timedelta(worst_day - 1, unit='D')
            peak_df = ndf.loc[ndf['t_int'].between(t_peak_start, t_peak_end)].copy()
            n_egid = topo_df_pl.filter(pl.col('grid_node') == node).get_column('EGID').count()
            return {
                'peak_loss':      peak_df['feedin_atnode_loss_kW'].sum(),
                'peak_netdemand': peak_df['netdemand_kW'].sum(),
                'n_egid':         n_egid,
                'peak_day_str':   peak_date.strftime('%d.%m.'),
            }

        top5_nodes = (
            gridnode_df
            .group_by('grid_node')
            .agg(pl.col('feedin_atnode_loss_kW').sum().alias('total_loss_kW'))
            .sort('total_loss_kW', descending=True)
            .head(5)
            .get_column('grid_node')
            .to_list()
        )

        replacements = {
            'figure_filename':      f'{export_name}_{scen}.png',
            'hist_figure_filename': f'hist_avgloss_pEGID_{scen}.png',
            'node1_color_rgb':      f'{rgb_line[0]},{rgb_line[1]},{rgb_line[2]}',
        }
        for i, node in enumerate(top5_nodes, start=1):
            s = _node_peak_stats(node, gridnode_df, topo_df)
            replacements[f'node_{i}']                = node
            replacements[f'peak_day_{i}']            = s['peak_day_str']
            replacements[f'total_excess_feedin_{i}'] = _fmt_ch(s['peak_loss'])
            replacements[f'n_houses_{i}']            = s['n_egid']
            replacements[f'avg_feedin_p_house_{i}']  = _fmt_ch(s['peak_loss'] / s['n_egid'])
            replacements[f'avg_demand_p_house_{i}']  = _fmt_ch(s['peak_netdemand'] / s['n_egid'])

        # also keep the original single-node keys (node 1 = worst node) for backwards compat
        replacements['worst_node'] = top5_nodes[0]

        self._write_latex_from_template(
            template_file='latex_table_template__worstnode_worstweek.txt',
            export_file='worstweek_node_peak.txt',
            replacements=replacements,
        )
        
        
        scen_color = (rgb_line[0] / 255, rgb_line[1] / 255, rgb_line[2] / 255)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        plt.figure(figsize=(plot_width, plot_height))
        sns.lineplot(
            data=worst_week_df,
            x='t_int',
            y='feedin_atnode_loss_kW',
            marker='',
            color=scen_color,
            linewidth=1.5,
            alpha=self.line_opacity,
            label=f'grid node {worst_node}',
        )

        # highlight the actual worst day with a red band behind the line
        ax = plt.gca()
        try:
            ax.axvspan(worst_peakweek1_start - 0.5, worst_peakweek1_end + 0.5, color='red', alpha=0.15, zorder=0)
        except Exception:
            pass
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(title=None, loc=legend_loc)
        plt.tight_layout()
        self._save_figure(
            os.path.join(self.dir_path_export, f'{export_name}_{scen}.png'),
            plot_width,
            plot_height,
        )
        # plt.show()
        plt.close()


    def hist_avgloss_pEGID(self,
                            scen = 'pvalloc_LRG3_max',
                            title = 'Average Excess Feed-in per House',
                            export_name = 'hist_avgloss_pEGID',
                            x_label = 'Excess Feed-in per House in 24h (kWh)',
                            y_label = 'Frequency (n Houses)',
                            hist_rgb =  (200, 50, 50),
                            x_example_tick = None,
                            x_example_str = '',
                            plot_width_func = None,
                            plot_height_func = None,
                            ):

        topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
        gridnode_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))

        topo_rows = []
        for k, v in topo.items():
            topo_rows.append({
                'EGID': k,
                'grid_node': v['grid_node'],
            })
        topo_df = pl.DataFrame(topo_rows)

        # worst day per node
        gridnode_day_df = (
            gridnode_df
            .with_columns(((pl.col('t_int') - 1) // 24 + 1).alias('day'))
            .sort(['grid_node', 't_int'], )
            )
        worst_per_node_day_df = (
            gridnode_df
            .with_columns(
                ((pl.col('t_int') - 1) // 24 + 1).alias('day')
            )
            .group_by(['grid_node', 'day'])
            .agg(pl.col('feedin_atnode_loss_kW').sum().alias('loss_day_kW'))
            .sort(['grid_node', 'loss_day_kW'], descending=[False, True])
            .group_by('grid_node', maintain_order=True)
            .first()
            .sort('loss_day_kW', descending=True)
        )
        worst_node751 = gridnode_day_df.filter(
            (pl.col('grid_node') == '751') & 
            (pl.col('day') == 118) 
            # (pl.col('t_int') >= 2809 + 10)
            )
        worst_node751['feedin_atnode_loss_kW'].sum() # 2809 + 10 = 2819, 2819 - 1 = 2818, 2818 / 24 = 117.4167, day = 118
        
        # Histogramm, avg. excess feedin p. house
        negid_node_df = topo_df.group_by('grid_node').agg(pl.col('EGID').count().alias('nEGID'))
        worst_day_df = worst_per_node_day_df.join(negid_node_df, on='grid_node', how='left')
        worst_day_df = worst_day_df.with_columns(
            (pl.col('loss_day_kW') / pl.col('nEGID')).alias('avg_loss_per_house_kW')
        )
        hist_excfeedin_df = worst_day_df.filter(
            pl.col('avg_loss_per_house_kW') > 0
        )

        scen_color = (hist_rgb[0] / 255, hist_rgb[1] / 255, hist_rgb[2] / 255)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func       
        plt.figure(figsize=(plot_width, plot_height))

        sns.histplot(
            data=hist_excfeedin_df.to_pandas(),
            x='avg_loss_per_house_kW',
            bins=15,
            color = scen_color,
            alpha=0.6,
        )   
        # sns.kdeplot(
        #     data=hist_excfeedin_df.to_pandas(),
        #     x='avg_loss_per_house_kW',
        #     color = scen_color,
        #     alpha=0.2,
        # )   
        if x_example_tick is not None:
            ax = plt.gca()
            ax.axvline(x=x_example_tick, color='black', linewidth=1.2, linestyle='--', zorder=5, ymax=0.80)
            ax.text(
                x_example_tick, 0.82, f'{x_example_str}{x_example_tick}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom',
                fontsize=8, color='black',
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        # plt.show()
        self._save_figure(os.path.join(self.dir_path_export, f'{export_name}_{scen}.png'), plot_width, plot_height)
        plt.close()


        plt.figure(figsize=(plot_width, plot_height))
        # sns.kdeplot(
        #     data=hist_excfeedin_df.to_pandas(),
        #     x='nEGID',
        #     color = scen_color,
        #     alpha=0.6,
        #     fill=True,
        # )
        sns.histplot(
            data=hist_excfeedin_df.to_pandas(),
            x='nEGID',
            color = scen_color,
            bins=15,
            alpha=0.6,
        )
        plt.xlabel('Number of Houses per Grid Node')
        plt.ylabel('Density')
        plt.title('Distribution of Houses per Grid Node')
        plt.tight_layout()
        # plt.show()
        self._save_figure(os.path.join(self.dir_path_export, f'distribution_houses_per_gridnode_{scen}.png'), plot_width, plot_height)
        plt.close()


    def TS_loss_pEGID(self,
                    scen = 'pvalloc_LRG3_max',
                    freq = 'daily',           # 'hourly', 'daily', 'weekly', or 'monthly'
                    title = 'Excess Feed-in Loss per House over Weather Year',
                    export_name = 'TS_loss_pEGID',
                    x_label = None,
                    y_label = 'Excess Feed-in Loss per House (kWh)',
                    iter = None, 
                    line_rgb = (200, 50, 50),
                    line_alpha = 0.3,
                    plot_width_func = None,
                    plot_height_func = None,
                    ):

        if freq not in ('hourly', 'daily', 'weekly', 'monthly'):
            raise ValueError(f"freq must be 'hourly', 'daily', 'weekly', or 'monthly', got '{freq}'")

        topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
        if iter is None:
            gridnode_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'gridnode_df.parquet'))
        else:
            gridnode_df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'pred_gridprem_node_by_M', f'gridnode_df_{iter}.parquet'))

        topo_rows = []
        for k, v in topo.items():
            topo_rows.append({'EGID': k, 'grid_node': v['grid_node']})
        topo_df = pl.DataFrame(topo_rows)
        negid_node_df = topo_df.group_by('grid_node').agg(pl.col('EGID').count().alias('nEGID'))

        # aggregate to chosen frequency
        if freq == 'hourly':
            period_col = 't_int'
            agg_df = (
                gridnode_df
                .join(negid_node_df, on='grid_node', how='left')
                .with_columns(
                    (pl.col('feedin_atnode_loss_kW') / pl.col('nEGID')).alias('loss_pEGID_kWh')
                )
                .select(['grid_node', 't_int', 'loss_pEGID_kWh'])
                .sort(['grid_node', 't_int'])
            )
        elif freq == 'daily':
            period_col = 'day'
            agg_df = (
                gridnode_df
                .with_columns(((pl.col('t_int') - 1) // 24 + 1).alias('day'))
                .group_by(['grid_node', 'day'])
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('loss_node_kWh'))
                .join(negid_node_df, on='grid_node', how='left')
                .with_columns(
                    (pl.col('loss_node_kWh') / pl.col('nEGID')).alias('loss_pEGID_kWh')
                )
                .select(['grid_node', 'day', 'loss_pEGID_kWh'])
                .sort(['grid_node', 'day'])
            )
        elif freq == 'weekly':
            period_col = 'week'
            agg_df = (
                gridnode_df
                .with_columns(((pl.col('t_int') - 1) // 168 + 1).alias('week'))
                .group_by(['grid_node', 'week'])
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('loss_node_kWh'))
                .join(negid_node_df, on='grid_node', how='left')
                .with_columns(
                    (pl.col('loss_node_kWh') / pl.col('nEGID')).alias('loss_pEGID_kWh')
                )
                .select(['grid_node', 'week', 'loss_pEGID_kWh'])
                .sort(['grid_node', 'week'])
            )
        else:  # monthly
            period_col = 'month'
            _month_end_h = [744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
            month_map = pl.DataFrame({
                't_int': list(range(1, 8761)),
                'month': [
                    next(m + 1 for m, end in enumerate(_month_end_h) if h <= end)
                    for h in range(1, 8761)
                ],
            })
            agg_df = (
                gridnode_df
                .join(month_map, on='t_int', how='left')
                .group_by(['grid_node', 'month'])
                .agg(pl.col('feedin_atnode_loss_kW').sum().alias('loss_node_kWh'))
                .join(negid_node_df, on='grid_node', how='left')
                .with_columns(
                    (pl.col('loss_node_kWh') / pl.col('nEGID')).alias('loss_pEGID_kWh')
                )
                .select(['grid_node', 'month', 'loss_pEGID_kWh'])
                .sort(['grid_node', 'month'])
            )

        x_label_default = {
            'hourly': 'Hour of Year',
            'daily': 'Day of Year',
            'weekly': 'Week of Year',
            'monthly': 'Month of Year',
        }
        x_label = x_label if x_label is not None else x_label_default[freq]

        plot_df = agg_df.to_pandas()
        line_color = (line_rgb[0] / 255, line_rgb[1] / 255, line_rgb[2] / 255)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        plt.figure(figsize=(plot_width, plot_height))
        for node, node_df in plot_df.groupby('grid_node'):
            plt.plot(
                node_df[period_col],
                node_df['loss_pEGID_kWh'],
                color=line_color,
                linewidth=0.6,
                alpha=line_alpha,
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        # plt.show()
        self._save_figure(
            os.path.join(self.dir_path_export, f'{export_name}_{freq}_{scen}.png'),
            plot_width,
            plot_height,
        )
        plt.close()




    def Loss_Subscost_Summary(self,
                              scen_list = None,
                              PVprod_csv_file           = 'plot_agg_line_PVproduction___export_plot_data___21scen.csv',
                              contcharact_csv_file = 'plot_agg_hist_contcharact_newinst___export_plot_data___18scen.csv',
                              n_iter_list = None):
        PVprod_file_path      = os.path.join(self.dir_path, PVprod_csv_file)
        contcaracht_file_path = os.path.join(self.dir_path, contcharact_csv_file)

        self._copy_csv_to_export(PVprod_file_path)
        self._copy_csv_to_export(contcaracht_file_path)
        PVprod_df = pd.read_csv(PVprod_file_path)
        contcharact_df = pd.read_csv(contcaracht_file_path)

        if scen_list is None:
            scen_list = [
                scen for scen in PVprod_df['scen'].dropna().unique().tolist()
                if scen in contcharact_df['scen'].dropna().unique().tolist()
            ]

        if n_iter_list is None:
            n_iter_list = sorted(
                set(PVprod_df['n_iter'].dropna().unique().tolist())
                .intersection(contcharact_df['n_iter'].dropna().unique().tolist())
            )

        scen_list = [scen for scen in scen_list if scen in PVprod_df['scen'].unique() and scen in contcharact_df['scen'].unique()]
        n_iter_list = list(n_iter_list)

        def format_value(value, decimals=0):
            if pd.isna(value):
                return '-'
            if decimals > 0:
                formatted = f"{value:,.{decimals}f}"
            else:
                formatted = f"{value:,.0f}"
            return formatted.replace(',', "'")

        def build_pivot_table(df, value_cols, aggfunc='sum'):
            grouped = (
                df.loc[
                    df['scen'].isin(scen_list) & df['n_iter'].isin(n_iter_list),
                    ['scen', 'n_iter'] + value_cols,
                ]
                .groupby(['scen', 'n_iter'], as_index=False)
                .agg(aggfunc)
            )
            table = grouped.pivot(index='scen', columns='n_iter', values=value_cols[0])
            table = table.reindex(index=scen_list, columns=n_iter_list)
            table.index.name = 'Scenario'
            table.columns.name = 'Iteration'
            return table

        loss_table = build_pivot_table(PVprod_df, ['feedin_atnode_loss_kW'])
        # Scale losses from kW to MWh (divide by 1000)
        loss_table = loss_table / 1000

        cont_summary = (
            contcharact_df.loc[
                contcharact_df['scen'].isin(scen_list) & contcharact_df['iter_round'].isin(n_iter_list),
                ['scen', 'iter_round', 'subs_nodeHC_chf', 'pena_nodeHC_chf', 'estim_pvinstcost_chf'],
            ]
            .groupby(['scen', 'iter_round'], as_index=False)
            .sum()
        )
        # Net subsidy = subsidy received - penalties - 30% of investment costs
        cont_summary['net_subsidy_chf'] = cont_summary['subs_nodeHC_chf'] - cont_summary['pena_nodeHC_chf'] - (0.3 * cont_summary['estim_pvinstcost_chf'])

        net_subsidy_table = (
            cont_summary[['scen', 'iter_round', 'net_subsidy_chf']]
            .pivot(index='scen', columns='iter_round', values='net_subsidy_chf')
            .reindex(index=scen_list, columns=n_iter_list)
        )
        net_subsidy_table.index.name = 'Scenario'
        net_subsidy_table.columns.name = 'Iteration'

        loss_table_latex = loss_table.apply(lambda col: col.map(lambda value: format_value(value, decimals=1))).to_latex(
            escape=True,
            index=True,
            na_rep='-',
        )
        net_subsidy_table_latex = net_subsidy_table.apply(lambda col: col.map(lambda value: format_value(value, decimals=0) + ' CHF' if not pd.isna(value) else '-')).to_latex(
            escape=True,
            index=True,
            na_rep='-',
        )

        replacements = {
            'loss_table': loss_table_latex,
            'net_subsidy_table': net_subsidy_table_latex,
        }

        self._write_latex_from_template(
            template_file='latex_table_template__loss_subscost_summary.txt',
            export_file='loss_subscost_summary.txt',
            replacements=replacements,
        )

        print('Loss and net subsidy summary written.')


    # AGG CSV - based plots ======================================================================
    def plot_productionHOY_per_node(self, 
                                    csv_file, 
                                    scen_incl_list,
                                    hours_incl_list,
                                    export_name, 
                                    
                                    plot_width_func = None,
                                    plot_height_func = None, 
                                    ):

        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        np.random.seed(42) 
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        # n_random_colors = [max(0, len(scen_not_default))]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        plt.figure(figsize=(plot_width, plot_height))

        for i, scen in enumerate(scen_incl_list):
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['t_int'].isin(hours_incl_list)),
                :
            ].copy()
            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x='t_int',
                    y='feedin_atnode_loss_kW',
                    label=self._get_scenario_label(scen),
                    alpha=self.line_opacity,
                )

        plt.xlabel('t (hours of year)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title('Aggregated Feed-in Loss (hourly)')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        self._save_figure(os.path.join(self.dir_path_export, f'{export_name}.png'), plot_width, plot_height)


    def plot_productionHOY_per_node_byiter(self, 
                                           csv_file,
                                           scen_incl_list,
                                           hours_incl_list,
                                           iter_incl_list,
                                           export_name,
                                           daynightbands = None,
                                           plot_width_func = None,
                                           plot_height_func = None,
                                             ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        band_settings = None
        if daynightbands is True:
            band_settings = {}
        elif isinstance(daynightbands, dict):
            band_settings = daynightbands.copy()

        np.random.seed(42) 
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        # n_random_colors = [max(0, len(scen_not_default))]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        plt.figure(figsize=(plot_width, plot_height))

        for i, scen in enumerate(scen_incl_list):
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['t_int'].isin(hours_incl_list)) &
                (df['iter'].isin(iter_incl_list)),
                :
            ].copy()


            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x='t_int',
                    y='feedin_atnode_loss_kW',
                    hue='iter',
                    palette=sns.color_palette("viridis", n_colors=len(iter_incl_list)),
                    linewidth=0.9,
                    alpha=self.line_opacity,
                    estimator=None
                    )

        if band_settings is not None:
            day_start_hour = int(band_settings.get('day_start_hour', 7))
            day_end_hour = int(band_settings.get('day_end_hour', 19))
            day_color = band_settings.get('day_color', '#fff7cc')
            night_color = band_settings.get('night_color', '#e6f0ff')
            band_alpha = float(band_settings.get('alpha', 0.25))

            if day_start_hour < day_end_hour:
                df_bands = df.loc[
                    (df['scen'].isin(scen_incl_list)) &
                    (df['t_int'].isin(hours_incl_list)) &
                    (df['iter'].isin(iter_incl_list)),
                    :
                ]

                if not df_bands.empty:
                    t_min = int(df_bands['t_int'].min())
                    t_max = int(df_bands['t_int'].max())

                    def is_day_hour(t_val):
                        hour_of_day = ((int(t_val) - 1) % 24) + 1
                        return day_start_hour <= hour_of_day < day_end_hour

                    ax = plt.gca()
                    segment_start = t_min
                    prev_is_day = is_day_hour(t_min)

                    for t_val in range(t_min + 1, t_max + 1):
                        curr_is_day = is_day_hour(t_val)
                        if curr_is_day != prev_is_day:
                            segment_color = day_color if prev_is_day else night_color
                            ax.axvspan(segment_start - 0.5, t_val - 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)
                            segment_start = t_val
                            prev_is_day = curr_is_day

                    segment_color = day_color if prev_is_day else night_color
                    ax.axvspan(segment_start - 0.5, t_max + 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)

                    for line in ax.lines:
                        line.set_zorder(3)

        plt.xlabel('t (hours)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title('Hourly Feed-in Loss by Iteration')
        plt.legend(title='Iteration')
        plt.tight_layout()
        # plt.show()
        self._save_figure(os.path.join(self.dir_path_export, f'{export_name}.png'), plot_width, plot_height)    

        
    def plot_PVproduction_line(self,
                               csv_file,
                               scen_incl_list,
                               n_iter_range_list,
                               export_name,
                               y_col,
                               title,
                               y_label,
                               y_scaling = 1.0,
                               start_year = 2024,
                               plot_width_func = None,
                               plot_height_func = None,
                                   ):
        if isinstance(csv_file, list) and len(csv_file) > 1:
            df_list = []
            for file in csv_file:
                temp_df = pd.read_csv(os.path.join(self.dir_path, file))
                df_list.append(temp_df)
            df = pd.concat(df_list, ignore_index=True)

        elif isinstance(csv_file, list) and len(csv_file) == 1:
            file_path = csv_file[0]
            self._copy_csv_to_export(file_path)
            df = pd.read_csv(file_path)

        else:
            file_path = os.path.join(self.dir_path, csv_file)
            self._copy_csv_to_export(file_path)
            df = pd.read_csv(file_path)
        df.loc[df['n_iter'] == 5 & df['scen'].isin(scen_incl_list), 
               ['scen', 'n_iter', 'feedin_atnode_loss_kW', 'demand_atnode_kW', 'feedin_atnode_kW',  y_col]]

        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        np.random.seed(42)
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        plt.figure(figsize=(plot_width, plot_height))

        for i, scen in enumerate(scen_incl_list):
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['n_iter'].isin(n_iter_range_list)),
                :
            ].copy()
            scen_color = (plot_color_map[scen][0] / 255, plot_color_map[scen][1] / 255, plot_color_map[scen][2] / 255)

            # unit conversion (kWh to MWh)
            df_plot[y_col] = df_plot[y_col] / y_scaling

            # map n_iter to calendar year
            x_col = 'n_iter'
            if start_year is not None:
                df_plot['year'] = start_year + df_plot['n_iter'] - 1
                x_col = 'year'

            # get scenario label for legend
            scen_label = self._get_scenario_label(scen)

            # get dash style
            scen_linedash, scen_marker = self._get_scenario_linedash_marker(scen)

            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x=x_col,
                    y=y_col,
                    color=scen_color,
                    label=scen_label,
                    marker=scen_marker,
                    linestyle=scen_linedash,
                    dashes=[(2,2)],
                    linewidth=1.5,
                    alpha=self.line_opacity,
                )

        x_label = 'Year' if start_year is not None else 'Model Iterations (Future Years)'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{title}')
        # if y_col == 'feedin_atnode_taken_kW':
        #     plt.ylabel(f'Aggregated {y_label} (GWh)')
        # elif y_col == 'feedin_atnode_loss_kW':
        #     plt.ylabel(f'Aggregated {y_label} (MWh)')
        # elif y_col == 'TotalPower':
        #     plt.ylabel(f'Aggregated {y_label} (MW)')
        # else:
        #     plt.title(f'Agg. {y_label}')

        plt.legend()
        plt.tight_layout()
        # plt.show()
        self._save_figure(os.path.join(self.dir_path_export, f'{export_name}.png'), plot_width, plot_height)


    def plot_ind_hist_contcharact_newinst(self,
                                          csv_file,
                                          scen_incl_list,
                                          iter_incl_list,
                                          x_col_incl_list,
                                          export_name,
                                          plot_width_func = None,
                                          plot_height_func = None,
                                          ):
        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)
        
        # Adjust plot width based on number of subplots
        n_cols = len(x_col_incl_list)
        plot_width = (self.plot_width * n_cols) if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        np.random.seed(42) 
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        # Create subplots side by side
        fig, axes = plt.subplots(1, n_cols, figsize=(plot_width, plot_height))
        
        # If only one column, axes is not an array, so convert it
        if n_cols == 1:
            axes = [axes]

        # Loop through scenarios (assuming one scenario for now)
        for scen in scen_incl_list:
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['iter_round'].isin(iter_incl_list)),
                :
            ].copy()

            if not df_plot.empty:
                # Loop through each column to create side-by-side histograms
                for col_idx, x_col in enumerate(x_col_incl_list):
                    sns.histplot(
                        data=df_plot,
                        x=x_col,
                        hue= 'iter_round',
                        multiple='layer',
                        bins=30,
                        alpha=0.6,
                        palette=sns.color_palette("viridis", n_colors=len(iter_incl_list)),
                        ax=axes[col_idx],
                        legend=True,
                        linewidth=0.1
                    )
                    axes[col_idx].set_xlabel(f'{x_col} (m2)')
                    axes[col_idx].set_ylabel('Count')
                    # axes[col_idx].set_title(f'Histogram of {x_col} by Iteration')
                    # Update legend title
                    legend = axes[col_idx].get_legend()
                    if legend:
                        legend.set_title('Iteration')
        
            plt.tight_layout()
            # plt.show()
            export_file = f'{export_name}_{scen}.png'
            self._save_figure(os.path.join(self.dir_path_export, export_file), plot_width, plot_height)
            plt.close()
        

    def plot_ind_line_catgcharact_newinst(self, 
                                          csv_file,
                                          scen_incl_list,
                                        #   iter_incl_list,
                                          x_col_incl_dict,
                                          export_name,
                                          iter_incl_list=[1, 2, 3, 4, 5, 6, 7,  ],
                                          crop_right_x = None,
                                          plot_width_func = None,
                                          plot_height_func = None,
                                          ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        # Loop through each scenario
        for scen in scen_incl_list:
            # Loop through each column type (e.g., 'GKLAS', 'are_typ')
            for col_name, category_groups in x_col_incl_dict.items():
                
                # Filter data for this scenario and column
                df_scen_col = df.loc[
                    (df['scen'] == scen) & 
                    (df['col'] == col_name) &
                    (df['iter'].isin(iter_incl_list)),
                    :
                ].copy()
                
                if df_scen_col.empty:
                    continue
                
                # Create new dataframe for aggregated data
                plot_data = []
                
                # Get all categories included in the dict for this column
                all_included_categories = []
                for label, categories_list in category_groups.items():
                    all_included_categories.extend(categories_list)
                
                # For each iteration, aggregate the data
                for iter_val in df_scen_col['iter'].unique():
                    df_iter = df_scen_col[df_scen_col['iter'] == iter_val]
                    total_count = df_iter['count'].sum()

                    # Aggregate each defined group
                    for label, categories_list in category_groups.items():
                        count_sum = df_iter[df_iter['category'].isin(categories_list)]['count'].sum()
                        plot_data.append({
                            'iter': iter_val,
                            'group': label,
                            'share': count_sum / total_count if total_count > 0 else 0
                        })

                    # Calculate "rest" for categories not in the dict
                    rest_count = df_iter[~df_iter['category'].isin(all_included_categories)]['count'].sum()
                    if rest_count > 0:
                        plot_data.append({
                            'iter': iter_val,
                            'group': 'rest',
                            'share': rest_count / total_count if total_count > 0 else 0
                        })

                # Convert to DataFrame
                df_plot = pd.DataFrame(plot_data)

                if df_plot.empty:
                    continue

                # Create the plot
                plt.figure(figsize=(plot_width, plot_height))

                # Plot each group as a line
                for group in df_plot['group'].unique():
                    df_group = df_plot[df_plot['group'] == group]
                    sns.lineplot(
                        data=df_group,
                        x='iter',
                        y='share',
                        label=group,
                        marker='o',
                        linewidth=1.5
                    )

                plt.xlabel('Iteration')
                plt.ylabel('Share')
                plt.title(f'{col_name}')
                plt.legend(title = None)
                
                # Set x-axis to show only integers
                ax = plt.gca()
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                if crop_right_x is not None:
                    ax.set_xlim(right=crop_right_x)
                
                plt.tight_layout()
                # plt.show() 
                # Save with unique name for each scenario-column combination
                export_file = f'{export_name}_{scen}_{col_name}.png'
                self._save_figure(os.path.join(self.dir_path_export, export_file), plot_width, plot_height)
                plt.close()


    def plot_ind_hist_contcharact_allscen(self,
                                      csv_file,
                                      scen_incl_list,
                                      iter_incl_list,
                                      x_col_incl_list,
                                      export_name,
                                      plot_hist_opacity = 0.6,
                                      plot_width_func=None,
                                      plot_height_func=None):
        # Load data
        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)

        # Filter for all selected scenarios and iterations
        df_plot = df[
            (df['scen'].isin(scen_incl_list)) &
            (df['iter_round'].isin(iter_incl_list))
        ].copy()

        if df_plot.empty:
            print("No data matches the selected scenarios and iterations.")
            return

        # Adjust plot width based on number of subplots
        n_cols = len(x_col_incl_list)
        plot_width = (self.plot_width * n_cols) if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        # Generate colors for scenarios
        np.random.seed(42)
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        # Create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(plot_width, plot_height))
        if n_cols == 1:
            axes = [axes]

        # Loop over columns (variables) for histograms
        for col_idx, x_col in enumerate(x_col_incl_list):
            ax = axes[col_idx]

            # Create a new column combining scenario and iteration for hue
            df_plot['scen_str_short'] = df_plot['scen'].apply(self._get_scenario_label)
            df_plot['scen_iter'] = df_plot['scen_str_short'] + '_iter' + df_plot['iter_round'].astype(str)

            # Generate a color palette for each unique scenario+iteration
            unique_scen_iter = df_plot['scen_iter'].unique()
            palette = sns.color_palette("tab10", n_colors=len(unique_scen_iter))

            sns.histplot(
                data=df_plot,
                x=x_col,
                hue='scen_iter',
                multiple='layer',
                bins=30,
                alpha=plot_hist_opacity,
                palette=palette,
                ax=ax,
                linewidth=0.1,
                legend=True
            )
            ax.set_xlabel(f'{x_col}')
            ax.set_ylabel('Count')
            ax.set_title(f'Histogram of {x_col} by Scenario and Iteration')
            # Adjust legend
            legend = ax.get_legend()
            if legend:
                legend.set_title('Scenario / Iteration')

        plt.tight_layout()
        self._save_figure(os.path.join(self.dir_path_export, f'{export_name}.png'), plot_width, plot_height)
        plt.close()
    
    



    # NOT WORKING PROPERLY YET =========================================
    def plot_productionHOY_iters_hue(self, 
                                    csv_file,
                                    scen_incl_list,
                                    hours_incl_list,
                                    iter_incl_list,
                                    export_name,
                                    plot_width_func=None,
                                    plot_height_func=None):
        
        file_path = os.path.join(self.dir_path, csv_file)
        self._copy_csv_to_export(file_path)
        df = pd.read_csv(file_path)

        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        # Filter for all scenarios and hours
        df_plot = df[
            (df['scen'].isin(scen_incl_list)) &
            (df['t_int'].isin(hours_incl_list)) &
            (df['iter'].isin(iter_incl_list))
        ]

        if df_plot.empty:
            print("No data matches the selected scenarios, hours, and iterations.")
            return

        df_plot = df_plot.copy()
        df_plot['scen_label'] = df_plot['scen'].apply(self._get_scenario_label)

        plt.figure(figsize=(plot_width, plot_height))

        # Map colors from scen_default_color_map
        color_map = {
            self._get_scenario_label(scen): tuple(np.array(self.scen_default_color_map[scen])/255)
            for scen in scen_incl_list
        }

        sns.lineplot(
            data=df_plot,
            x='t_int',
            y='feedin_atnode_loss_kW',
            hue='scen_label',   # color = scenario
            palette=color_map,
            linewidth=1.5,
            alpha=self.line_opacity,
            estimator=None
        )

        plt.xlabel('t (hours)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title(f'Hourly Feed-in Loss - Iteration {iter_incl_list[0]}')
        plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        self._save_figure(os.path.join(self.dir_path_export, f'{export_name}.png'), plot_width, plot_height)
        plt.close()

    
if __name__ == "__main__":

    plotter = static_plotter_class()
    plotter.scen_default_color_map = {
        'pvalloc_29nbfs_LRG2_max':          (200, 50, 50),      # red (keep)
    
        'pvalloc_29nbfs_LRG2_max_epzb1':                (230, 140, 140),  # soft pastel red
        'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1':    (60, 180, 60),                # bright green           
        'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3':    (150, 220, 150),               # warm light orange (instead of yellow)

        'pvalloc_29nbfs_LRG2_max_sAs6p0':   (60, 120, 200),  # strong blue
        'pvalloc_29nbfs_LRG2_max_sBs0p8':   (60, 160, 90),   # green (less neon)
        'pvalloc_29nbfs_LRG2_max_sCs2p8':   (200, 140, 40),  # orange (instead of yellow)

        'pvalloc_29nbfs_LRG2_max_1hll':     (100, 180, 180),  # turquoise (instead of yellow)
        'pvalloc_29nbfs_LRG2_max_pksh':      (0, 80, 80),  # dark teal (instead of cyan)
        
        'pvalloc_29nbfs_LRG2_gridoptim_max':    (140, 60, 180),    # bright purple
        # 'pvalloc_29nbfs_LRG2_gridoptim_max':    (160, 120, 180),    # lavendar
        # 'pvalloc_29nbfs_LRG2_gridoptim_max':    (180, 140, 190),    # lilac
    }
    plotter.simple_scen_name_mapping = {
            'pvalloc_29nbfs_LRG2_max': 'default scenario',
            'scenario2': 'Scenario 2',
            'scenario3': 'Scenario 3',
            'scenario4': 'Scenario 4',
        }


    # SCEN - individual tables + data ======================================================================
    # plotter.copy_standalone_graphs_to_presentation_dir()

    # SCEN - individual plots  ======================================================================

    # plotter.plot_EGID_pvprod_demand_HOY(plot_cols_incl_list= ['pvprod_kW'])
    # plotter.plot_EGID_pvprod_demand_HOY(plot_cols_incl_list= ['pvprod_kW', 'demand_kW'])
    # plotter.plot_EGID_pvprod_demand_HOY(plot_cols_incl_list= ['pvprod_kW', 'demand_kW', 'netfeedin_kW'])

    # plotter.NPVhist()

    # plotter.plot_gridnode_HOY()

    # plotter.worstnode_worstweek()
        
    plotter.hist_avgloss_pEGID()

    plotter.TS_loss_pEGID(
        freq = 'daily',
        x_label = 'Hour of Year',
    )
    plotter.TS_loss_pEGID(
        freq = 'weekly',
        x_label = 'Week of Year',
    )


    # plotter.plot_constrcapa_comparison()



    print('\n*********************\n******** end ********\n*********************\n\n')








