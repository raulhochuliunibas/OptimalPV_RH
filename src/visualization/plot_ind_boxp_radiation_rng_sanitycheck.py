# python integrated packages
import sys
import os as os
import glob
import json
from itertools import chain

# external packages
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skewnorm

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndBoxpRadiationRngSanitycheck(VisualizationCore):
    """Plot of the pvalloc_initalization + sanitycheck stage."""

    def plot_ind_boxp_radiation_rng_sanitycheck(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile(f'plot_ind_boxp_radiation_rng_sanitycheck', self.visual_sett.log_name)

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)


            kWpeak_per_m2, share_roof_area_available = self.pvalloc_scen.TECspec_kWpeak_per_m2, self.pvalloc_scen.TECspec_share_roof_area_available
            inverter_efficiency = self.pvalloc_scen.TECspec_inverter_efficiency
            panel_efficiency_print = 'dynamic' if self.pvalloc_scen.PEFspec_variable_panel_efficiency_TF else 'static'

            # data import
            sanity_scen_data_path = f'{self.visual_sett.data_path}/pvalloc/{scen}/sanity_check_byEGID'
            pv = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/pv.parquet')
            topo = json.load(open(f'{sanity_scen_data_path}/topo_egid.json', 'r'))
            gwr_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')

            egid_with_pvdf = [egid for egid in topo.keys() if topo[egid]['pv_inst']['info_source'] == 'pv_df']
            xtf_in_topo = [topo[egid]['pv_inst']['xtf_id'] for egid in egid_with_pvdf]
            topo_subdf_paths = glob.glob(f'{sanity_scen_data_path}/topo_subdf_*.parquet')
            topo.get(egid_with_pvdf[0])


            # functions per subdf
            def distr_comp(df, colname):
                    srs = df[colname]
                    return(srs.mean(), srs.std(), srs.quantile(0.5), srs.quantile(0.25), srs.quantile(0.75), srs.min(), srs.max())

            def generate_distribution(mean, std, skew, kurt, min_val, max_val, num_points=1000):
                x = np.linspace(min_val, max_val, num_points)
                # Generate a normal distribution
                # y = norm.pdf(x, mean, std)
                # Adjust for skewness
                y = skewnorm.pdf(x, skew, mean, std)
                return x, y

            def generate_kde(df_srs, num_points=8000):
                kde = gaussian_kde(df_srs)
                x = np.linspace(min(df_srs), max(df_srs), num_points)
                y = kde(x)
                return kde
                # return x, y

            agg_by_method = "subdf"
            export_neg_rad_egid_counter, export_lrgthn1_rad_rel_locmax_counter = 0, 0

            unit_id_list = []
            debug_subdf_rad_dir_list, debug_subdf_rad_diff_list, debug_subdf_radiation_list, debug_subdf_radiation_rel_locmax_list = [], [], [], []
            all_rad_dir_list, all_rad_diff_list, all_radiation_list, all_radiation_rel_locmax_list = [], [], [], []

            i_path, path = 0, topo_subdf_paths[0]
            for i_path, path in enumerate(topo_subdf_paths):
                print(f'\tsubdf {i_path+1}/{len(topo_subdf_paths)}')    if i_path < 4 else None
                subdf= pd.read_parquet(path)

                if agg_by_method == "subdf":    # if agg debug subdf by subdf file "number"
                    unit_id_list.append(i_path)

                    debug_subdf_rad_dir_list.append(distr_comp(subdf, 'rad_direct'))
                    debug_subdf_rad_diff_list.append(distr_comp(subdf, 'rad_diffuse'))
                    debug_subdf_radiation_list.append(distr_comp(subdf, 'radiation'))
                    debug_subdf_radiation_rel_locmax_list.append(distr_comp(subdf, 'radiation_rel_locmax'))

                    all_rad_diff_list.append(subdf['rad_diffuse'])
                    all_rad_dir_list.append(subdf['rad_direct'])
                    all_radiation_list.append(subdf['radiation'])
                    all_radiation_rel_locmax_list.append(subdf['radiation_rel_locmax'])

                    # export subdf_by_egid if contains negative radiation:
                    subdf_neg_rad = subdf[subdf['radiation'] < 0]
                    if not subdf_neg_rad.empty:
                        while export_neg_rad_egid_counter < 2:
                            for egid in subdf_neg_rad['EGID'].unique():
                                egid_df = subdf[subdf['EGID'] == egid]
                                if egid_df['radiation'].min() < 0:
                                    export_neg_rad_egid_counter += 1
                                    if plot_export:
                                        egid_df.to_excel(f'{self.visual_sett.data_path}/output/{scen}/subdf_egid{egid}_neg_rad.xlsx')
                                    print(f'exported neg rad egid {export_neg_rad_egid_counter}')
                                if export_neg_rad_egid_counter == 2:
                                    break
                    # export subdf_by_egid if radiation_rel_locmax is > 1:
                    if False:
                        subdf_rad_rel_locmax = subdf[subdf['radiation_rel_locmax'] > 1]
                        if not subdf_rad_rel_locmax.empty:
                            while export_lrgthn1_rad_rel_locmax_counter <2:
                                for egid in subdf_rad_rel_locmax['EGID'].unique():
                                    egid_df = subdf[subdf['EGID'] == egid]
                                    if egid_df['radiation_rel_locmax'].max() > 1:
                                        export_lrgthn1_rad_rel_locmax_counter += 1
                                        if plot_export:
                                            egid_df.to_excel(f'{self.visual_sett.data_path}/output/{scen}/subdf_egid{egid}_lrgthn1_rad_rel_locmax.xlsx')
                                        print(f'exported lrgthn1 rad_rel_locmax egid {export_lrgthn1_rad_rel_locmax_counter}')
                                    if export_lrgthn1_rad_rel_locmax_counter == 2:
                                        break


                elif agg_by_method == "egid":
                        for egid in subdf['EGID'].unique():
                            egid_df = subdf[subdf['EGID'] == egid]

                            unit_id_list.append(egid)
                            debug_subdf_rad_dir_list.append(distr_comp(egid_df, 'rad_direct'))
                            debug_subdf_rad_diff_list.append(distr_comp(egid_df, 'rad_diffuse'))
                            debug_subdf_radiation_list.append(distr_comp(egid_df, 'radiation'))
                            debug_subdf_radiation_rel_locmax_list.append(distr_comp(egid_df, 'radiation_rel_locmax'))

            # aggregated on subdf level
            debug_rad_df = pd.DataFrame({'i_subdf_file': unit_id_list,
                                            'rad_direct_mean': [tupl_val[0] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_std':  [tupl_val[1] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_median': [tupl_val[2] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_1q': [tupl_val[3] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_3q': [tupl_val[4] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_min': [tupl_val[5] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_max': [tupl_val[6] for tupl_val in debug_subdf_rad_dir_list],

                                            'rad_diff_mean': [tupl_val[0] for tupl_val in debug_subdf_rad_diff_list], 
                                            'rad_diff_std':  [tupl_val[1] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_median': [tupl_val[2] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_1q': [tupl_val[3] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_3q': [tupl_val[4] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_min': [tupl_val[5] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_max': [tupl_val[6] for tupl_val in debug_subdf_rad_diff_list],

                                            'radiation_mean': [tupl_val[0] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_std':  [tupl_val[1] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_median': [tupl_val[2] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_1q': [tupl_val[3] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_3q': [tupl_val[4] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_min': [tupl_val[5] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_max': [tupl_val[6] for tupl_val in debug_subdf_radiation_list],

                                            'radiation_rel_locmax_mean': [tupl_val[0] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_std':  [tupl_val[1] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_median': [tupl_val[2] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_1q': [tupl_val[3] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_3q': [tupl_val[4] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_min': [tupl_val[5] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_max': [tupl_val[6] for tupl_val in debug_subdf_radiation_rel_locmax_list],                                            
                                            })

            # not aggregated, have all values in one list
            all_rad_dir = np.fromiter(chain.from_iterable(all_rad_dir_list), dtype=float)
            all_rad_diff = np.fromiter(chain.from_iterable(all_rad_diff_list), dtype=float)
            all_radiation = np.fromiter(chain.from_iterable(all_radiation_list), dtype=float)
            all_radiation_rel_locmax = np.fromiter(chain.from_iterable(all_radiation_rel_locmax_list), dtype=float)

            all_rad_df = pd.DataFrame({ 
                'all_rad_direct_mean':     all_rad_dir.mean(),
                'all_rad_direct_std':      all_rad_dir.std(),
                'all_rad_direct_median':   np.median(all_rad_dir),
                'all_rad_direct_1q':       np.quantile(all_rad_dir, 0.25),
                'all_rad_direct_3q':       np.quantile(all_rad_dir, 0.75),
                'all_rad_direct_min':      all_rad_dir.min(),
                'all_rad_direct_max':      all_rad_dir.max(),

                'all_rad_diff_mean':     all_rad_diff.mean(),
                'all_rad_diff_std':      all_rad_diff.std(),
                'all_rad_diff_median':   np.median(all_rad_diff),
                'all_rad_diff_1q':       np.quantile(all_rad_diff, 0.25),
                'all_rad_diff_3q':       np.quantile(all_rad_diff, 0.75),
                'all_rad_diff_min':      all_rad_diff.min(),
                'all_rad_diff_max':      all_rad_diff.max(),

                'all_radiation_mean':     all_radiation.mean(),
                'all_radiation_std':      all_radiation.std(),
                'all_radiation_median':   np.median(all_radiation),
                'all_radiation_1q':       np.quantile(all_radiation, 0.25),
                'all_radiation_3q':       np.quantile(all_radiation, 0.75),
                'all_radiation_min':      all_radiation.min(),
                'all_radiation_max':      all_radiation.max(),

                'all_radiation_rel_locmax_mean':     all_radiation_rel_locmax.mean(),   
                'all_radiation_rel_locmax_std':      all_radiation_rel_locmax.std(),
                'all_radiation_rel_locmax_median':   np.median(all_radiation_rel_locmax),
                'all_radiation_rel_locmax_1q':       np.quantile(all_radiation_rel_locmax, 0.25),
                'all_radiation_rel_locmax_3q':       np.quantile(all_radiation_rel_locmax, 0.75),
                'all_radiation_rel_locmax_min':      all_radiation_rel_locmax.min(),
                'all_radiation_rel_locmax_max':      all_radiation_rel_locmax.max(),
            }, index=[0])


            # PLOTS ---------------------------------
            # plot normal disttribution and kde -> problem: data is not normally distributed!
            if False:                            
                fig = go.Figure()
                # only_two_iter = ['rad_direct', 'rad_diff', ]
                for colname in ['rad_direct', 'rad_diff', 'radiation', 'radiation_rel_locmax']:
                    fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'', opacity=0))
                    fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'{colname}', opacity=0))

                    for index, row in debug_rad_df.iterrows():
                        x,y, = generate_distribution(row[f'{colname}_mean'], row[f'{colname}_std'], row[f'{colname}_skew'], row[f'{colname}_kurt'], row[f'{colname}_min'], row[f'{colname}_max'])
                        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{colname}_dist_{index}'))

                fig.update_layout(
                    title='Distribution Functions and KDE',
                    xaxis_title='Radiation [W/m2]',
                    yaxis_title='Frequency',
                    barmode='overlay'
                )
                fig.show()
                print('asdf')

            # plot boxplot -----
            fig_box = go.Figure()
            columns = ['rad_direct', 'rad_diff', 'radiation', 'radiation_rel_locmax']
            colors = ['blue', 'green', 'orange', 'purple']  # Colors for each colname

            for col_idx, colname in enumerate(columns):
                for idx, row in debug_rad_df.iterrows():
                    stats = [ row[f'{colname}_min'], row[f'{colname}_1q'], row[f'{colname}_median'], row[f'{colname}_3q'], row[f'{colname}_max'] ]

                    fig_box.add_trace(go.Box(
                        y=stats,
                        name=f"{colname} - {row['i_subdf_file']}",
                        legendgroup=colname,  # Group for the legend
                        marker_color=colors[col_idx],  # Use the same color for the group
                        boxpoints='all',  # Show individual points
                        jitter=0.3,       # Spread points for readability
                        pointpos=-1.8     # Offset points for better visualization
                    ))
                fig_box.update_layout(
                    title="Grouped Boxplots for rad_direct, rad_diff, radiation, and radiation_rel_locmax",
                    xaxis_title="Categories",
                    yaxis_title="Values",
                    boxmode="group",  # Group boxplots
                    template="plotly_white",
                    showlegend=True   # Enable legend
                )

            # Show the figure
            # fig_box.show()


            # plot ONE boxplot -----
            fig_onebox = go.Figure()
            columns = ['rad_direct', 'rad_diff', 'radiation', 'radiation_rel_locmax']
            colors = ['blue', 'green', 'orange', 'purple']

            for col_idx, colname in enumerate(columns):
                for idx, row in all_rad_df.iterrows():
                    stats = [ row[f'all_{colname}_min'], row[f'all_{colname}_1q'], row[f'all_{colname}_median'], row[f'all_{colname}_3q'], row[f'all_{colname}_max'] ]

                    fig_onebox.add_trace(go.Box(
                        y=stats,
                        name=f"{colname}",
                        legendgroup=colname,  # Group for the legend
                        marker_color=colors[col_idx],  # Use the same color for the group
                        boxpoints='all',  # Show individual points
                        jitter=0.3,       # Spread points for readability
                        pointpos=-1.8     # Offset points for better visualization
                    ))
            fig_onebox.update_layout(
                title="Boxplots for rad_direct, rad_diff, radiation, and radiation_rel_locmax",
                xaxis_title="Categories",
                yaxis_title="Radiation [W/m2]",
                boxmode="group",  # Group boxplots
                template="plotly_white",
                showlegend=True   # Enable legend
            )

            fig_onebox = self.add_scen_name_to_plot(fig_onebox, scen, self.pvalloc_scen)

            if plot_show:
                fig_onebox.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig_onebox.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___ind_boxp_radiation_rng_sanitycheck.html')
                else:
                    fig_onebox.write_html(f'{self.visual_sett.visual_path}/{scen}___ind_boxp_radiation_rng_sanitycheck.html')
                print_to_logfile(f'\texport: plot_ind_boxp_radiation_rng_sanitycheck.html (for: {scen})', self.visual_sett.log_name)
