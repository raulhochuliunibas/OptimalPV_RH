# python integrated packages
import sys
import os as os
import glob

# external packages
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndHistNPVFreepartitions(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_hist_NPV_freepartitions(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_hist_NPV_freepartitions', self.visual_sett.log_name)

        # init agg plots and color dict
        fig_agg = go.Figure()
        fig_agg_true_npv  = go.Figure()
        scen_filter_tags_list = []
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            # setup + import ----------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)
            exclude_filter_tags = self.visual_sett.plot_ind_hist_NPV_freepartitions_specs['exclude_filter_tags']
            nbins               = self.visual_sett.plot_ind_hist_NPV_freepartitions_specs['nbins']
            tag_subsidy_tuples  = self.visual_sett.plot_ind_hist_NPV_freepartitions_specs['tag_subsidy_tuples']

            # import ----------------
            npv_df = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_1.parquet')

            egid_filter_tags = [col for col in npv_df.columns if 'filter_tag' in col]
            dfuid_filter_tags = [col for col in npv_df.columns if 'filt_dfuid' in col]
            joint_tags = egid_filter_tags + dfuid_filter_tags

            filter_tags =  [col for col in joint_tags if col not in exclude_filter_tags]

            for tag in filter_tags:
                scen_filter_tags_list.append(f'{scen}_{tag}')

            scen_filter_tags_list.append(f'{scen}_all')

        scen_filter_tags_list = list(set(scen_filter_tags_list))
        color_map_scen_filter_tag_dict = {
            scen_filter_tag: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, scen_filter_tag in enumerate(scen_filter_tags_list)
        }


        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            # setup + import ----------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)
            exclude_filter_tags = self.visual_sett.plot_ind_hist_NPV_freepartitions_specs['exclude_filter_tags']
            nbins               = self.visual_sett.plot_ind_hist_NPV_freepartitions_specs['nbins']
            tag_subsidy_tuples  = self.visual_sett.plot_ind_hist_NPV_freepartitions_specs['tag_subsidy_tuples']

            # import ----------------
            npv_df = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_1.parquet')

            egid_filter_tags = [col for col in npv_df.columns if 'filter_tag' in col]
            dfuid_filter_tags = [col for col in npv_df.columns if 'filt_dfuid' in col]
            joint_tags = egid_filter_tags + dfuid_filter_tags

            filter_tags =  [col for col in joint_tags if col not in exclude_filter_tags]


            # plot ----------------
            fig = go.Figure()
            scen_filter_tag = f'{scen}_all'

            fig.add_trace(go.Histogram(x=npv_df['NPV_uid'], name=f'all EGID', opacity=0.5, nbinsx = nbins))
            kde_all = gaussian_kde(npv_df['NPV_uid'])
            x_range_all = np.linspace(npv_df['NPV_uid'].min(), npv_df['NPV_uid'].max(), 1000)
            fig.add_trace(go.Scatter(x=x_range_all, y=kde_all(x_range_all), mode='lines', name=f'kde all EGID', line=dict(color='black', width=2,), opacity=0.8, yaxis='y2'))

            fig_agg.add_trace(         go.Histogram(x=npv_df['NPV_uid'], name=f'all EGID scen: {scen}', opacity=0.5, nbinsx = nbins, marker_color = color_map_scen_filter_tag_dict[scen_filter_tag],))
            fig_agg.add_trace(         go.Scatter(x=x_range_all, y=kde_all(x_range_all), mode='lines', name=f'kde all EGID scen: {scen}', line=dict(color = color_map_scen_filter_tag_dict[scen_filter_tag], width=2,), opacity=0.8, yaxis='y2'))

            fig_agg_true_npv.add_trace(go.Histogram(x=npv_df['NPV_uid'], name=f'all EGID scen: {scen}', opacity=0.5, nbinsx = nbins))
            fig_agg_true_npv.add_trace(go.Scatter(x=x_range_all, y=kde_all(x_range_all), mode='lines', name=f'kde all EGID scen: {scen}',                                     opacity=0.8, yaxis='y2'))

            for tag in filter_tags:
                scen_filter_tag = f'{scen}_{tag}'
                df_filter = npv_df.loc[npv_df[tag] == True, ]
                fig.add_trace(go.Histogram(x=df_filter['NPV_uid'],      name=f'hist {tag:25}',              opacity=0.5, nbinsx = nbins))
                fig_agg.add_trace(go.Histogram(x=df_filter['NPV_uid'],  name=f'hist {tag:25}   -   {scen}', opacity=0.5, nbinsx = nbins, marker_color = color_map_scen_filter_tag_dict[scen_filter_tag],))

            for tag in filter_tags:
                scen_filter_tag = f'{scen}_{tag}'
                df_filter = npv_df.loc[npv_df[tag] == True, ]
                if len(df_filter) > 1:
                    kde = gaussian_kde(df_filter['NPV_uid'])
                    x_range = np.linspace(df_filter['NPV_uid'].min(), df_filter['NPV_uid'].max(), 1000)
                    fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name=f'kde  {tag:25}',                  opacity=0.8, yaxis='y2'))
                    fig_agg.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name=f'kde  {tag:25}   -   {scen}', line = dict(color = color_map_scen_filter_tag_dict[scen_filter_tag],), opacity=0.8,  yaxis='y2'))

                # add subsidized NPVs 
                subsidized_tags = [tupl[0] for tupl in tag_subsidy_tuples]
                if tag in subsidized_tags:
                    for tupl in tag_subsidy_tuples:
                        tag = tupl[0]
                        subsidy_amount = tupl[1]
                        if tag in filter_tags:
                            df_filter = npv_df.loc[npv_df[tag] == True, ]
                            npv_with_subsidy = df_filter['NPV_uid'] + subsidy_amount
                            kde_subsidy = gaussian_kde(npv_with_subsidy)
                            x_range_subsidy = np.linspace(npv_with_subsidy.min(), npv_with_subsidy.max(), 1000)
                            fig.add_trace(go.Scatter(x=x_range_subsidy, y=kde_subsidy(x_range_subsidy), mode='lines', 
                                                    name=f'kde  {tag:25} + subsidy {subsidy_amount}', opacity=0.8, yaxis='y2', line=dict(dash='dash')))
                            fig_agg.add_trace(go.Scatter(x=x_range_subsidy, y=kde_subsidy(x_range_subsidy), mode='lines', 
                                                        name=f'kde  {tag:25} + subsidy {subsidy_amount}   -   {scen}', opacity=0.8, yaxis='y2', line=dict(dash='dash')))

            fig.update_layout(
                xaxis_title=f'Net Present Value (NPV, interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr)',
                yaxis_title='Frequency',
                title = f'NPV Distribution of possible PV installations, first / last year (weather year: {self.pvalloc_scen.WEAspec_weather_year})',
                barmode = 'overlay',
                template='plotly_white',
                yaxis2=dict(
                    title='KDE Density',
                    overlaying='y',
                    side='right',
                    showgrid = False,
                ))

            fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)

            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_hist_NPV_freepartitions___{scen}.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_hist_NPV_freepartitions___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_hist_NPV_freepartitions.html (for: {scen})', self.visual_sett.log_name)


            # aggregate plot ----------------
            fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))


        # export AGGREGATION ----------------
        fig_agg.update_layout(
            xaxis_title=f'NPV, interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr, weather year: {self.pvalloc_scen.WEAspec_weather_year}',
            yaxis_title='Frequency',
            title = f'Comparison NPV Distribution of poss. PV installations ({len(self.visual_sett.pvalloc_scen_list)} scen)',
            barmode = 'overlay',
            template='plotly_white',
            yaxis2=dict(
                title='KDE Density',
                overlaying='y',
                side='right',
                showgrid = False,
            ))
        fig_agg_true_npv.update_layout(
            xaxis_title=f'NPV (no subsampling), interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr, weather year: {self.pvalloc_scen.WEAspec_weather_year}',
            yaxis_title='Frequency',
            title = f'Comparison NPV Distribution of poss. PV installations ({len(self.visual_sett.pvalloc_scen_list)} scen)',
            barmode = 'overlay',
            template='plotly_white',
            yaxis2=dict(
                title='KDE Density',
                overlaying='y',
                side='right',
                showgrid = False,
            ))

        if plot_show:
            fig_agg.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                          f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
            fig_agg.write_html(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen.html')

        if plot_show:
            fig_agg_true_npv.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_hist_trueNPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_hist_trueNPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_hist_trueNPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                          f'{self.visual_sett.visual_path}/plot_agg_hist_trueNPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
            fig_agg_true_npv.write_html(f'{self.visual_sett.visual_path}/plot_agg_hist_trueNPV_freepartitions___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
