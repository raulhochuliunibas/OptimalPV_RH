# python integrated packages
import sys
import os as os
import glob
import json

# external packages
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndVarSummaryStats(VisualizationCore):
    """Plot of the pvalloc_initalization + sanitycheck stage."""

    def plot_ind_var_summary_stats(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_var_summary_stats', self.visual_sett.log_name)

        # "settings" ----------
        n_cols_catg = 3 if len(self.visual_sett.pvalloc_scen_list) >= 3 else len(self.visual_sett.pvalloc_scen_list) 
        n_cols_cont = 1

        catg_col_plot = [
            'GKLAS',
            'GSTAT',
            'GBAUJ_cat',
            'are_typ',
            'sfhmfh_typ',
        ]
        cont_col_plot = [
            'GAREA', 
            'FLAECHE', 
            'AUSRICHTUNG_mean',
        ]

        # create Plots and Color Dict ----------------------------------------
        n_rows_catg = int(np.ceil(len(self.visual_sett.pvalloc_scen_list) / n_cols_catg))
        fig_smry_catg_bar = make_subplots(rows=n_rows_catg, cols=n_cols_catg,
                                                subplot_titles=[f'{scen}' for scen in self.visual_sett.pvalloc_scen_list])
        n_rows_cont = int(np.ceil(len(self.visual_sett.pvalloc_scen_list) / n_cols_cont))
        fig_smry_cont_hist = make_subplots(rows=n_rows_cont, cols=n_cols_cont,
                                            subplot_titles=[f'{scen}' for scen in self.visual_sett.pvalloc_scen_list])


        # color dict ----------
        global_color_rows = []
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # import
            topo = json.load(open(f'{self.visual_sett.data_path}/pvalloc/{scen}/sanity_check_byEGID/topo_egid.json', 'r'))

            # compute topo_df
            for k,v in topo.items():

                partitions = v.get('solkat_partitions', {})
                gwr_info = v.get('gwr_info', {})
                pv_inst = v.get('pv_inst', {})

                for k_p, v_p in partitions.items():
                    row = {
                        'EGID': k,
                        'df_uid': k_p,
                        'bfs': gwr_info.get('bfs'),
                        'GKLAS': gwr_info.get('gklas'),
                        'GSTAT': gwr_info.get('gstat'),
                        'GAREA': gwr_info.get('garea'),
                        'GBAUJ': gwr_info.get('gbauj'),
                        'are_typ': gwr_info.get('are_typ'),
                        'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                        'demand_arch_typ': v.get('demand_arch_typ'),
                        'demand_elec_dem_pGAREA': v.get('demand_elec_dem_pGAREA'),
                        'grid_node': v.get('grid_node'),

                        'inst_TF': pv_inst.get('inst_TF'),
                        'info_source': pv_inst.get('info_source'),
                        'pvid': pv_inst.get('xtf_id'),
                        'pvtarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
                        'TotalPower': pv_inst.get('TotalPower'),
                        'df_uid_w_inst': v.get('df_uid_w_inst'),

                        'FLAECHE': v_p.get('FLAECHE'),
                        'AUSRICHTUNG': v_p.get('AUSRICHTUNG'),
                        'STROMERTRAG': v_p.get('STROMERTRAG'),
                        'NEIGUNG': v_p.get('NEIGUNG'),
                        'MSTRAHLUNG': v_p.get('MSTRAHLUNG'),
                        # 'elecpri_Rp_kWh': v.get('elecpri_Rp_kWh'),
                    }
                    global_color_rows.append(row)

        topo_df_catg_raw = pl.DataFrame(global_color_rows)
        topo_df_cont_raw = pl.DataFrame(global_color_rows)


        # transform data to relevant columns ----------
        # GBAUJ into bins
        if True: 
            topo_df_catg_raw = topo_df_catg_raw.with_columns(
                pl.col("GBAUJ")
                .cast(pl.Int32, strict=False)  # converts invalid values (e.g. "") to null
                .fill_null(0)
                .alias("GBAUJ_int")
            )
            bins = np.linspace(1900, 2030, 14)  # 13 bins => 14 edges
            labels = [f"{int(bins[i])}-{int(bins[i+1]-1)}" for i in range(len(bins)-1)]

            # Initialize category column
            topo_df_catg_raw = topo_df_catg_raw.with_columns(
                pl.when(pl.col("GBAUJ") == "")
                .then(pl.lit("missing"))
                .otherwise(pl.lit("unknown"))
                .alias("GBAUJ_cat")
            )

            # Assign bins for non-missing values
            for i in range(len(bins) - 1):
                lower = int(bins[i])
                upper = int(bins[i+1])

                label = f"{lower}-{upper-1}"

                topo_df_catg_raw = topo_df_catg_raw.with_columns(
                    pl.when(
                        (pl.col("GBAUJ") != "") &  # skip empty values
                        (pl.col("GBAUJ_int") >= lower) &
                        (pl.col("GBAUJ_int") < upper)
                    )
                    .then(pl.lit(label))
                    .otherwise(pl.col("GBAUJ_cat"))
                    .alias("GBAUJ_cat")
                )
            topo_df_catg_raw[['GBAUJ', 'GBAUJ_int', 'GBAUJ_cat' ]]


        # make color mapping CATG ----------
        topo_df_catg = topo_df_catg_raw.group_by('EGID').agg([
            pl.col('GKLAS').first().alias('GKLAS'),
            pl.col('GSTAT').first().alias('GSTAT'),
            pl.col('GBAUJ_cat').first().alias('GBAUJ_cat'),
            pl.col('are_typ').first().alias('are_typ'),
            pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
        ])

        all_categories = (
            topo_df_catg
            .select(catg_col_plot)          # select only categorical columns
            .melt()                         # turn wide → long
            .select("value")                # keep only values
            .unique()                       # unique category values
            .to_series()                    # convert to a simple Series
            .to_list()                      # list of category strings
        )
        color_map_catg = {
            cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, cat in enumerate(all_categories)
        }


        # make color mapping CONT ----------
        topo_df_cont = topo_df_cont_raw.group_by('EGID').agg([
            pl.col('GAREA').sum().alias('GAREA'),
            pl.col('FLAECHE').sum().alias('FLAECHE'),
            pl.col('AUSRICHTUNG').mean().alias('AUSRICHTUNG_mean'),
        ])


        color_map_cont = {
            col: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, col in enumerate(cont_col_plot)
        }



        # Plots ----------------------------------------
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # import
            topo                     = json.load(open(f'{self.visual_sett.data_path}/pvalloc/{scen}/sanity_check_byEGID/topo_egid.json', 'r'))
            # compute topo_df
            rows = []

            for k,v in topo.items():

                partitions = v.get('solkat_partitions', {})
                gwr_info = v.get('gwr_info', {})
                pv_inst = v.get('pv_inst', {})

                for k_p, v_p in partitions.items():
                    row = {
                        'EGID': k,
                        'df_uid': k_p,
                        'bfs': gwr_info.get('bfs'),
                        'GKLAS': gwr_info.get('gklas'),
                        'GSTAT': gwr_info.get('gstat'),
                        'GAREA': gwr_info.get('garea'),
                        'GBAUJ': gwr_info.get('gbauj'),
                        'are_typ': gwr_info.get('are_typ'),
                        'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                        'demand_arch_typ': v.get('demand_arch_typ'),
                        'demand_elec_dem_pGAREA': v.get('demand_elec_dem_pGAREA'),
                        'grid_node': v.get('grid_node'),

                        'inst_TF': pv_inst.get('inst_TF'),
                        'info_source': pv_inst.get('info_source'),
                        'pvid': pv_inst.get('xtf_id'),
                        'pvtarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
                        'TotalPower': pv_inst.get('TotalPower'),
                        'df_uid_w_inst': v.get('df_uid_w_inst'),

                        'FLAECHE': v_p.get('FLAECHE'),
                        'AUSRICHTUNG': v_p.get('AUSRICHTUNG'),
                        'STROMERTRAG': v_p.get('STROMERTRAG'),
                        'NEIGUNG': v_p.get('NEIGUNG'),
                        'MSTRAHLUNG': v_p.get('MSTRAHLUNG'),
                        'elecpri_Rp_kWh': v.get('elecpri_Rp_kWh'),
                    }
                    rows.append(row)

            topo_df_catg_raw = pl.DataFrame(rows)
            topo_df_cont_raw = pl.DataFrame(rows)


            # summary BAR plot ----------------------------------------
            if True: 
                # transform data to relevant columns ----------
                # GBAUJ into bins
                if True: 
                    topo_df_catg_raw = topo_df_catg_raw.with_columns(
                        pl.col("GBAUJ")
                        .cast(pl.Int32, strict=False)  # converts invalid values (e.g. "") to null
                        .fill_null(0)
                        .alias("GBAUJ_int")
                    )
                    bins = np.linspace(1900, 2030, 14)  # 13 bins => 14 edges
                    labels = [f"{int(bins[i])}-{int(bins[i+1]-1)}" for i in range(len(bins)-1)]

                    # Initialize category column
                    topo_df_catg_raw = topo_df_catg_raw.with_columns(
                        pl.when(pl.col("GBAUJ") == "")
                        .then(pl.lit("missing"))
                        .otherwise(pl.lit("unknown"))
                        .alias("GBAUJ_cat")
                    )

                    # Assign bins for non-missing values
                    for i in range(len(bins) - 1):
                        lower = int(bins[i])
                        upper = int(bins[i+1])

                        label = f"{lower}-{upper-1}"

                        topo_df_catg_raw = topo_df_catg_raw.with_columns(
                            pl.when(
                                (pl.col("GBAUJ") != "") &  # skip empty values
                                (pl.col("GBAUJ_int") >= lower) &
                                (pl.col("GBAUJ_int") < upper)
                            )
                            .then(pl.lit(label))
                            .otherwise(pl.col("GBAUJ_cat"))
                            .alias("GBAUJ_cat")
                        )
                    topo_df_catg_raw[['GBAUJ', 'GBAUJ_int', 'GBAUJ_cat' ]]

                topo_df_catg = topo_df_catg_raw.group_by('EGID').agg([
                    pl.col('GKLAS').first().alias('GKLAS'),
                    pl.col('GSTAT').first().alias('GSTAT'),
                    pl.col('GBAUJ_cat').first().alias('GBAUJ_cat'),
                    pl.col('are_typ').first().alias('are_typ'),
                    pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                ])


                # add traces ----------
                for col in catg_col_plot:
                    # value counts as dict: {category: count}
                    vc = topo_df_catg[col].value_counts().to_dict(as_series=True)
                    categories = vc[col]
                    counts = vc["count"]
                    scen_col_legend = f"{scen} – {col}"

                    for idx, (cat, count) in enumerate(zip(categories, counts)):
                        show_legend_for_trace = True if idx == 0 else False
                        fig_smry_catg_bar.add_trace(
                            go.Bar(
                                x=[col],
                                y=[count],
                                name = scen_col_legend,
                                marker=dict(color=color_map_catg[cat]),   
                                legendgroup=scen_col_legend,
                                showlegend=show_legend_for_trace,
                                # name=cat,                     # category shown in legend
                                # legendgroup=cat,              # same category across columns/scenarios
                                base=None,                    # required for stacking
                                text=[f"{cat}: {count}"],
                                hovertemplate=(
                                    f"Scenario: {scen}<br>"
                                    f"Column: {col}<br>"
                                    f"Category: {cat}<br>"
                                    f"Share: {count / sum(counts) * 100:.2f}%"
                                    "Count: %{y}<extra></extra>"
                                )
                            ),
                            row = (i_scen // n_cols_catg) + 1, 
                            col = (i_scen % n_cols_catg) + 1,
                        )    



            # summary HIST plot ----------------------------------------
            if True:
                # transform data to relevant columns ----------
                # ...
                topo_df_cont = topo_df_cont_raw.group_by('EGID').agg([
                    pl.col('GAREA').sum().alias('GAREA'),
                    pl.col('FLAECHE').sum().alias('FLAECHE'),
                    pl.col('AUSRICHTUNG').mean().alias('AUSRICHTUNG_mean'),
                ])


                # add traces ----------
                show_first_trace = True if i_scen == 0 else False
                for col in cont_col_plot:
                    data_series = topo_df_cont[col]
                    scen_col_legend = f"{scen} – {col}"

                    fig_smry_cont_hist.add_trace(
                        go.Histogram(
                            x=data_series.to_list(),
                            name=scen_col_legend,
                            marker=dict(color=color_map_cont[col]),
                            # legendgroup=scen,
                            showlegend=True,
                            nbinsx=30,
                            opacity=0.6,
                            texttemplate="%{y}",
                            hovertemplate=(
                                f"Scenario: {scen}<br>"
                                f"Column: {col}<br>"
                                "Value: %{x}<br>"
                                "Count: %{y}<extra></extra>"
                            )
                        ),
                        row = (i_scen // n_cols_cont) + 1, 
                        col = (i_scen % n_cols_cont) + 1,
                    )


        # update + export ----------------------------------------
        # bar plot
        fig_smry_catg_bar.update_layout(
            title = 'Summary EGID Charact. of Sample per Scen (categorical)',
            barmode='stack',
            template='plotly_white',
            )

        if plot_show:
            fig_smry_catg_bar.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_ind_summary_catg_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_ind_summary_catg_var___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_ind_summary_catg_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                        f'{self.visual_sett.visual_path}/plot_ind_summary_catg_var___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')   
            fig_smry_catg_bar.write_html(f'{self.visual_sett.visual_path}/plot_ind_summary_catg_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_ind_summary_catg_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name) 


        # histogramm
        fig_smry_cont_hist.update_layout(
            title = 'Summary EGID Charact. of Sample per Scen (coontinuous)',
            barmode='overlay',
            template='plotly_white',
            )

        if plot_show:
            fig_smry_cont_hist.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_ind_summary_cont_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_ind_summary_cont_var___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_ind_summary_cont_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                        f'{self.visual_sett.visual_path}/plot_ind_summary_cont_var___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
            fig_smry_cont_hist.write_html(f'{self.visual_sett.visual_path}/plot_ind_summary_cont_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_ind_summary_cont_var___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)
