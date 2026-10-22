# python integrated packages
import sys
import os as os
import glob
import json

# external packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.colors import qualitative
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndBarCatgcharactNewinst(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_bar_catgcharact_newinst(self, plot_show: bool = True, plot_export: bool = True):
        checkpoint_to_logfile('plot_ind_bar_catgcharact_newinst', self.visual_sett.log_name)
        color_map_catg = {}
        export_plot_data_list = []

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # setup + import ----------
            cols_catg_charact_inst  = self.visual_sett.plot_ind_XX_charact_newinst_specs['cols_catg_charact_inst']
            n_cols_subplot          = self.visual_sett.plot_ind_XX_charact_newinst_specs['n_cols_catg_charact']
            max_n_rows_catg_charact = self.visual_sett.plot_ind_XX_charact_newinst_specs['max_n_rows_catg_charact']
            cols_as_filter_tags     = self.visual_sett.plot_ind_XX_charact_newinst_specs['cols_as_filter_tags']

            topo         = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            pred_inst_df = pl.read_parquet( f'{self.visual_sett.mc_data_path}/pred_inst_df.parquet')
            npv_df       = pl.read_parquet( os.path.join(self.visual_sett.mc_data_path, 'pred_npv_inst_by_M', 'npv_df_1.parquet'))
            # npv_df       = pl.read_parquet( os.path.join(self.visual_sett.data_path, 'pvalloc', scen, 'zMC1', 'pred_npv_inst_by_M', 'npv_df_1.parquet'))

            pred_inst_df = pred_inst_df.with_columns([
                pl.col('iter_round').cast(pl.Int64).alias('iter_round'),
            ])

            # topo_df
            rows = []
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
                        'GWAERZH1': gwr_info.get('gwaerzh1'),
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
            # transform data to relevant columns ----------
            topo_df_catg_raw = topo_df_catg_raw.with_columns([
                pl.when(pl.col('GWAERZH1').is_in(['7410', '7411']))
                    .then(pl.lit('heatpump'))
                    .otherwise(pl.lit('no_heatpump'))
                    .alias('heatpump_TF'),
            ])

            # GBAUJ into bins
            if True: 
                topo_df_catg_raw = topo_df_catg_raw.with_columns(
                    pl.col("GBAUJ")
                    .cast(pl.Int32, strict=False)  # converts invalid values to null
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


            # Add filter_tags
            EGID_filter_tags_map = npv_df.select(['EGID',] + cols_as_filter_tags)
            topo_df_catg_raw = topo_df_catg_raw.join(EGID_filter_tags_map, on='EGID', how='left')
            topo_df_catg_raw = topo_df_catg_raw.with_columns([
                pl.lit('na').alias('filter_tag')
            ])
            for tag_col in cols_as_filter_tags: 
                tag_str = tag_col.split('__')[-1]
                topo_df_catg_raw = topo_df_catg_raw.with_columns([
                    pl.when(pl.col(tag_col)
                    ).then(pl.lit(tag_str)
                    ).otherwise(pl.col('filter_tag')
                    ).alias('filter_tag')
                ])

            # make color mapping CATG ----------
            topo_df_catg = topo_df_catg_raw.group_by('EGID').agg([
                pl.col('GKLAS').first().alias('GKLAS'),
                pl.col('GSTAT').first().alias('GSTAT'),
                pl.col('GBAUJ_cat').first().alias('GBAUJ_cat'),
                pl.col('are_typ').first().alias('are_typ'),
                pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                pl.col('grid_node').first().alias('grid_node'),
                pl.col('heatpump_TF').first().alias('heatpump_TF'),
                pl.col('filter_tag').first().alias('filter_tag'),
            ])


            # all_categories = (
            #     topo_df_catg
            #     .select(cols_catg_charact_inst)
            #     .unpivot(on=cols_catg_charact_inst)   # equivalent to melt()
            #     .select("value")
            #     .unique()
            #     .to_series()
            #     .to_list()
            # )
            # categoires_not_in_color_map = [cat for cat in all_categories if cat not in color_map_catg.keys()]
            # color_map_catg.update({
            #     cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            #     for i, cat in enumerate(categoires_not_in_color_map)
            # })

            # alternative color_mapping => by column
            column_categories = {}
            for col in cols_catg_charact_inst:
                column_categories[col] = (
                    topo_df_catg
                    .select(pl.col(col))
                    .unique()
                    .to_series()
                    .drop_nulls()
                    .to_list()
                )
                color_map_catg = {}
                palette = qualitative.Plotly  # or Dark24, Set3, etc.
                for col, cats in column_categories.items():
                    color_map_catg[col] = {
                        cat: palette[i % len(palette)]
                        for i, cat in enumerate(cats)
                    }




            # Plots ----------------------------------------
            df_plot = pred_inst_df.join(topo_df_catg, on='EGID', how='left')

            n_rows_subplot_all = int(np.ceil(len(list(df_plot['iter_round'].unique())) / n_cols_subplot)) 
            n_rows_subplot = n_rows_subplot_all if n_rows_subplot_all <= max_n_rows_catg_charact else max_n_rows_catg_charact

            iter_list = sorted(list(df_plot['iter_round'].unique()))
            if len(iter_list)> (n_rows_subplot * n_cols_subplot):
                iter_list = iter_list[:n_rows_subplot * n_cols_subplot]

            fig = make_subplots(rows = n_rows_subplot, cols = n_cols_subplot,
                    subplot_titles = [f'iter {iter}' for iter in iter_list])

            for i_iter, iter in enumerate(iter_list):
                # df_iter = df_plot.loc[df_plot['iter_round'] == iter]
                df_iter = df_plot.filter(df_plot['iter_round'] == iter)

                for i_col, col in enumerate(cols_catg_charact_inst):

                    vc = df_iter[col].value_counts().to_dict(as_series=True)
                    # categories = list(vc.keys()) # vc[col]
                    # counts = list(vc.values())    # vc["count"]
                    categories = vc[col]
                    counts = vc["count"]
                    # scen_col_legend = f"{scen} – {col} - iter {iter}"

                    for idx, (cat, count) in enumerate(zip(categories, counts)):
                        show_legend_for_trace = True # if idx == 0 else False
                        scen_col_legend = f"{col} - {cat}"
                        fig.add_trace(
                            go.Bar(
                                x=[col],
                                y=[count],
                                marker=dict(color=color_map_catg[col][cat]),   
                                name = scen_col_legend,
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
                            row = (i_iter // n_cols_subplot) + 1,
                            col = (i_iter % n_cols_subplot) + 1,
                        )

                        # append export plot data
                        row_plot_data = {
                            'scen':           scen,
                            'iter':           iter,
                            'col':            col,
                            'category':       cat,
                            'count':         count,
                        }
                        export_plot_data_list.append(row_plot_data)



            # update layout + export
            fig.update_layout(
                title_text=f'Charact. of EGID with new PV (categorical by iteration, scen: {scen})', 
                barmode= 'stack',
                template= 'plotly_white',
            )
            if plot_show:
                fig.show()
            if plot_export and not self.visual_sett.export_only_agg_comparison_plots:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_bar_catgcharact_newinst___{scen}.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_bar_catgcharact_newinst___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_bar_catgcharact_newinst (for: {scen})', self.visual_sett.log_name)


        # export plot data as csv
        if plot_export:
            export_plot_data_df = pd.DataFrame(export_plot_data_list)
            os.makedirs(f'{self.visual_sett.visual_static_export_path}', exist_ok=True)
            export_df_name         = 'plot_agg_bar_catgcharact_newinst___export_plot_data___'
            export_df_name_scenlen = f'{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen'
            if os.path.exists(f'{self.visual_sett.visual_static_export_path}/{export_df_name_scenlen}.csv'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen*.csv'))
                os.rename(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', 
                          f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.csv')
            export_plot_data_df.to_csv(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', index=False)
            print_to_logfile(f'\texport: {export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', self.visual_sett.log_name)
