# python integrated packages
import sys
import os as os
import copy
import glob
import json
import random

# external packages
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLinePVproduction(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_PVproduction(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_PVproduction', self.visual_sett.log_name)

        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     

        fig_agg_color_palettes = ['Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  'Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',
                                  ]
        fig_agg = go.Figure()
        export_plot_data_list = []
        CSTRspec_capacity_type_added_TF = False
        max_end_modelled_pv_list = []

        unique_node_color_map = {}


        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            # setup + import --------------------------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            gridnode_df_paths = glob.glob(f'{self.visual_sett.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_*.parquet')
            trange_prediction = pd.read_parquet(f'{self.visual_sett.mc_data_path}/trange_prediction.parquet')
            topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            gridnode_df_main    = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
            constrcapa =  pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/constrcapa.parquet')


            # select nodes to for stacking ------------------
            loss_nodes = gridnode_df_main.group_by('grid_node').agg([
                pl.sum('feedin_atnode_loss_kW').alias('feedin_atnode_loss_kW'),
            ]).sort('feedin_atnode_loss_kW', descending=True)

            n_top_nodes = self.visual_sett.plot_ind_line_PVproduction_bynode_specs['n_top_loss_nodes']
            manually_added_nodes_sett = self.visual_sett.plot_ind_line_PVproduction_bynode_specs['select_nodes_stacked_traces']
            n_pick_top_nodes = n_top_nodes - len(manually_added_nodes_sett) 
            top_loss_nodes = sorted(manually_added_nodes_sett + loss_nodes.head(n_pick_top_nodes).select('grid_node').unique().to_series().to_list() )

            node_not_in_map = [node for node in top_loss_nodes if node not in topo.keys()]                    
            unique_node_color_map.update({
                node: pc.qualitative.Plotly[i % len(pc.qualitative.Plotly)]
                for i, node in enumerate(node_not_in_map)
            })


            # add constracapa  capacity ------------------
            constrcapa = constrcapa.merge(trange_prediction[['date', 'n_iter']] , on='date', how='left')
            constrcapa = constrcapa.loc[ ~ constrcapa['n_iter'].isna()]
            # if self.pvalloc_scen.CSTRspec_capacity_type == 'ep2050_zerobasis':
            #     if not CSTRspec_capacity_type_added_TF:
            #         CSTRspec_capacity_type_added_TF = True

            #         constrcapa = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/constrcapa.parquet')
            #         constrcapa = constrcapa.loc[constrcapa['year'] >= self.pvalloc_scen.T0_year_prediction].copy()
            #         constrcapa['n_inter'] = range(1, len(constrcapa) + 1)
            #         fig_agg.add_trace(go.Scatter(x=constrcapa['n_inter'], y=constrcapa['constr_capacity_kw'], name='EP2050+ Inst.Cap. (kW)', mode='lines', line=dict(color='darkred', width=3, dash='dash') )) # , marker=dict(symbol='star',), ))


            # girdnode_df: transform and prep ------------------
            gridnode_df_by_iter_list, agg_bynode_df_by_iter_list = [] , []
            print(f'** processing gridnode_df for scen {scen} **\n** n_paths: {len(gridnode_df_paths)} **') if i_scen == 0 else None
            for i_path, path in enumerate( gridnode_df_paths ):
                n_iter = int(path.split('gridnode_df_')[1].split('.parquet')[0])

                # taken 1:1 from plot_ind_line_productionHOY_per_node() ==========
                gridnode_df = pl.read_parquet(path)
                gridnode_df = gridnode_df.with_columns([
                    pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                ])
                gridnode_df = gridnode_df.sort("t_int", descending=False)
                if i_path < 5: 
                    print(f'\t\t processing iter {n_iter} from path: {path}') 
                    print(f'\t\t shape gridnode_df: {gridnode_df.shape}, rows / 8760: {gridnode_df.shape[0] / 8760:.2f} ')


                # calc holding capacity
                gridnode_df = gridnode_df.with_columns([
                    ((pl.col('kW_threshold') - pl.col('max_demand_feedin_atnode_kW')) / pl.col('kW_threshold')).alias('holding_capacity')
                ])

                # aggregate to total production per HOY
                gridnode_total_df = gridnode_df.group_by(['t', 't_int']).agg([

                    pl.col('demand_kW').sum().alias('demand_kW'),
                    pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                    pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                    pl.col('netdemand_kW').sum().alias('netdemand_kW'),

                    pl.col('feedin_atnode_kW').sum().alias('feedin_atnode_kW'),
                    pl.col('demand_atnode_kW').sum().alias('demand_atnode_kW'),
                    pl.col('max_demand_feedin_atnode_kW').sum().alias('max_demand_feedin_atnode_kW'),
                    pl.col('feedin_atnode_taken_kW').sum().alias('feedin_atnode_taken_kW'),
                    pl.col('feedin_atnode_loss_kW').sum().alias('feedin_atnode_loss_kW'),
                    pl.col('kW_threshold').sum().alias('kW_threshold'),

                    pl.col('holding_capacity').min().alias('holding_capacity_min'),
                    pl.col('holding_capacity').max().alias('holding_capacity_max'),
                    pl.col('holding_capacity').mean().alias('holding_capacity_mean'),
                ])
                gridnode_total_df = gridnode_total_df.sort("t_int", descending=False)
                # ==========

                agg_gridnode_row = gridnode_total_df.select([
                    pl.sum('demand_kW').alias('demand_kW'),
                    pl.sum('pvprod_kW').alias('pvprod_kW'),
                    pl.sum('selfconsum_kW').alias('selfconsum_kW'),
                    pl.sum('netdemand_kW').alias('netdemand_kW'),

                    pl.sum('feedin_atnode_kW').alias('feedin_atnode_kW'), 
                    pl.sum('demand_atnode_kW').alias('demand_atnode_kW'), 
                    pl.sum('max_demand_feedin_atnode_kW').alias('max_demand_feedin_atnode_kW'),
                    pl.sum('feedin_atnode_taken_kW').alias('feedin_atnode_taken_kW'), 
                    pl.sum('feedin_atnode_loss_kW').alias('feedin_atnode_loss_kW'), 
                    pl.first('kW_threshold').alias('kW_threshold'),

                    pl.col('holding_capacity_min').min().alias('holding_capacity_min_abs'),
                    pl.col('holding_capacity_mean').min().alias('holding_capacity_min_of_mean'),
                    pl.col('holding_capacity_mean').mean().alias('holding_capacity_mean_of_mean'),
                    pl.col('holding_capacity_mean').max().alias('holding_capacity_max_of_mean'),
                    pl.col('holding_capacity_max').max().alias('holding_capacity_max_abs'),
                ]).with_columns([
                    pl.lit(n_iter).alias('n_iter'),
                    pl.lit(scen).alias('scen'),
                ])

                # add TotalPower per iter
                trange_prediction = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/trange_prediction.parquet')
                n_iter_date_min = trange_prediction.loc[trange_prediction['n_iter'] == n_iter, 'date'].values[0] - pd.Timedelta(days=364)
                n_iter_date_max = trange_prediction.loc[trange_prediction['n_iter'] == n_iter, 'date'].values[0]
                topo_egid_TotalPower_list = []
                for k, v in topo.items():
                    Topo_BeginOp_date = pd.to_datetime(v['pv_inst']['BeginOp'], errors='coerce')
                    if (Topo_BeginOp_date >= n_iter_date_min ) & (Topo_BeginOp_date <= n_iter_date_max ) & (v['pv_inst']['inst_TF'] == True):
                        topo_egid_TotalPower_list.append(v['pv_inst']['TotalPower'])

                agg_gridnode_row = agg_gridnode_row.with_columns([
                    pl.lit(float(sum(topo_egid_TotalPower_list))).alias('TotalPower'),
                ])
                gridnode_df_by_iter_list.append(agg_gridnode_row)


                # add individual grid specific rows to second df
                agg_bynode_row = gridnode_df.filter(pl.col('grid_node').is_in(top_loss_nodes)).group_by('grid_node').agg([
                    pl.lit(n_iter).alias('n_iter'),
                    pl.lit(scen).alias('scen'),
                    pl.sum('feedin_atnode_loss_kW').alias('feedin_atnode_loss_kW'),
                ])
                agg_bynode_df_by_iter_list.append(agg_bynode_row)


            gridnode_df_by_iter     = pl.concat(gridnode_df_by_iter_list, how="vertical")
            gridnode_df_by_iter     = gridnode_df_by_iter.with_columns([
                pl.col('TotalPower').cum_sum().alias('TotalPower_cum')
            ])
            agg_bynode_df_by_iter   = pl.concat(agg_bynode_df_by_iter_list, how="vertical")


            gridnode_df_by_iter = gridnode_df_by_iter.with_columns([
                (pl.col('feedin_atnode_loss_kW') / pl.col('feedin_atnode_taken_kW')).alias('netfeedin_loss_ratio')
            ])
            gridnode_df_by_iter = gridnode_df_by_iter.sort("n_iter", descending=False)


            # topo_df: transform and prep ------------------
            egid_list, dfuid_list, info_source_list, inst_TF_list, grid_node_list, BeginOp_list = [], [], [], [], [], []
            for k, v in topo.items():
                dfuid_tupls = [tpl[1] for tpl in v['pv_inst']['dfuid_w_inst_tuples'] if tpl[3] > 0.0]

                egid_list.append(k)
                dfuid_list.append(dfuid_tupls)    # dfuid_list.append(v['pv_inst']['df_uid_w_inst'])
                info_source_list.append(v['pv_inst']['info_source'])
                inst_TF_list.append(v['pv_inst']['inst_TF'])
                grid_node_list.append(v['grid_node'])
                BeginOp_list.append(v['pv_inst']['BeginOp'])

            topo_df = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'info_source': info_source_list, 
                                    'inst_TF': inst_TF_list, 'grid_node': grid_node_list, 'BeginOp': BeginOp_list,})
            topo_df['BeginOp'] = pd.to_datetime(topo_df['BeginOp'], errors='coerce')

            topo_df_iter = copy.deepcopy(trange_prediction)
            topo_df_iter['n_EGID_pvinst'] = 0
            for i_row, row in topo_df_iter.iterrows():
                topo_iter_subdf = topo_df.loc[(topo_df['BeginOp'] <= row['date']) & 
                                              (topo_df['BeginOp'] >= row['date'] - pd.Timedelta(days=364)), ]  

                topo_df_iter.iloc[i_row, topo_df_iter.columns.get_loc('n_EGID_pvinst')] = len(topo_iter_subdf)

            topo_alloc_algo = [k for k,v in topo.items() if v['pv_inst']['info_source'] != 'pv_df']
            topo_df_iter['n_EGID_pvinst_cum'] = topo_df_iter['n_EGID_pvinst'].cumsum()
            topo_df_iter['ratio_EGID_pvinst'] = topo_df_iter['n_EGID_pvinst_cum'] / len(topo_alloc_algo)


            # plot ind line ----------------
            if True: 
                fig = go.Figure()
                gridnode_total_df = gridnode_total_df.to_pandas()

                fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['feedin_atnode_kW'],        name='Total feedin_atnode_kW',        mode='lines+markers',  line=dict(color='black',),            marker=dict(symbol='cross',), ))
                fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['demand_atnode_kW'],        name='Total demand_atnode_kW',        mode='lines+markers',  line=dict(color='grey',),             marker=dict(symbol='cross',), ))
                fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['feedin_atnode_taken_kW'],  name='Total feedin_atnode_taken_kW',  mode='lines+markers',  line=dict(color='green',),            marker=dict(symbol='cross',), ))
                fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['feedin_atnode_loss_kW'],   name='Total feedin_atnode_loss_kW',   mode='lines+markers',  line=dict(color='red',  ),            marker=dict(symbol='cross',), ))

                fig.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['n_EGID_pvinst'],                 name='n pvinst insample',           mode='lines+markers',  line=dict(color='purple', dash = 'dot'), opacity = 0.5   , ))
                fig.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['n_EGID_pvinst_cum'],             name='n pvinst cumulative',         mode='lines+markers',  line=dict(color='purple', dash = 'dash'),opacity = 0.8   , ))
                fig.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['ratio_EGID_pvinst'],             name='ratio n pvinst cumulative',     mode='lines+markers',  line=dict(color='purple',)                               , yaxis ='y2'))

                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['demand_kW'],               name='Total demand_kW',               mode='lines+markers',  line=dict(color='darkblue',),         marker=dict(symbol='circle',), ))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['pvprod_kW'],               name='Total pvprod_kW',               mode='lines+markers',  line=dict(color='darkcyan',),         marker=dict(symbol='circle',), ))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['selfconsum_kW'],           name='Total selfconsum_kW',           mode='lines+markers',  line=dict(color='cornflowerblue',),   marker=dict(symbol='circle',), ))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netdemand_kW'],            name='Total netdemand_kW',            mode='lines+markers',  line=dict(color='aquamarine',),       marker=dict(symbol='circle',), ))                   
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['kW_threshold'],                name='Total kW_threshold',              mode='lines+markers',  line=dict(color='blue', ),  marker=dict(symbol='cross',),  ))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_loss_ratio'],        name='netfeedin_loss_ratio',            mode='lines+markers',  line=dict(color='orange', dash = 'dot'), opacity = 1     , yaxis = 'y2'))                
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_max_abs'],       name='Holding Capac max abs',      mode='lines+markers',  line=dict(color='goldenrod', dash = 'dot'),  opacity = 0.5, yaxis = 'y2'))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_max_of_mean'],   name='Holding Capac max of mean',  mode='lines+markers',  line=dict(color='goldenrod', dash = 'dash'), opacity = 0.8, yaxis = 'y2'))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_mean_of_mean'],  name='Holding Capac mean of mean', mode='lines+markers',  line=dict(color='goldenrod', ),              opacity = 1.0, yaxis = 'y2'))
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_min_of_mean'],   name='Holding Capac min of mean',  mode='lines+markers',  line=dict(color='goldenrod', dash = 'dash'), opacity = 0.8, yaxis = 'y2'))     
                # fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_min_abs'],       name='Holding Capac min abs',      mode='lines+markers',  line=dict(color='goldenrod', dash = 'dot'),  opacity = 0.5, yaxis = 'y2'))

                fig.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Energy (kWh Feedin / Loss)',
                    yaxis=dict(
                        title='Energy Feedin / Loss (kW)',
                        range=[0, max(gridnode_df_by_iter['max_demand_feedin_atnode_kW']) * 1.2],  # Adjust range as needed
                    ),
                    yaxis2=dict(
                        title='Holding Capacity',
                        overlaying='y',
                        side='right',
                        range=[0, max(gridnode_df_by_iter['holding_capacity_max_abs']) * 1.2],  # Adjust range as needed
                        showgrid = False, 
                    ),
                    legend_title='Legend',
                    title=f'Development of Energy Feedin / Loss for 1 Year (weather year: {self.pvalloc_scen.WEAspec_weather_year}) Over Time',
                    template='plotly_white',
                )

                fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)



                # export ----------------
                if plot_show:
                    fig.show()
                if plot_export and not self.visual_sett.export_only_agg_comparison_plots:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_line_PVproduction___{scen}.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_line_PVproduction___{scen}.html')
                    print_to_logfile(f'\texport: plot_ind_line_PVproduction.html (for: {scen})', self.visual_sett.log_name)                    


            # plot AGGREGATION ----------------
            color_pal = trace_color_dict[fig_agg_color_palettes[i_scen]]
            color_pal_idx = len(color_pal)
            # index of 2nd zero
            if len(topo_df_iter.index[topo_df_iter["n_EGID_pvinst"] == 0]) >= 2:
                iter_where_n_EGID_pvinst02nd = topo_df_iter.loc[topo_df_iter.index[topo_df_iter["n_EGID_pvinst"] == 0][1], "n_iter"]
                max_end_modelled_pv_list.append(iter_where_n_EGID_pvinst02nd)
            else: 
                max_end_modelled_pv_list.append(topo_df_iter['n_iter'].max())


            fig_agg.add_trace(go.Scatter(x=[None,], y=[None,], name='', opacity=0, line=dict(color = 'rgba(0,0,0,0)',), ))  # empty trace for spacing in legend

            if not self.visual_sett.reduce_information_content: 
                for node in top_loss_nodes:
                    opacity_value = random.uniform(0.2, 0.5)
                    agg_bynode_subdf = agg_bynode_df_by_iter.filter(pl.col('grid_node') == node).to_pandas()
                    fig_agg.add_trace(go.Scatter(x=agg_bynode_subdf['n_iter'], y=agg_bynode_subdf['feedin_atnode_loss_kW'], name=f'node {node} loss_kW', mode='lines+markers', line=dict(color=unique_node_color_map[node], width=1,), marker=dict(symbol='cross',), opacity=opacity_value, stackgroup=scen,))

            fig_agg.add_trace(go.Scatter(x=[None,], y=[None,], name=scen, opacity=0, ))
            fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['demand_atnode_kW'] / 10,           name='Total demand_atnode_kW (1/10)',   mode='lines',         line=dict(color=color_pal[color_pal_idx-1],),                 marker=dict(symbol='cross',),   ))
            fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['feedin_atnode_taken_kW'],          name='Total feedin_atnode_taken_kW',    mode='lines',         line=dict(color=color_pal[color_pal_idx-2],),                 marker=dict(symbol='cross',),   ))
            fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['feedin_atnode_loss_kW'],           name='Total feedin_atnode_loss_kW',     mode='lines',         line=dict(color=color_pal[color_pal_idx-3],),                 marker=dict(symbol='cross',),   ))
            fig_agg.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['n_EGID_pvinst'],                          name='n pvinst insample',               mode='lines+markers', line=dict(color=color_pal[color_pal_idx-4],),                 marker=dict(symbol='circle',),  ))                    
            fig_agg.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=gridnode_df_by_iter['TotalPower'],                      name='TotalPower',                      mode='lines+markers', line=dict(color=color_pal[color_pal_idx-5],),                 marker=dict(symbol='circle',),  ))                    
            fig_agg.add_trace(go.Scatter(x=constrcapa['n_iter'],          y=constrcapa['constr_capacity_kw'],                       name='ConstrCapa (kW)',                 mode='lines+markers', line=dict(color=color_pal[color_pal_idx-6],),                 marker=dict(symbol='circle',),  ))                    
            fig_agg.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['ratio_EGID_pvinst'],                      name='ratio n pvinst cumulative',       mode='lines+markers', line=dict(color=color_pal[color_pal_idx-7],),                 marker=dict(symbol='circle',), yaxis = 'y2'  ))                    


            # fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['feedin_atnode_taken_kW'],          name='Total feedin_atnode_taken_kW',    mode='lines',         line=dict(color=color_pal[color_pal_idx-3],),                 marker=dict(symbol='cross',),   ))
            # fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_loss_ratio'],            name='netfeedin_loss_ratio',            mode='lines',         line=dict(color=color_pal[color_pal_idx-5],),                 marker=dict(symbol='circle',),  yaxis = 'y2'))
            # fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_mean_of_mean'],   name='Holding Capac mean of mean',      mode='lines+markers', line=dict(color=color_pal[color_pal_idx-7],),                 marker=dict(symbol='diamond',), yaxis = 'y2'))
            # fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_max_abs'],        name='Holding Capac max abs',           mode='lines',         line=dict(color=color_pal[color_pal_idx-7], dash = 'dash'),                                   yaxis = 'y2'))
            # fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_min_abs'],        name='Holding Capac min abs',           mode='lines',         line=dict(color=color_pal[color_pal_idx-7], dash = 'dash'),                                   yaxis = 'y2'))
            gridnode_df_by_iter_pd = gridnode_df_by_iter.to_pandas()
            tmp_df = gridnode_df_by_iter_pd.loc[:,
                [
                    'n_iter',
                    'demand_kW', 
                    'pvprod_kW',
                    'selfconsum_kW', 
                    'netdemand_kW', 

                    'feedin_atnode_kW',
                    'demand_atnode_kW',

                    'feedin_atnode_loss_kW',
                    'feedin_atnode_taken_kW',

                    'TotalPower',
                ]]
            # Align by 'n_iter' to avoid length mismatch when assigning
            try:
                tmp_df = tmp_df.merge(
                    topo_df_iter[['n_iter', 'n_EGID_pvinst', 'ratio_EGID_pvinst']],
                    on='n_iter', how='left'
                )
            except Exception as e:
                print(f"Error merging topo_df_iter with gridnode_df_by_iter_pd for scen {scen}: {e}")

            tmp_df.loc[:, 'scen'] = scen

            export_plot_data_list.append(tmp_df)



        # export AGGREGATION ----------------
        fig_agg.update_layout(
            xaxis_title='Time',
            yaxis_title='Energy (kWh Feedin / Loss)',
            xaxis=dict(
                range=[1, max(max_end_modelled_pv_list) * 1.05],  # Adjust range as needed
            ),
            yaxis=dict(
                title='Energy Feedin / Loss (kW)',
                range=[0, max(gridnode_df_by_iter['max_demand_feedin_atnode_kW']) * 1.2],  # Adjust range as needed
            ),  
            yaxis2=dict(
                title='',
                overlaying='y',
                side='right',
                range=[0, max(gridnode_df_by_iter['holding_capacity_mean_of_mean']) * 1.2]  # Adjust range as needed
            ),
            legend_title='Legend',
            title=f'Development of Energy Feedin / Loss for 1 Year (weather year: {self.pvalloc_scen.WEAspec_weather_year}) Over Time',
            template='plotly_white',
        )

        if plot_show:
            fig_agg.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                          f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
            fig_agg.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_agg_line_PVproduction___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)

            # export plot data as csv              
            export_plot_data_df = pd.concat(export_plot_data_list, ignore_index=True)
            os.makedirs(f'{self.visual_sett.visual_static_export_path}', exist_ok=True)
            export_df_name         = 'plot_agg_line_PVproduction___export_plot_data___'
            export_df_name_scenlen = f'{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen'
            if os.path.exists(f'{self.visual_sett.visual_static_export_path}/{export_df_name_scenlen}.csv'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen*.csv'))
                os.rename(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', 
                          f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.csv')
            export_plot_data_df.to_csv(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', index=False)
            print_to_logfile(f'\texport: {export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', self.visual_sett.log_name)
