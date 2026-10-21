# python integrated packages
import sys
import os as os
import copy
import glob
import json

# external packages
import geopandas as gpd
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLineInstalledCap(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_installedCap(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_installedCap', self.visual_sett.log_name)

        # available color palettes
        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     

        fig_agg_color_palettes = [
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'Mint', 'solar',
            'Teal', 'Magenta', 'Plotly3', 'Viridis', 'Turbo', 'Blackbody', 'Bluered', 
            'Aggrnyl', 'Agsunset', 'Rainbow', 'Burg', 'Cividis', 'Inferno', 'Magma', 'Plasma',

            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'Mint', 'solar',
            'Teal', 'Magenta', 'Plotly3', 'Viridis', 'Turbo', 'Blackbody', 'Bluered', 
            'Aggrnyl', 'Agsunset', 'Rainbow', 'Burg', 'Cividis', 'Inferno', 'Magma', 'Plasma',

            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'Mint', 'solar',
            'Teal', 'Magenta', 'Plotly3', 'Viridis', 'Turbo', 'Blackbody', 'Bluered', 
            'Aggrnyl', 'Agsunset', 'Rainbow', 'Burg', 'Cividis', 'Inferno', 'Magma', 'Plasma',
        ]

        fig_agg_pmonth = go.Figure()
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            Map_egid_pv = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/Map_egid_pv.parquet')

            gm_shp = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson') 
            pv_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/pv_gdf.geojson')
            gwr_all_building_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_all_building_gdf.geojson')


            # get pvinst_df from topo
            egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list= [], [], [], [], [], []
            for k,v, in topo.items():
                egid_list.append(k)
                inst_TF_list.append(v['pv_inst']['inst_TF'])
                info_source_list.append(v['pv_inst']['info_source'])
                BeginOp_list.append(v['pv_inst']['BeginOp'])
                TotalPower_list.append(v['pv_inst']['TotalPower'])
                bfs_list.append(v['gwr_info']['bfs'])

            pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list, 
                                    'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list})
            pvinst_df = pvinst_df.loc[pvinst_df['inst_TF'] == True]

            pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')
            pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
            pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
            pvinst_df['bfs'] = pvinst_df['bfs'].astype(str)


            # arange pv_df for comparison
            instcomp_df = copy.deepcopy(pv_gdf)
            instcomp_df.rename(columns={'BeginningOfOperation': 'BeginOp', }, inplace=True)
            instcomp_df = instcomp_df.loc[instcomp_df['BFS_NUMMER'].isin([str(str_nr) for str_nr in self.pvalloc_scen.bfs_numbers])]


            gwr_all_building_gdf.drop(columns=['index_right', ], inplace=True, errors='ignore')
            gwr_buff_select_BUT_year = copy.deepcopy(gwr_all_building_gdf)
            gwr_buff_select_BUT_year = gwr_buff_select_BUT_year.loc[(gwr_buff_select_BUT_year['GGDENR'].isin(self.pvalloc_scen.bfs_numbers)) &
                                                                    (gwr_buff_select_BUT_year['GSTAT'].isin(self.pvalloc_scen.GWRspec_GSTAT)) & 
                                                                    (gwr_buff_select_BUT_year['GKLAS'].isin(self.pvalloc_scen.GWRspec_GKLAS)) ]
            # gwr_buff_select_BUT_year.set_crs("EPSG:32632", allow_override=True, inplace=True)
            # gwr_buff_select_BUT_year['geometry'] = gwr_buff_select_BUT_year['geometry'].buffer(self.pvalloc_scen.SOLKAT_GWR_EGID_buffer_size)
            gwr_buff_select_BUT_year = gwr_buff_select_BUT_year.to_crs(gm_shp.crs)
            instcomp_df = instcomp_df.to_crs(gm_shp.crs)

            # # merge and add all the buildings in the gwr selection
            instcomp_mrg_df = gpd.sjoin(instcomp_df, gwr_buff_select_BUT_year, how='left', predicate='within')
            instcomp_mrg_df = instcomp_mrg_df.merge(Map_egid_pv, on = 'xtf_id', how = 'inner')

            instcomp_mrg_df = instcomp_mrg_df.loc[instcomp_mrg_df['TotalPower'] <= 30]
            # not a good subselection because, I need to look at the range of houses during a time, not all the houses that are predicted to install 
            # => might cause a miss of several install installations in that geo range because houses were not predicted to install. 
            # instcomp_mrg_df = instcomp_mrg_df.loc[instcomp_mrg_df['EGID'].isin(pvinst_df['EGID'].unique())]



            # plot ind - line: Installed Capacity per Month ===========================
            if True:  #['plot_ind_line_installedCap_per_month']:
                checkpoint_to_logfile('plot_ind_line_installedCap_per_month', self.visual_sett.log_name)

                capa_month_df = pvinst_df.copy()
                capa_month_df['BeginOp_month'] = capa_month_df['BeginOp'].dt.to_period('M')
                capa_month_df = capa_month_df.groupby(['BeginOp_month', 'info_source'])['TotalPower'].sum().reset_index().copy()
                capa_month_df['BeginOp_month'] = capa_month_df['BeginOp_month'].dt.to_timestamp()
                capa_month_built = capa_month_df.loc[capa_month_df['info_source'] == 'pv_df'].copy()
                capa_month_predicted = capa_month_df.loc[capa_month_df['info_source'] == 'alloc_algorithm'].copy()

                capa_cumm_month_df =  pvinst_df.copy()
                capa_cumm_month_df['BeginOp_month'] = capa_cumm_month_df['BeginOp'].dt.to_period('M')
                capa_cumm_month_df.sort_values(by='BeginOp_month', inplace=True)
                capa_cumm_month_df = capa_cumm_month_df.groupby(['BeginOp_month',])['TotalPower'].sum().reset_index().copy()
                capa_cumm_month_df['Cumm_TotalPower'] = capa_cumm_month_df['TotalPower'].cumsum()
                capa_cumm_month_df['BeginOp_month'] = capa_cumm_month_df['BeginOp_month'].dt.to_timestamp()

                capa_year_df = pvinst_df.copy()
                capa_year_df['BeginOp_year'] = capa_year_df['BeginOp'].dt.to_period('Y')
                capa_year_df = capa_year_df.groupby(['BeginOp_year', 'info_source'])['TotalPower'].sum().reset_index().copy()
                capa_year_df['BeginOp_year'] = capa_year_df['BeginOp_year'].dt.to_timestamp()
                capa_year_built = capa_year_df.loc[capa_year_df['info_source'] == 'pv_df'].copy()
                capa_year_predicted = capa_year_df.loc[capa_year_df['info_source'] == 'alloc_algorithm'].copy()

                capa_cumm_year_df =  pvinst_df.copy()
                capa_cumm_year_df['BeginOp_year'] = capa_cumm_year_df['BeginOp'].dt.to_period('Y')
                capa_cumm_year_df.sort_values(by='BeginOp_year', inplace=True)
                capa_cumm_year_df = capa_cumm_year_df.groupby(['BeginOp_year',])['TotalPower'].sum().reset_index().copy()
                capa_cumm_year_df['Cumm_TotalPower'] = capa_cumm_year_df['TotalPower'].cumsum()
                capa_cumm_year_df['BeginOp_year'] = capa_cumm_year_df['BeginOp_year'].dt.to_timestamp()
                # capa_cumm_year_built = capa_cumm_year_df.loc[capa_cumm_year_df['info_source'] == 'pv_df'].copy()
                # capa_cumm_year_predicted = capa_cumm_year_df.loc[capa_cumm_year_df['info_source'] == 'alloc_algorithm'].copy()

                instcomp_month_df = copy.deepcopy(instcomp_mrg_df)
                instcomp_month_df['BeginOp'] = pd.to_datetime(instcomp_month_df['BeginOp'], format='%Y-%m-%d')
                instcomp_month_df['BeginOp_month'] = instcomp_month_df['BeginOp'].dt.to_period('M')
                instcomp_month_df = instcomp_month_df.groupby(['BeginOp_month', ])['TotalPower'].sum().reset_index().copy()
                instcomp_month_df['BeginOp_month'] = instcomp_month_df['BeginOp_month'].dt.to_timestamp()

                instcomp_year_df = copy.deepcopy(instcomp_mrg_df)
                instcomp_year_df['BeginOp'] = pd.to_datetime(instcomp_year_df['BeginOp'], format='%Y-%m-%d')
                instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp'].dt.to_period('Y')
                instcomp_year_df = instcomp_year_df.groupby(['BeginOp_year',])['TotalPower'].sum().reset_index().copy()
                instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp_year'].dt.to_timestamp()
                instcomp_year_df['Cumm_TotalPower'] = instcomp_year_df['TotalPower'].cumsum()
                instcomp_year_df['growth_cumm_TotalPower'] = instcomp_year_df['Cumm_TotalPower'].diff() / instcomp_year_df['Cumm_TotalPower'].shift(1) 
                instcomp_year_df[['Cumm_TotalPower', 'growth_cumm_TotalPower']] 


                # -- DEBUGGING ----------
                if plot_export:
                    capa_year_df.to_csv(f'{self.visual_sett.visual_path}/{scen}_capa_year_df.csv')
                    capa_cumm_year_df.to_csv(f'{self.visual_sett.visual_path}/{scen}_capa_cumm_year_df.csv')
                    instcomp_mrg_df.to_csv(f'{self.visual_sett.visual_path}/{scen}_instcomp_mrg_df.csv') 
                # -- DEBUGGING ----------


                # plot ----------------
                fig1 = go.Figure()

                fig1.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'],      y=capa_month_built['TotalPower'],             line = dict(color = 'deepskyblue'),     name='built (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'],  y=capa_month_predicted['TotalPower'],         line = dict(color = 'cornflowerblue'),  name='predicted (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'],         y=capa_month_df['TotalPower'],                line = dict(color = 'navy'),            name='built + predicted (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_cumm_month_df['BeginOp_month'],    y=capa_cumm_month_df['Cumm_TotalPower'],      line = dict(color = 'blue'),            name='cumulative built (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=[None, ],                               y=[None, ],                                   line = dict(color = 'white'),           name='-',                         mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'],        y=capa_year_built['TotalPower'],              line = dict(color = 'lightgreen'),      name='built (year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'],    y=capa_year_predicted['TotalPower'],          line = dict(color = 'limegreen'),       name='predicted (year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'],           y=capa_year_df['TotalPower'],                 line = dict(color = 'forestgreen'),     name='built + predicted (year)', mode='lines+markers',))
                fig1.add_trace(go.Scatter(x=capa_cumm_year_df['BeginOp_year'],      y=capa_cumm_year_df['Cumm_TotalPower'],       line = dict(color = 'purple'),          name='cumulative built (year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=[None, ],                               y=[None, ],                                   line = dict(color = 'white'),           name='-',                         mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=instcomp_month_df['BeginOp_month'],     y=instcomp_month_df['TotalPower'],            line = dict(color = 'orange'),          name='pv_df raw (blw 30kwp month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'],       y=instcomp_year_df['TotalPower'],             line = dict(color = 'darkorange'),      name='pv_df raw (blw 30kwp year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'],       y=instcomp_year_df['Cumm_TotalPower'],        line = dict(color = 'orangered'),       name='pv_df raw (blw 30kwp cumulative)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'],       y=instcomp_year_df['growth_cumm_TotalPower'], line = dict(color = 'orangered'),       name='pv_df raw (blw 30kwp growth)', mode='lines+markers'))

                fig1.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='Time steps',
                    title = f'Installed Capacity per Month (weather year: {self.pvalloc_scen.WEAspec_weather_year})'
                )

                # add T0 prediction
                T0_prediction = f'{self.pvalloc_scen.T0_year_prediction}-01-01 00:00:00' 
                date = '2008-01-01 00:00:00'
                fig1.add_shape(
                    # Line Vertical
                    dict(
                        type="line",
                        x0=T0_prediction,
                        y0=0,
                        x1=T0_prediction,
                        y1=max(capa_year_df['TotalPower'].max(), capa_cumm_year_df['Cumm_TotalPower'].max()),  # Dynamic height
                        line=dict(color="black", width=1, dash="dot"),
                    )
                )
                fig1.add_annotation(
                    x=  T0_prediction,
                    y=max(capa_year_df['TotalPower'].max(), capa_cumm_year_df['Cumm_TotalPower'].max()),
                    text="T0 Prediction",
                    showarrow=False,
                    yshift=10
                )

                fig1 = self.add_scen_name_to_plot(fig1, scen, self.pvalloc_scen)
                fig1 = self.set_default_fig_zoom_year(fig1, self.visual_sett.default_zoom_year, capa_year_df, 'BeginOp_year')

                if plot_show:
                    fig1.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig1.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_line_installedCap_per_month.html')
                    else:
                        fig1.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_line_installedCap_per_month.html')
                    print_to_logfile(f'\texport: plot_ind_line_installedCap_per_month.html (for: {scen})', self.visual_sett.log_name)                    



            # plot ind - line: Installed Capacity per BFS ===========================
            if False: # self.visual_sett.plot_ind_line_installedCap_TF[0]:  #['plot_ind_line_installedCap_per_BFS']:
                checkpoint_to_logfile(f'plot_ind_line_installedCap_per_BFS', self.visual_sett.log_name)
                capa_bfs_df = pvinst_df.copy()
                gm_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
                gm_gdf.rename(columns={'BFS_NUMMER': 'bfs'}, inplace=True)
                gm_gdf['bfs'] = gm_gdf['bfs'].astype(str)
                capa_bfs_df = capa_bfs_df.merge(gm_gdf[['bfs', 'NAME']], on='bfs', how = 'left' )
                capa_bfs_df['BeginOp_month'] = capa_bfs_df['BeginOp'].dt.to_period('M')
                capa_bfs_month_df = capa_bfs_df.groupby(['BeginOp_month', 'bfs'])['TotalPower'].sum().reset_index().copy()
                capa_bfs_month_df['BeginOp_month'] = capa_bfs_month_df['BeginOp_month'].dt.to_timestamp()

                capa_bfs_df['BeginOp_year'] = capa_bfs_df['BeginOp'].dt.to_period('Y')
                capa_bfs_year_df = capa_bfs_df.groupby(['BeginOp_year', 'bfs'])['TotalPower'].sum().reset_index().copy()
                capa_bfs_year_df['BeginOp_year'] = capa_bfs_year_df['BeginOp_year'].dt.to_timestamp()

                # plot ----------------
                fig2 = go.Figure()
                for bfs in capa_bfs_month_df['bfs'].unique():
                    name = gm_gdf.loc[gm_gdf['bfs'] == bfs, 'NAME'].values[0]
                    subdf = capa_bfs_month_df.loc[capa_bfs_month_df['bfs'] == bfs].copy()
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_month'], y=subdf['TotalPower'], name=f'{name} (by month)', legendgroup = 'By Month',  mode = 'lines'))

                for bfs in capa_bfs_year_df['bfs'].unique():
                    name = gm_gdf.loc[gm_gdf['bfs'] == bfs, 'NAME'].values[0]
                    subdf = capa_bfs_year_df.loc[capa_bfs_year_df['bfs'] == bfs].copy()
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_year'], y=subdf['TotalPower'], name=f'{name} (by year)', legendgroup = 'By Year', mode = 'lines'))

                fig2.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='BFS',
                    title = f'Installed Capacity per Municipality (BFS) (weather year: {self.pvalloc_scen.WEAspec_weather_year})',
                    showlegend=True, 
                    legend=dict(
                        title='Legend',  # You can customize the legend title here
                        itemsizing='trace',  # Control the legend item sizing (can be 'trace' or 'constant')
                    )
                )

                fig2.add_shape(
                    # Line Vertical
                    dict(
                        type="line",
                        x0=T0_prediction,
                        y0=0,
                        x1=T0_prediction,
                        y1=capa_bfs_year_df['TotalPower'],  # Dynamic height
                        line=dict(color="black", width=1, dash="dot"),
                    )
                )
                fig2.add_annotation(
                    x=  T0_prediction,
                    y=1,
                    text="T0 Prediction",
                    showarrow=False,
                    yshift=10
                )

                fig2 = self.add_scen_name_to_plot(fig2, scen, self.pvalloc_scen)
                fig2 = self.set_default_fig_zoom_year(fig2, self.visual_sett.default_zoom_year, capa_bfs_year_df, 'BeginOp_year')

                if plot_show:
                    fig2.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig2.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_line_installedCap_per_BFS.html')
                    else:
                        fig2.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_line_installedCap_per_BFS.html')
                    print_to_logfile(f'\texport: plot_ind_line_installedCap_per_BFS.html (for: {scen})', self.visual_sett.log_name)


            # plot add aggregated - line: Installed Capacity per Year ===========================
            if True:  #['plot_ind_line_installedCap_per_month']:

                color_pal = trace_color_dict[fig_agg_color_palettes[i_scen]]
                color_pal_idx = len(color_pal)
                # Some palettes are short; wrap negative offsets to a valid index.
                def color_at(offset):
                    return color_pal[offset % color_pal_idx]

                # color_allscen_list = [list(trace_color_dict.keys())[i_scen] for i_scen in range(len(self.visual_sett.pvalloc_scen_list))]
                # color_palette = trace_color_dict[list(trace_color_dict.keys())[i_scen]]

                # fig_agg_pmonth.add_trace(go.Scatter(x=[0,], y=[0,],  name=f'',opacity=1, ))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],  name='',opacity=0, ))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'black'), name=f'{scen}',opacity=0, mode='lines+markers'))

                # fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'],          opacity = 0.75, line = dict(color = color_palette[0+0]),    name='-- built (month)', mode='lines+markers'))
                # fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'],  opacity = 0.75, line = dict(color = color_palette[0+1]),    name='-- predicted (month)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],                opacity = 0.75, line = dict(color = color_at(-2)),    name='-- built + predicted (month)', mode='lines+markers'))
                # fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'],             opacity = 0.75, line = dict(color = color_palette[0+2]),    name='-- built (year)', mode='lines+markers'))
                # fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'],     opacity = 0.75, line = dict(color = color_palette[0+3]),    name='-- predicted (year)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'],                   opacity = 0.75, line = dict(color = color_at(-5)),    name='-- built + predicted (year)', mode='lines+markers',))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_cumm_year_df['BeginOp_year'], y=capa_cumm_year_df['Cumm_TotalPower'],    opacity = 0.75, line = dict(color = color_at(-4)),    name='-- cumulative built + pred (year)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=instcomp_month_df['BeginOp_month'], y=instcomp_month_df['TotalPower'],        opacity = 0.75, line = dict(color = color_at(-5)),    name='-- pv_df (blw 30kwp month)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['TotalPower'],           opacity = 0.75, line = dict(color = color_at(-6)),    name='-- pv_df (blw 30kwp year)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['Cumm_TotalPower'],      opacity = 0.75, line = dict(color = color_at(-7)),    name='-- pv_df (blw 30kwp cumulative)', mode='lines+markers'))

                # export plot add aggregated - line: Installed Capacity per Year 
                if i_scen == len(self.visual_sett.pvalloc_scen_list)-1:
                    fig_agg_pmonth.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='Time steps',
                    title = f'Installed Capacity per Month/Year, {len(self.visual_sett.pvalloc_scen_list)}scen (weather year: {self.pvalloc_scen.WEAspec_weather_year})'
                    )

                    # add T0 prediction
                    T0_prediction = f'{self.pvalloc_scen.T0_year_prediction}-01-01 00:00:00'
                    date = '2008-01-01 00:00:00'
                    fig_agg_pmonth.add_shape(
                        # Line Vertical
                        dict(
                            type="line",
                            x0=T0_prediction,
                            y0=0,
                            x1=T0_prediction,
                            y1=max(capa_year_df['TotalPower'].max(), capa_cumm_year_df['Cumm_TotalPower'].max()),  # Dynamic height
                            line=dict(color="black", width=1, dash="dot"),
                        )
                      )
                    fig_agg_pmonth.add_annotation(
                        x=  T0_prediction,
                        y=max(capa_year_df['TotalPower'].max(), capa_cumm_year_df['Cumm_TotalPower'].max()),
                        text="T0 Prediction",
                        showarrow=False,
                        yshift=10
                    )

                    fig_agg_pmonth = self.set_default_fig_zoom_year(fig_agg_pmonth, self.visual_sett.default_zoom_year, capa_year_df, 'BeginOp_year')

                    if plot_show:
                        fig_agg_pmonth.show()
                    if plot_export:
                        if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                            n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                            os.rename(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                                      f'{self.visual_sett.visual_path}/plot_agg_line_installedCap___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
                        fig_agg_pmonth.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
                        print_to_logfile(f'\texport: plot_agg_line_installedCap___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)
