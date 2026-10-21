# python integrated packages
import sys
import os as os
import json

# external packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, get_bfsnr_name_tuple_list, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndCharacOmittedGwr(VisualizationCore):
    """Plot of the pvalloc_initalization + sanitycheck stage."""

    def plot_ind_charac_omitted_gwr(self, plot_show: bool = True, plot_export: bool = True):
        plot_ind_charac_omitted_gwr_specs = self.visual_sett.plot_ind_charac_omitted_gwr_specs
        checkpoint_to_logfile(f'plot_ind_charac_omitted_gwr', self.visual_sett.log_name)

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # omitted egids from data prep -----      
            gwr_mrg_all_building_in_bfs = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_mrg_all_building_in_bfs.parquet')
            gwr = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr.parquet')
            topo = json.load(open(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_egid.json', 'r'))

            gwr_mrg_all_building_in_bfs.rename(columns={'GGDENR': 'BFS_NUMMER'}, inplace=True)
            gwr_mrg_all_building_in_bfs['BFS_NUMMER'] = gwr_mrg_all_building_in_bfs['BFS_NUMMER'].astype(int)
            gwr_mrg_all_building_in_bfs = gwr_mrg_all_building_in_bfs.loc[gwr_mrg_all_building_in_bfs['BFS_NUMMER'].isin([int(x) for x in self.pvalloc_scen.bfs_numbers])]


            # only look at existing buildings!
            gwr_mrg_all_building_in_bfs = gwr_mrg_all_building_in_bfs.loc[gwr_mrg_all_building_in_bfs['GSTAT'] == '1004']

            omitt_gwregid_from_topo = gwr_mrg_all_building_in_bfs.loc[~gwr_mrg_all_building_in_bfs['EGID'].isin(list(topo.keys()))]

            # subsamples to visualizse ratio of selected gwr in topo to all buildings
            gwr_select_but_not_in_topo = gwr.loc[gwr['GGDENR'].isin([str(x) for x in self.pvalloc_scen.bfs_numbers])]
            gwr_select_but_not_in_topo = gwr_select_but_not_in_topo.loc[~gwr_select_but_not_in_topo['EGID'].isin(list(topo.keys()))]

            gwr_rest = gwr_mrg_all_building_in_bfs.loc[~gwr_mrg_all_building_in_bfs['EGID'].isin(list(topo.keys()))]
            gwr_rest = gwr_rest.loc[~gwr_rest['EGID'].isin(gwr_select_but_not_in_topo['EGID'])]


            # plot discrete characteristics -----
            disc_cols = plot_ind_charac_omitted_gwr_specs['disc_cols']

            fig = go.Figure()
            i, col = 0, disc_cols[0]
            for i, col in enumerate(disc_cols):
                unique_categories = omitt_gwregid_from_topo[col].unique()
                col_df = omitt_gwregid_from_topo[col].value_counts().to_frame().reset_index()

                col_df ['count'] = col_df['count'] / col_df['count'].sum()
                col_df.sum(axis=0)

                # j, cat = 0, unique_categories[1]
                for j, cat in enumerate(unique_categories):
                    if col == 'BFS_NUMMER':
                        cat_label = f'{get_bfsnr_name_tuple_list([cat,])}'
                    elif col == 'GKLAS':
                        if cat in [tpl[0] for tpl in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GKLAS']]:
                            tpl_pick = [x for x in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GKLAS'] if x[0] == cat][0]
                            cat_label = f"{tpl_pick[0]} - {tpl_pick[1]}"
                        else:   
                            cat_label = cat
                    elif col == 'GSTAT':
                        if cat in [tpl[0] for tpl in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GSTAT']]:
                            tpl_pick = [x for x in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GSTAT'] if x[0] == cat][0]
                            cat_label = f"{tpl_pick[0]} - {tpl_pick[1]}"
                        else:   
                            cat_label = cat
                    elif col == 'GKAT':
                        cat_label = cat

                    count_value = col_df.loc[col_df[col] == cat, 'count'].values[0]
                    fig.add_trace(go.Bar(x=[col], y=[count_value], 
                        name=cat_label,
                        text=f'{count_value:.2f} - {cat_label}',  # Add text to display the count
                        textposition='outside'    # Position the text outside the bar
                    ))
                fig.add_trace(go.Scatter(x=[col], y=[0], name=col, opacity=0,))  
                fig.add_trace(go.Scatter(x=[col], y=[0], name='', opacity=0,))  

            # add overview for all buildings covered by topo from gwr
            fig.add_trace(go.Bar(x=['share EGID in topo',], y=[len(list(topo.keys()))/gwr_mrg_all_building_in_bfs['EGID'].nunique(),], 
                                name=f'gwrEGID_in_topo ({len(list(topo.keys()))} nr in sample)',
                                text=f'{len(list(topo.keys()))/len(gwr_mrg_all_building_in_bfs["EGID"].unique()):.2f} ({len(list(topo.keys()))} nEGIDs)',  # Add text to display the count
                                textposition='outside'))
            fig.add_trace(go.Bar(x=['share EGID in topo',], y=[gwr_select_but_not_in_topo['EGID'].nunique()/gwr_mrg_all_building_in_bfs['EGID'].nunique(),],
                                    name=f'gwrEGID_in_sample ({gwr_select_but_not_in_topo["EGID"].nunique()} nr in sample by gwr selection criteria)',
                                    text=f'{gwr_select_but_not_in_topo["EGID"].nunique()/gwr_mrg_all_building_in_bfs["EGID"].nunique():.2f} ({gwr_select_but_not_in_topo["EGID"].nunique()} nEGIDs)',  # Add text to display the count
                                    textposition='outside'))
            fig.add_trace(go.Bar(x=['share EGID in topo',], y=[gwr_rest['EGID'].nunique()/gwr_mrg_all_building_in_bfs['EGID'].nunique(),],
                                name=f'gwrEGID_not_in_sample ({gwr_mrg_all_building_in_bfs["EGID"].nunique()} nr bldngs in bfs region)',
                                text=f'{gwr_rest["EGID"].nunique()/gwr_mrg_all_building_in_bfs["EGID"].nunique():.2f} ({gwr_rest["EGID"].nunique()}, total {gwr_mrg_all_building_in_bfs["EGID"].nunique()} nEGIDs)',  # Add text to display the count
                                textposition='outside'))
            fig.add_trace(go.Scatter(x=[col], y=[0], name='share EGID in topo', opacity=0,))  

            fig.update_layout(  
                barmode='stack',
                xaxis_title='Characteristics',
                yaxis_title='Frequency',
                title = f'Characteristics of omitted GWR EGIDs (scen: {scen})'
            )

            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_pie_disc_charac_omitted_gwr.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_pie_disc_charac_omitted_gwr.html')
                print_to_logfile(f'\texport: plot_ind_pie_disc_charac_omitted_gwr.png (for: {scen})', self.visual_sett.log_name)



            # plot continuous characteristics -----
            cont_cols = plot_ind_charac_omitted_gwr_specs['cont_cols']
            ncols = 2
            nrows = int(np.ceil(len(cont_cols) / ncols))

            fig = make_subplots(rows = nrows, cols = ncols)

            i, col = 0, cont_cols[1]
            for i, col in enumerate(cont_cols):
                if col in omitt_gwregid_from_topo.columns:
                    omitt_gwregid_from_topo[col].value_counts()
                    col_df  = omitt_gwregid_from_topo[col].replace('', np.nan).dropna().astype(float)
                    # if col in ['GBAUJ', 'GBAUM']:
                        # col_df.sort_values(inplace=True)
                    fig.add_trace(go.Histogram(x=col_df, name=col), row = int(i / ncols) + 1, col = i % ncols + 1)
                    fig.update_xaxes(title_text=col, row = int(i / ncols) + 1, col = i % ncols + 1)
                    fig.update_yaxes(title_text='Frequency', row = int(i / ncols) + 1, col = i % ncols + 1)
            fig.update_layout(
                title = f'Continuous Characteristics of omitted GWR EGIDs (scen: {scen})'
            )

            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_hist_cont_charac_omitted_gwr.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_hist_cont_charac_omitted_gwr.html')
                print_to_logfile(f'\texport: plot_ind_hist_cont_charac_omitted_gwr.png (for: {scen})', self.visual_sett.log_name)
