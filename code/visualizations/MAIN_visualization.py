import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import itertools
import glob
import datetime as datetime
import shutil
import fnmatch
import pickle
import copy
from scipy.stats import gaussian_kde
from scipy.stats import skewnorm
from itertools import chain


from typing_extensions import List, Dict

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code.auxiliary_functions import get_bfsnr_name_tuple_list, chapter_to_logfile, checkpoint_to_logfile, print_to_logfile

from code.MAIN_pvallocation import PVAllocScenario

# ------------------------
# *** PV VISUALIZATION ***
# ------------------------


class VisualSetting:
    def __init__(self, 
                pvalloc_scen_list : List[str] = [],
                pvalloc_exclude_pattern_list : List[str] = ['*.txt',
                                                            '*old_vers*',
                                                            ],
                plot_show: bool = True,
                save_plot_by_scen_directory: bool = True,
                remove_old_plot_scen_directories: bool = False,
                remove_old_plots_in_visualization: bool = False,
                MC_subdir_for_plot: str = '*MC*1',
                mc_plots_individual_traces: bool = True,

                default_zoom_year: List[int] = [2002, 2030],
                default_zoom_hour: List[int] = [2400, 2400+(24*7)],
                default_map_zoom: int = 11,
                default_map_center: List[float] = [47.48, 7.57],
                node_selection_for_plots: List[str] = ['1', '3', '5'],

                # PLOT CHUCK --------------------------------------->  [run plot,  show plot,  show all scen]
                # for pvalloc_inital + sanitycheck
                plot_ind_var_summary_stats_TF: List[bool] =               [True, True, False],
                plot_ind_hist_pvcapaprod_sanitycheck_TF: List[bool] =     [True, True, False],
                plot_ind_boxp_radiation_rng_sanitycheck_TF: List[bool] =  [True, True, False],
                plot_ind_hist_pvprod_deviation_TF: List[bool] =           [True, True, False],
                plot_ind_charac_omitted_gwr_TF: List[bool] =              [True, True, False],
                plot_ind_line_meteo_radiation_TF: List[bool] =            [True, True, False],

                plot_ind_hist_pvcapaprod_sanitycheck_specs: Dict = {
                    'xbins_hist_instcapa_abs': 0.5,
                    'xbins_hist_instcapa_stand': 0.1,
                    'xbins_hist_totalprodkwh_abs': 500, 
                    'xbins_hist_totalprodkwh_stand': 0.05,
                    'trace_color_palettes': ['Turbo', 'Viridis', 'Aggrnyl', 'Agsunset'],    #  ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'Mint', 'solar', 'Teal', 'Magenta', 'Plotly3', 'Viridis', 'Turbo', 'Blackbody']
                    'trace_colval_max': 60,                            # max value for color scale; the higher the max value and the lower the increments, the more colors will be picked within the same color range of the palette
                    'trace_colincr': 10,                                # increment for color scale
                    'uniform_scencolor_and_KDE_TF': True,
                    'export_spatial_data_for_prod0': True, 
                    },
                plot_ind_charac_omitted_gwr_specs: Dict = {
                    'disc_cols': ['BFS_NUMMER','GSTAT','GKAT','GKLAS'], 
                    'disc_ncols': 2, 
                    'disc_figsize': [15, 10],
                    'cont_cols': ['GBAUJ','GBAUM','GAREA','GEBF','WAZIM','WAREA'],
                    'cont_ncols': 3,
                    'cont_figsize': [15, 10],
                    'cont_bins': 20,
                    'gwr_code_name_tuples_GKLAS': [
                        ('1110', 'Bldg. w one flat (incl double, row houses, w indiv roofs)'),
                        ('1121', 'Bldg. w two flat (incl double, row houses, w 2 flats'),
                        ('1276', 'Bldg. for animal shelter'), ],
                    'gwr_code_name_tuples_GSTAT': [
                        ('1004', 'Existing bldg.'),]
                },

                # for pvalloc_MC_algorithm
                plot_ind_line_installedCap_TF: List[bool] =            [True,    True,       False],
                plot_ind_line_PVproduction_TF: List[bool] =            [True,    True,       False],
                plot_ind_line_productionHOY_per_node_TF: List[bool] =  [True,    True,       False],
                plot_ind_line_gridPremiumHOY_per_node_TF: List[bool] = [True,    True,       False],
                plot_ind_line_gridPremium_structure_TF: List[bool] =   [True,    True,       False],
                plot_ind_hist_NPV_freepartitions_TF: List[bool] =      [True,    True,       False],
                plot_ind_hist_pvcapaprod_TF: List[bool] =              [True,    True,       False],
                plot_ind_map_topo_egid_TF: List[bool] =                [True,    False,      False],
                plot_ind_map_node_connections_TF: List[bool] =         [True,    False,      False],
                plot_ind_map_omitted_egids_TF: List[bool] =            [True,    True,       False],
                plot_ind_lineband_contcharact_newinst_TF: List[bool] = [True,    True,       False],

                plot_ind_map_topo_egid_specs: Dict = {
                    'uniform_municip_color': '#fff2ae',
                    'shape_opacity': 0.2,
                    'point_opacity': 0.7,
                    'point_opacity_sanity_check': 0.4,
                    'point_size_pv': 6,
                    'point_size_rest': 4.5,
                    'point_size_sanity_check': 20,
                    'point_color_pv_df': '#54f533',      # green
                    'point_color_solkat': '#f06a1d',     # turquoise
                    'point_color_alloc_algo': '#ffa600', # yellow 
                    'point_color_rest': '#383838',       # dark grey
                    'point_color_sanity_check': '#0041c2', # blue
                }, 
                plot_ind_map_node_connections_specs: Dict = {
                    'uniform_municip_color': '#fff2ae',
                    'shape_opacity': 0.2,   
                    'point_opacity_all': 0.5,
                    'point_size_all': 4,
                    'point_opacity_bynode': 0.7,
                    'point_size_bynode': 6,
                    'point_color_all': '#383838',       # dark grey
                    'point_color_palette': 'Turbo',
                    'point_size_dsonode_loc': 15,
                    'point_opacity_dsonode_loc': 1
                },
                plot_ind_map_omitted_egids_specs: Dict = {
                    'point_opacity': 0.7,
                    'point_size_select_but_omitted': 10,
                    'point_size_rest_not_selected': 1, # 4.5,
                    'point_color_select_but_omitted': '#ed4242', # red
                    'point_color_rest_not_selected': '#ff78ef',  # pink
                    'export_gdfs_to_shp': True, 
                }, 
                plot_ind_line_contcharact_newinst_specs: Dict =  {
                        'trace_color_palette': 'Turbo',
                        'upper_lower_bound_interval': [0.05, 0.95],
                        'colnames_cont_charact_installations_AND_numerator': 
                        [('pv_tarif_Rp_kWh',        1), 
                        ('elecpri_Rp_kWh',         1),
                        ('selfconsum_kW',          1),
                        ('FLAECHE',                1), 
                        ('netdemand_kW',           1000), 
                        ('estim_pvinstcost_chf',   1000),
                        ('TotalPower',             1),
                        ], 
                        },

                # for aggregated MC_algorithms
                plot_mc_line_PVproduction: List[bool] = [False,    True,       False],
                ):
        self.pvalloc_scen_list : List[str] = pvalloc_scen_list
        self.pvalloc_exclude_pattern_list : List[str] = pvalloc_exclude_pattern_list

        self.plot_show : bool = plot_show
        # self.remove_previous_plots: bool = remove_previous_plots
        self.save_plot_by_scen_directory: bool = save_plot_by_scen_directory
        self.remove_old_plot_scen_directories: bool = remove_old_plot_scen_directories
        self.remove_old_plots_in_visualization: bool = remove_old_plots_in_visualization
        self.MC_subdir_for_plot: str = MC_subdir_for_plot
        self.mc_plots_individual_traces: bool = mc_plots_individual_traces

        self.default_zoom_year: List[int] = default_zoom_year
        self.default_zoom_hour: List[int] = default_zoom_hour
        self.default_map_zoom: int = default_map_zoom
        self.default_map_center: List[float] = default_map_center
        self.node_selection_for_plots: List[str] = node_selection_for_plots


        self.plot_ind_var_summary_stats_TF: List[bool] = plot_ind_var_summary_stats_TF
        self.plot_ind_hist_pvcapaprod_sanitycheck_TF: List[bool] = plot_ind_hist_pvcapaprod_sanitycheck_TF
        self.plot_ind_boxp_radiation_rng_sanitycheck_TF: List[bool] = plot_ind_boxp_radiation_rng_sanitycheck_TF
        self.plot_ind_hist_pvprod_deviation_TF: List[bool] = plot_ind_hist_pvprod_deviation_TF
        self.plot_ind_charac_omitted_gwr_TF: List[bool] = plot_ind_charac_omitted_gwr_TF
        self.plot_ind_line_meteo_radiation_TF: List[bool] = plot_ind_line_meteo_radiation_TF

        self.plot_ind_hist_pvcapaprod_sanitycheck_specs: Dict = plot_ind_hist_pvcapaprod_sanitycheck_specs
        self.plot_ind_charac_omitted_gwr_specs: Dict = plot_ind_charac_omitted_gwr_specs
        
        self.plot_ind_line_installedCap_TF: List[bool] = plot_ind_line_installedCap_TF
        self.plot_ind_line_PVproduction_TF: List[bool] = plot_ind_line_PVproduction_TF
        self.plot_ind_line_productionHOY_per_node_TF: List[bool] = plot_ind_line_productionHOY_per_node_TF
        self.plot_ind_line_gridPremiumHOY_per_node_TF: List[bool] = plot_ind_line_gridPremiumHOY_per_node_TF
        self.plot_ind_line_gridPremium_structure_TF: List[bool] = plot_ind_line_gridPremium_structure_TF
        self.plot_ind_hist_NPV_freepartitions_TF: List[bool] = plot_ind_hist_NPV_freepartitions_TF
        self.plot_ind_hist_pvcapaprod_TF: List[bool] = plot_ind_hist_pvcapaprod_TF
        self.plot_ind_map_topo_egid_TF: List[bool] = plot_ind_map_topo_egid_TF
        self.plot_ind_map_node_connections_TF: List[bool] = plot_ind_map_node_connections_TF
        self.plot_ind_map_omitted_egids_TF: List[bool] = plot_ind_map_omitted_egids_TF
        self.plot_ind_lineband_contcharact_newinst_TF: List[bool] = plot_ind_lineband_contcharact_newinst_TF

        self.plot_ind_map_topo_egid_specs: Dict = plot_ind_map_topo_egid_specs
        self.plot_ind_map_node_connections_specs: Dict = plot_ind_map_node_connections_specs
        self.plot_ind_map_omitted_egids_specs: Dict = plot_ind_map_omitted_egids_specs
        self.plot_ind_line_contcharact_newinst_specs: Dict = plot_ind_line_contcharact_newinst_specs

        self.plot_mc_line_PVproduction: List[bool] = plot_mc_line_PVproduction
        
        # self.setup_visualization()


        # def setup_visualization(self, ):
        # SETUP --------------------
        self.wd_path = os.getcwd()
        self.data_path = os.path.join(self.wd_path, 'data')
        self.visual_path = os.path.join(self.data_path, 'visualization')
        self.log_name = f'{self.visual_path}/visual_log.txt'

        os.makedirs(self.visual_path, exist_ok=True)

        # create a str list of scenarios in pvalloc to visualize (exclude by pattern recognition)
        scen_in_pvalloc_list = os.listdir(f'{self.data_path}/pvalloc')
        self.pvalloc_scen_list: list[str] = [
            scen for scen in scen_in_pvalloc_list
            if not any(fnmatch.fnmatch(scen, pattern) for pattern in self.pvalloc_exclude_pattern_list)
        ]     
        
        # create new visual directories per scenario (+ remove old ones)
        for scen in self.pvalloc_scen_list:
            visual_scen_path = f'{self.visual_path}/{scen}'
            if os.path.exists(visual_scen_path):
                n_same_names = len(glob.glob(f'{visual_scen_path}*/'))
                old_dir_rename = f'{visual_scen_path}_{n_same_names}_old_vers'
                os.rename(visual_scen_path, old_dir_rename)

            os.makedirs(visual_scen_path) if self.save_plot_by_scen_directory else None

        if self.remove_old_plot_scen_directories:
            old_plot_scen_dirs = glob.glob(f'{self.visual_path}/*old_vers')
            for dir in old_plot_scen_dirs:
                try:    
                    shutil.rmtree(dir)
                except Exception as e:
                    print(f'Could not remove {dir}: {e}')

        if self.remove_old_plots_in_visualization: 
            old_plots = glob.glob(f'{self.visual_path}/*.html')
            for file in old_plots:
                os.remove(file)


        chapter_to_logfile('start MASTER_visualization\n', self.log_name, overwrite_file=True)
        print('end_setup')


    # ------------------------------------------------------------------------------------------------------
    # PLOT-AUXILIARY FUNCTIONS
    # ------------------------------------------------------------------------------------------------------

    def get_pvallocscen_pickle_IN_SCEN_output(self,pvalloc_scen_name):
        pickle_path = glob.glob(f'{self.data_path}/pvalloc/{pvalloc_scen_name}/*.pkl')[0]
        with open(pickle_path, 'rb') as f:
            pvalloc_scen = pickle.load(f)
            
        self.pvalloc_scen = pvalloc_scen

    def add_scen_name_to_plot(self, fig_func, scen, pvalloc_scen):
        # add scenario name
        fig_func.add_annotation(
            text=f'Scen: {scen}, (start T0: {self.pvalloc_scen.T0_prediction.split(" ")[0]}, {self.pvalloc_scen.months_prediction} prediction months)',
            xref="paper", yref="paper",
            x=0.5, y=1.05, showarrow=False,
            font=dict(size=12)
        )
        return fig_func
    
    def set_default_fig_zoom_year(self, fig, zoom_window, df, datecol):
        start_zoom = pd.to_datetime(f'{zoom_window[0]}-01-01')
        max_date = df[datecol].max() + pd.DateOffset(years=1)
        if pd.to_datetime(f'{zoom_window[1]}-01-01') > max_date:
            end_zoom = max_date
        else:
            end_zoom = pd.to_datetime(f'{zoom_window[1]}-01-01')
        fig.update_layout(
            xaxis = dict(range=[start_zoom, end_zoom])
        )
        return fig 

    def set_default_fig_zoom_hour(self, fig, zoom_window):
        start_zoom, end_zoom = zoom_window[0], zoom_window[1]
        fig.update_layout(
            xaxis_range=[start_zoom, end_zoom])
        return fig




    # ------------------------------------------------------------------------------------------------------------------------
    # ALL AVAILABLE PLOTS 
    # ------------------------------------------------------------------------------------------------------------------------

    # PLOT IND SCEN: pvalloc_initalization + sanitycheck ----------------------------------------
    def plot_ALL_init_sanitycheck(self, ):
        self.plot_ind_var_summary_stats()
        self.plot_ind_hist_pvcapaprod_sanitycheck()
        self.plot_ind_boxp_radiation_rng_sanitycheck()
        self.plot_ind_charac_omitted_gwr()
        self.plot_ind_line_meteo_radiation()

    if True: 
        def plot_ind_var_summary_stats(self, ):
            if self.plot_ind_var_summary_stats_TF[0]:

                checkpoint_to_logfile(f'plot_ind_var_summary_stats', self.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    # total kWh by demandtypes ------------------------
                    demandtypes = pd.read_parquet(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/demandtypes.parquet')
                    
                    demandtypes_names = [col for col in demandtypes.columns if 't' not in col]
                    totaldemand_kWh = [demandtypes[type].sum() for type in demandtypes_names]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=demandtypes_names, y=totaldemand_kWh, name='Total Demand [kWh]'))
                    fig.update_layout(
                        xaxis_title='Demand Type',
                        yaxis_title='Total Demand [kWh], 1 year',
                        title = f'Total Demand per Demand Type (scen: {scen})'
                    )
                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)

                    if self.plot_show and self.plot_ind_var_summary_stats_TF[1]:
                        if self.plot_ind_var_summary_stats_TF[2]:
                            fig.show()
                        elif not self.plot_ind_var_summary_stats_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_bar_totaldemand_by_type.html')
                    else:
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_bar_totaldemand_by_type.html')
                    print_to_logfile(f'\texport: plot_ind_bar_totaldemand_by_type.html (for: {scen})', self.log_name)



        def plot_ind_hist_pvcapaprod_sanitycheck(self,):
            if self.plot_ind_hist_pvcapaprod_sanitycheck_TF[0]:

                checkpoint_to_logfile(f'plot_ind_hist_pvcapaprod_sanitycheck', self.log_name)
                    
                # available color palettes
                trace_color_dict = {
                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
                    'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
                    'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                }        
                
                # visual settings
                plot_ind_hist_pvcapaprod_sanitycheck_specs= self.plot_ind_hist_pvcapaprod_sanitycheck_specs
                xbins_hist_instcapa_abs, xbins_hist_instcapa_stand = plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_instcapa_abs'], plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_instcapa_stand']
                xbins_hist_totalprodkwh_abs, xbins_hist_totalprodkwh_stand = plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_totalprodkwh_abs'], plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_totalprodkwh_stand']
                
                trace_colval_max, trace_colincr, uniform_scencolor_and_KDE_TF = plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_colval_max'], plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_colincr'], plot_ind_hist_pvcapaprod_sanitycheck_specs['uniform_scencolor_and_KDE_TF']
                trace_color_palettes = plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_color_palettes']
                trace_color_palettes_list= [trace_color_dict[color] for color in trace_color_palettes]

                color_pv_df, color_solkat, color_rest = self.plot_ind_map_topo_egid_specs['point_color_pv_df'], self.plot_ind_map_topo_egid_specs['point_color_solkat'], self.plot_ind_map_topo_egid_specs['point_color_rest']
                
                # switch to rerun plot for uniform_scencolor_and_KDE_TF if turned on 
                if uniform_scencolor_and_KDE_TF:
                    uniform_scencolor_and_KDE_TF_list = [True, False]
                elif not uniform_scencolor_and_KDE_TF:
                    uniform_scencolor_and_KDE_TF_list = [False,]

                export_subdf_egid_counter = 0
                # plot --------------------
                for uniform_scencolor_and_KDE_TF in uniform_scencolor_and_KDE_TF_list:
                    fig_agg_abs, fig_agg_stand = go.Figure(), go.Figure()

                    i_scen, scen = 0, self.pvalloc_scen_list[0]
                    for i_scen, scen in enumerate(self.pvalloc_scen_list):
                        
                        self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)
                        panel_efficiency_print = 'dynamic' if self.pvalloc_scen.PEFspec_variable_panel_efficiency_TF else 'static'

                        # data import
                        self.sanity_scen_data_path = f'{self.data_path}/pvalloc/{scen}/sanity_check_byEGID'
                        pv = pd.read_parquet(f'{self.data_path}/pvalloc/{scen}/pv.parquet')
                        topo = json.load(open(f'{self.sanity_scen_data_path}/topo_egid.json', 'r'))
                        gwr_gdf = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')

                        egid_with_pvdf = [egid for egid in topo.keys() if topo[egid]['pv_inst']['info_source'] == 'pv_df']
                        xtf_in_topo = [topo[egid]['pv_inst']['xtf_id'] for egid in egid_with_pvdf]
                        topo_subdf_paths = glob.glob(f'{self.sanity_scen_data_path}/topo_subdf_*.parquet')
                        topo.get(egid_with_pvdf[0])
                        
                        aggdf_combo_list = []
                        path = topo_subdf_paths[0]
                        for i_path, path in enumerate(topo_subdf_paths):
                            subdf= pd.read_parquet(path)
                            # subdf = subdf.loc[subdf['EGID'].isin(egid_with_pvdf)]
                            # > compute pvprod by using TotalPower of pv_df, check if it overlaps with computation of STROMERTRAG
                            # > problem with multiple df_uids per EGID, pvprod_TotalPower will be multiply counted. Only assign  pvprod_TotalPower 
                            # > to first df_uid of EGID. 

                            # subdf['pvprod_TotalPower_kW'] = subdf['radiation_rel_locmax'] * subdf['TotalPower'] *  self.pvalloc_scen.TECspec_inverter_efficiency * self.pvalloc_scen.TECspec_share_roof_area_available * subdf['panel_efficiency'] 
                            # subdf.loc[:, 'pvprod_TotalPower_kW'] = subdf['radiation_rel_locmax'] * subdf['TotalPower'] * self.pvalloc_scen.TECspec_inverter_efficiency * self.pvalloc_scen.TECspec_share_roof_area_available * subdf['panel_efficiency']
                            # subdf.loc[subdf['info_source'] != 'pv_df', 'pvprod_TotalPower_kW'] = np.nan
                            subdf['first_dfuid'] = subdf.groupby('EGID')['df_uid'].transform(lambda x: x == x.iloc[0])
                            subdf['pvprod_TotalPower_kW'] = np.where(
                                subdf['first_dfuid'],
                                subdf['radiation_rel_locmax'] * subdf['TotalPower'] * self.pvalloc_scen.TECspec_inverter_efficiency * self.pvalloc_scen.TECspec_share_roof_area_available * subdf['panel_efficiency'],
                                np.nan
                            )

                            agg_subdf = subdf.groupby(['EGID', 'df_uid', 'first_dfuid', 'FLAECHE', 'STROMERTRAG']).agg({'pvprod_kW': 'sum', 
                                                                                                        'pvprod_TotalPower_kW': 'sum'}).reset_index()
                            aggsub_npry = np.array(agg_subdf)
                            
                            egid_list, dfuid_list, flaeche_list, pvprod_list, pvprod_ByTotalPower_list, stromertrag_list, first_dfuid_list = [], [], [], [], [], [], []
                            egid = subdf['EGID'].unique()[0]

                            for egid in subdf['EGID'].unique():
                                mask_egid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('EGID')], egid)
                                df_uids  = list(aggsub_npry[mask_egid_subdf, agg_subdf.columns.get_loc('df_uid')])
                                
                                for r in range(1,len(df_uids)+1):
                                    for combo in itertools.combinations(df_uids,r):
                                        combo_key_str = '_'.join([str(c) for c in combo])
                                        mask_dfuid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))

                                        egid_list.append(egid)
                                        dfuid_list.append(combo_key_str)
                                        flaeche_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECHE')].sum())
                                        pvprod_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_kW')].sum())
                                        pvprod_ByTotalPower_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_TotalPower_kW')].sum())
                                        stromertrag_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('STROMERTRAG')].sum())

                                        # first_dfuid_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('first_dfuid')][0])
                                        # combinations can by definition not be the first df_uid
                                        if r == 1:
                                            first_dfuid_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('first_dfuid')][0])
                                        else:
                                            first_dfuid_list.append(False)
                                        
                            aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 
                                                        'FLAECHE': flaeche_list, 'pvprod_kW': pvprod_list, 
                                                        'pvprod_ByTotalPower_kW': pvprod_ByTotalPower_list,
                                                        'STROMERTRAG': stromertrag_list, 
                                                        'first_dfuid': first_dfuid_list})
                            
                            aggdf_combo_list.append(aggsubdf_combo)
                        
                        aggdf_combo = pd.concat(aggdf_combo_list, axis=0)


                        # installed Capapcity kW --------------------------------
                        if True:            
                            aggdf_combo['inst_capa_kW'] = aggdf_combo['FLAECHE'] * self.pvalloc_scen.TECspec_kWpeak_per_m2 * self.pvalloc_scen.TECspec_share_roof_area_available
                            aggdf_combo['inst_capa_kW_stand'] = (aggdf_combo['inst_capa_kW'] - aggdf_combo['inst_capa_kW'].mean()) / aggdf_combo['inst_capa_kW'].std()
                            pv['TotalPower_stand'] = (pv['TotalPower'] - pv['TotalPower'].mean()) / pv['TotalPower'].std()

                            fig = make_subplots(specs=[[{"secondary_y": True}]])

                            fig.add_trace(go.Histogram(x=aggdf_combo['inst_capa_kW'], 
                                                    name='Modeled Potential Capacity (all partition combos, EGIDs in topo, in pv_df)', 
                                                    opacity=0.5, marker_color = color_rest, 
                                                    xbins=dict(size=xbins_hist_instcapa_abs)), 
                                                    secondary_y=False)
                            fig.add_trace(go.Histogram(x=pv.loc[pv['xtf_id'].isin(xtf_in_topo), 'TotalPower'], 
                                                    name='Installed Capacity (pv_df in topo)', 
                                                    opacity=0.5, marker_color = color_pv_df, 
                                                    xbins=dict(size = xbins_hist_instcapa_abs)), 
                                                    secondary_y=False)

                            fig.add_trace(go.Histogram(x=aggdf_combo['inst_capa_kW_stand'], 
                                                    name='Modeled Potential Capacity (all partition combos, EGIDs in topo, in pv_df), standardized', 
                                                    opacity=0.5, marker_color = color_rest,
                                                    xbins=dict(size= xbins_hist_instcapa_stand)),
                                                    secondary_y=True)
                            fig.add_trace(go.Histogram(x=pv['TotalPower_stand'],
                                                        name='Installed Capacity (pv_df in topo), standardized',
                                                        opacity=0.5, marker_color = color_pv_df,
                                                        xbins=dict(size= xbins_hist_instcapa_stand)),
                                                        secondary_y=True)

                            fig.update_layout(
                                barmode='overlay', 
                                xaxis_title='Capacity [kW]',
                                yaxis_title='Frequency (Modelled Capacity, possible installations)',
                                title = f'SANITY CHECK: Modelled vs Installed Capacity (kWp_m2:{self.pvalloc_scen.TECspec_kWpeak_per_m2}, share roof: {self.pvalloc_scen.TECspec_share_roof_area_available})'
                            ) 
                            fig.update_yaxes(title_text="Frequency (standardized)", secondary_y=True)
                            fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)

                            if self.plot_show and self.plot_ind_hist_pvcapaprod_sanitycheck_TF[1]:
                                if self.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show()
                                elif not self.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show() if i_scen == 0 else None
                            if self.save_plot_by_scen_directory:
                                fig.write_html(f'{self.visual_path}/{scen}/{scen}__ind_hist_instCapa_kW.html')
                            else:
                                fig.write_html(f'{self.visual_path}/{scen}__ind_hist_instCapa_kW.html')    
                            print_to_logfile(f'\texport: plot_ind_hist_SanityCheck_instCapa_kW.html (for: {scen})', self.log_name)


                        # annual PV production kWh --------------------------------
                        if True:
                            # standardization for plot
                            aggdf_combo['pvprod_kW_stand'] =                (aggdf_combo['pvprod_kW'] - aggdf_combo['pvprod_kW'].mean())                            / aggdf_combo['pvprod_kW'].std() 
                            aggdf_combo['pvprod_ByTotalPower_kW_stand'] =   (aggdf_combo['pvprod_ByTotalPower_kW'] - aggdf_combo['pvprod_ByTotalPower_kW'].mean())  / aggdf_combo['pvprod_ByTotalPower_kW'].std()
                            aggdf_combo['STROMERTRAG_stand'] =              (aggdf_combo['STROMERTRAG'] - aggdf_combo['STROMERTRAG'].mean())                        / aggdf_combo['STROMERTRAG'].std()

                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            fig.add_trace(go.Histogram(x=aggdf_combo['pvprod_kW'], 
                                                    name='Modeled Potential Yearly Production (kWh)',
                                                    opacity=0.5, marker_color = color_rest, 
                                                    xbins = dict(size=xbins_hist_totalprodkwh_abs)), secondary_y=False)
                            fig.add_trace(go.Histogram(x=aggdf_combo['STROMERTRAG'], 
                                                    name='STROMERTRAG (solkat estimated production)',
                                                    opacity=0.5, marker_color = color_solkat, 
                                                    xbins = dict(size=xbins_hist_totalprodkwh_abs)), secondary_y=False)
                            fig.add_trace(go.Histogram(x=aggdf_combo.loc[aggdf_combo['first_dfuid'] == True, 'pvprod_ByTotalPower_kW'],
                                                        name='Yearly Prod. TotalPower (pvdf estimated production)', 
                                                        opacity=0.5, marker_color = color_pv_df,
                                                        xbins=dict(size=xbins_hist_totalprodkwh_abs)), secondary_y=False)

                            fig.add_trace(go.Histogram(x=aggdf_combo['pvprod_kW_stand'], 
                                                    name='Modeled Potential Yearly Production (kWh), standardized',
                                                    opacity=0.5, marker_color = color_rest,
                                                    xbins=dict(size=xbins_hist_totalprodkwh_stand)), secondary_y=True)
                            fig.add_trace(go.Histogram(x=aggdf_combo['STROMERTRAG_stand'],
                                                        name='STROMERTRAG (solkat estimated production), standardized',
                                                        opacity=0.5, marker_color = color_solkat,
                                                        xbins=dict(size=xbins_hist_totalprodkwh_stand)), secondary_y=True)
                            fig.add_trace(go.Histogram(x=aggdf_combo.loc[aggdf_combo['first_dfuid'] == True, 'pvprod_ByTotalPower_kW_stand'],
                                                        name='Yearly Prod. TotalPower (pvdf estimated production), standardized',
                                                        opacity=0.5, marker_color = color_pv_df,
                                                        xbins=dict(size=xbins_hist_totalprodkwh_stand)), secondary_y=True)
                            fig.update_layout(
                                barmode = 'overlay', 
                                xaxis_title='Production [kWh]',
                                yaxis_title='Frequency, absolute',
                                title = f'SanityCheck: Modeled vs Estimated Yearly PRODUCTION (kWp_m2:{self.pvalloc_scen.TECspec_kWpeak_per_m2}, share roof available: {self.pvalloc_scen.TECspec_share_roof_area_available}, {panel_efficiency_print} panel efficiency, inverter efficiency: {self.pvalloc_scen.TECspec_inverter_efficiency})'
                            )
                            fig.update_yaxes(title_text="Frequency (standardized)", secondary_y=True)
                            fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                            

                            if self.plot_show and self.plot_ind_hist_pvcapaprod_sanitycheck_TF[1]:
                                if self.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show()
                                elif not self.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show() if i_scen == 0 else None
                            if self.save_plot_by_scen_directory:
                                fig.write_html(f'{self.visual_path}/{scen}/{scen}__ind_hist_annualPVprod_kWh.html')
                            else:
                                fig.write_html(f'{self.visual_path}/{scen}__ind_hist_annualPVprod_kWh.html')
                            print_to_logfile(f'\texport: plot_ind_hist_SanityCheck_annualPVprod_kWh.html (for: {scen})', self.log_name)


                        # Functions for Agg traces plot  --------------------------------
                        if True:
                            palette = trace_color_palettes_list[i_scen % len(trace_color_palettes_list)]
                            trace_colpal = pc.sample_colorscale(palette, [i/(trace_colval_max-1) for i in range(trace_colval_max)])
                            trace_colval = trace_colval_max

                            # get color from a specific palette
                            def col_from_palette(trace_colpal, trace_colval):
                                trace_colval = max(0, min(trace_colval_max, trace_colval))
                                return trace_colpal[trace_colval-1]
                            
                            def update_trace_color(trace_colval, trace_colincr):
                                return trace_colval - trace_colincr
                            

                            # kernel density traces
                            def add_kde_gaussian_trace(fig, data, name, color, uniform_col_AND_KDE = uniform_scencolor_and_KDE_TF):
                                kde = gaussian_kde(data)
                                x_range = np.linspace(min(data), max(data), 1000)
                                if uniform_col_AND_KDE:
                                    fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', line = dict(color = color), 
                                                                name=f'{name}', yaxis= 'y2'))  
                                            
                            # histogram traces
                            def add_histogram_trace(fig, data, name, color, xbins, uniform_col_AND_KDE = uniform_scencolor_and_KDE_TF):
                                if uniform_col_AND_KDE:
                                    fig.add_trace(go.Histogram(x=data, name=name, opacity=0.5, xbins= dict(size=xbins), marker_color = color,))
                                else:
                                    fig.add_trace(go.Histogram(x=data, name=name, opacity=0.5, xbins= dict(size=xbins),))

                            def add_scen_title_traces(fig, scen):
                                fig.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
                                fig.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,))         


                        # Agg: Absolute Values --------------------------------   
                        if True:
                            add_scen_title_traces(fig_agg_abs, scen)

                            # inst capacity kW
                            add_histogram_trace(fig_agg_abs, aggdf_combo['inst_capa_kW'],
                                                f' - Modeled Potential Capacity (kW)',   
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_abs)
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            add_histogram_trace(fig_agg_abs, pv.loc[pv['xtf_id'].isin(xtf_in_topo), 'TotalPower'],
                                                f' - Installed Capacity (pv_df in topo, kW)',
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_abs)
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            # annual PV production kWh
                            add_histogram_trace(fig_agg_abs, aggdf_combo['pvprod_kW'], 
                                                f' - Modeled Potential Yearly Production (kWh)', 
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_abs)
                            add_kde_gaussian_trace(fig_agg_abs, aggdf_combo['pvprod_kW'],f'  KDE Modeled Potential Yearly Production (kWh)',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            add_histogram_trace(fig_agg_abs, aggdf_combo['STROMERTRAG'], 
                                                f' - STROMERTRAG (solkat estimated production)', 
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_abs)
                            add_kde_gaussian_trace(fig_agg_abs, aggdf_combo['STROMERTRAG'],f'  KDE STROMERTRAG (solkat estimated production)',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            add_histogram_trace(fig_agg_abs, aggdf_combo.loc[aggdf_combo['first_dfuid'] == True, 'pvprod_ByTotalPower_kW'],
                                                f' - Yearly Prod. TotalPower (pvdf estimated production)', 
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_abs)
                            add_kde_gaussian_trace(fig_agg_abs, aggdf_combo.loc[aggdf_combo['first_dfuid'] == True, 'pvprod_ByTotalPower_kW'],
                                                f'  KDE Yearly Prod. TotalPower (pvdf estimated production)',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr) 


                        # Agg: Standardized Values --------------------------------
                        if True:
                            trace_colval = trace_colval_max
                            add_scen_title_traces(fig_agg_stand, scen)  

                            # inst capacity kW
                            add_histogram_trace(fig_agg_stand, aggdf_combo['inst_capa_kW_stand'],
                                                f' - Modeled Potential Capacity (kW), standardized',   
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_stand)
                            add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['inst_capa_kW_stand'],f'  KDE Modeled Potential Capacity (kW)',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            add_histogram_trace(fig_agg_stand, pv['TotalPower_stand'],
                                                f' - Installed Capacity (pv_df in topo), standardized',
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_stand)
                            add_kde_gaussian_trace(fig_agg_stand, pv['TotalPower_stand'],f'  KDE Installed Capacity (pv_df in topo, kW)',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            # annual PV production kWh
                            add_histogram_trace(fig_agg_stand, aggdf_combo['pvprod_kW_stand'],
                                                f' - Modeled Potential Yearly Production (kWh), standardized', 
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_stand)
                            add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['pvprod_kW_stand'],f'  KDE Modeled Potential Yearly Production (kWh), standardized',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            add_histogram_trace(fig_agg_stand, aggdf_combo['STROMERTRAG_stand'],
                                                f' - STROMERTRAG (solkat estimated production), standardized', 
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_stand)
                            add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['STROMERTRAG_stand'],f'  KDE STROMERTRAG (solkat estimated production), standardized',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)

                            add_histogram_trace(fig_agg_stand, aggdf_combo.loc[aggdf_combo['first_dfuid'] == True, 'pvprod_ByTotalPower_kW_stand'],
                                                f' - Yearly Prod. TotalPower (pvdf estimated production), standardized', 
                                                col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_stand)
                            add_kde_gaussian_trace(fig_agg_stand, aggdf_combo.loc[aggdf_combo['first_dfuid'] == True, 'pvprod_ByTotalPower_kW_stand'],f'  KDE Yearly Prod. TotalPower (pvdf estimated production), standardized',col_from_palette(trace_colpal,trace_colval), )
                            trace_colval = update_trace_color(trace_colval, trace_colincr)


                    # Export Agg plots --------------------------------
                    if True:
                        fig_agg_abs.update_layout(
                            barmode='overlay',
                            xaxis_title='Capacity [kW]',
                            yaxis_title='Frequency (Modelled Capacity, possible installations)',
                            title = f'SANITY CHECK: Agg. Modelled vs Installed Cap. & Yearly Prod. ABSOLUTE (kWp_m2:{self.pvalloc_scen.TECspec_kWpeak_per_m2}, share roof available: {self.pvalloc_scen.TECspec_share_roof_area_available}, {panel_efficiency_print} panel eff, inverter eff: {self.pvalloc_scen.TECspec_inverter_efficiency})',
                            yaxis2 = dict(overlaying='y', side='right', title='')
                        )
                        fig_agg_stand.update_layout(
                            barmode='overlay',
                            xaxis_title='Production [kWh]',
                            yaxis_title='Frequency, absolute',
                            title = f'SANITY CHECK: Agg. Modelled vs Installed Cap. & Yearly Prod. STANDARDIZED (kWp_m2:{self.pvalloc_scen.TECspec_kWpeak_per_m2}, share roof available: {self.pvalloc_scen.TECspec_share_roof_area_available}, {panel_efficiency_print} panel eff, inverter eff: {self.pvalloc_scen.TECspec_inverter_efficiency})',
                            yaxis2 = dict(overlaying='y', side='right', title='', position=0.95, ),
                        )


                        if self.plot_show and self.plot_ind_hist_pvcapaprod_sanitycheck[1]:
                            fig_agg_abs.show()
                            fig_agg_stand.show()
                        fig_agg_abs.write_html(f'{self.visual_path}/plot_agg_hist_pvCapaProd_abs_values__{len(self.pvalloc_scen_list)}scen_KDE{uniform_scencolor_and_KDE_TF}.html')   
                        fig_agg_stand.write_html(f'{self.visual_path}/plot_agg_hist_pvCapaProd_stand_values__{len(self.pvalloc_scen_list)}scen_KDE{uniform_scencolor_and_KDE_TF}.html')
                        print_to_logfile(f'\texport: plot_agg_hist_SanityCheck_instCapa_kW.html ({len(self.pvalloc_scen_list)} scens, KDE: {uniform_scencolor_and_KDE_TF})', self.log_name)
            

                    # Export shapes with 0 kWh annual production --------------------
                    if plot_ind_hist_pvcapaprod_sanitycheck_specs['export_spatial_data_for_prod0']:
                        os.mkdir(f'{self.data_path}/pvalloc/{scen}/topo_spatial_data', exist_ok=True)

                        # EGID_no_prod = aggdf_combo.loc[aggdf_combo['pvprod_kW'] == 0, 'EGID'].unique()
                        aggdf_combo_noprod = aggdf_combo.loc[aggdf_combo['pvprod_kW'] == 0]

                        # match GWR geom to gdf
                        aggdf_noprod_gwrgeom_gdf = aggdf_combo_noprod.merge(gwr_gdf, on='EGID', how='left')
                        aggdf_noprod_gwrgeom_gdf = gpd.GeoDataFrame(aggdf_noprod_gwrgeom_gdf, geometry='geometry')
                        aggdf_noprod_gwrgeom_gdf.set_crs(epsg=2056, inplace=True)
                        aggdf_noprod_gwrgeom_gdf.to_file(f'{self.data_path}/pvalloc/{scen}/topo_spatial_data/aggdf_noprod_gwrgeom_gdf.geojson', driver='GeoJSON')
                        print_to_logfile(f'\texport: aggdf_noprod_gwrgeom_gdf.geojson (scen: {scen}) for sanity check', self.log_name)

                        # try to match solkat geom to gdf
                        solkat_gdf = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/solkat_gdf.geojson')
                        aggdf_noprod_solkatgeom_gdf = copy.deepcopy(aggdf_combo_noprod)
                        aggdf_noprod_solkatgeom_gdf['geometry'] = 'NA'
                        
                        # i, row = 0, aggdf_combo_noprod.iloc[0]
                        # i, row = 2, aggdf_combo_noprod.iloc[2]
                        if not aggdf_combo_noprod.empty:
                            for i, row in aggdf_combo_noprod.iterrows():
                                dfuid_row_list = row['df_uid'].split('_')
                                dfuid_row_solkatgeom = solkat_gdf.loc[solkat_gdf['DF_UID'].isin(dfuid_row_list)]
                                if dfuid_row_solkatgeom.shape[0] == 1:
                                    aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = dfuid_row_solkatgeom.iloc[0]['geometry']
                                elif dfuid_row_solkatgeom.shape[0] > 1:
                                    # aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = MultiPolygon([geom for geom in dfuid_row_solkatgeom['geometry']])
                                    aggdf_noprod_gwrgeom_gdf.loc[i, 'geometry'] = dfuid_row_solkatgeom.unary_union
                                elif len(dfuid_row_solkatgeom) == 0:
                                    aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = 'NA_dfuid_aggdf_combo_notin_solkat_gdf'

                        aggdf_noprod_solkatgeom_gdf.loc[aggdf_noprod_solkatgeom_gdf['geometry'] == 'NA', 'geometry'] = None
                        aggdf_noprod_solkatgeom_gdf = gpd.GeoDataFrame(aggdf_noprod_solkatgeom_gdf, geometry='geometry')
                        aggdf_noprod_solkatgeom_gdf.set_crs(epsg=2056, inplace=True)
                        aggdf_noprod_solkatgeom_gdf.to_file(f'{self.data_path}/pvalloc/{scen}/topo_spatial_data/aggdf_noprod_solkatgeom_gdf.geojson', driver='GeoJSON')
                        checkpoint_to_logfile(f'\texport: aggdf_noprod_solkatgeom_gdf.geojson (scen: {scen}) for sanity check', self.log_name)



        def plot_ind_boxp_radiation_rng_sanitycheck(self,): 
            if self.plot_ind_boxp_radiation_rng_sanitycheck_TF[0]:

                checkpoint_to_logfile(f'plot_ind_boxp_radiation_rng_sanitycheck', self.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)


                    kWpeak_per_m2, share_roof_area_available = self.pvalloc_scen.TECspec_kWpeak_per_m2, self.pvalloc_scen.TECspec_share_roof_area_available
                    inverter_efficiency = self.pvalloc_scen.TECspec_inverter_efficiency
                    panel_efficiency_print = 'dynamic' if self.pvalloc_scen.PEFspec_variable_panel_efficiency_TF else 'static'
                    
                    # data import
                    sanity_scen_data_path = f'{self.data_path}/pvalloc/{scen}/sanity_check_byEGID'
                    pv = pd.read_parquet(f'{self.data_path}/pvalloc/{scen}/pv.parquet')
                    topo = json.load(open(f'{sanity_scen_data_path}/topo_egid.json', 'r'))
                    gwr_gdf = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')

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
                                            egid_df.to_excel(f'{self.data_path}/output/{scen}/subdf_egid{egid}_neg_rad.xlsx')
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
                                                egid_df.to_excel(f'{self.data_path}/output/{scen}/subdf_egid{egid}_lrgthn1_rad_rel_locmax.xlsx')
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


                    if self.plot_show and self.plot_ind_boxp_radiation_rng_sanitycheck_TF[1]:
                        if self.plot_ind_boxp_radiation_rng_sanitycheck_TF[2]:
                            fig_onebox.show()
                        elif not self.plot_ind_boxp_radiation_rng_sanitycheck_TF[2]:
                            fig_onebox.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig_onebox.write_html(f'{self.visual_path}/{scen}/{scen}__ind_boxp_radiation_rng_sanitycheck.html')
                    else:
                        fig_onebox.write_html(f'{self.visual_path}/{scen}__ind_boxp_radiation_rng_sanitycheck.html')
                    print_to_logfile(f'\texport: plot_ind_boxp_radiation_rng_sanitycheck.html (for: {scen})', self.log_name)



        def plot_ind_charac_omitted_gwr(self, ): 
            if self.plot_ind_charac_omitted_gwr_TF[0]:
                plot_ind_charac_omitted_gwr_specs = self.plot_ind_charac_omitted_gwr_specs
                checkpoint_to_logfile(f'plot_ind_charac_omitted_gwr', self.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    # omitted egids from data prep -----      
                    gwr_mrg_all_building_in_bfs = pd.read_parquet(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_mrg_all_building_in_bfs.parquet')
                    gwr = pd.read_parquet(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr.parquet')
                    topo = json.load(open(f'{self.data_path}/pvalloc/{scen}/topo_egid.json', 'r'))
                    
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
                                    cat_label = f"{[x for x in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GKLAS'] if x[0] == cat]}"
                                else:   
                                    cat_label = cat
                            elif col == 'GSTAT':
                                if cat in [tpl[0] for tpl in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GSTAT']]:
                                    cat_label = f"{[x for x in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GSTAT'] if x[0] == cat]}"
                                else: 
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
                                    
                    if self.plot_show and self.plot_ind_charac_omitted_gwr_TF[1]:
                        if self.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show()
                        elif not self.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_pie_disc_charac_omitted_gwr.html')
                    else:
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_pie_disc_charac_omitted_gwr.html')
                    print_to_logfile(f'\texport: plot_ind_pie_disc_charac_omitted_gwr.png (for: {scen})', self.log_name)



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
                    
                    if self.plot_show and self.plot_ind_charac_omitted_gwr_TF[1]:
                        if self.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show()
                        elif not self.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_hist_cont_charac_omitted_gwr.html')
                    else:
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_hist_cont_charac_omitted_gwr.html')
                    print_to_logfile(f'\texport: plot_ind_hist_cont_charac_omitted_gwr.png (for: {scen})', self.log_name)
                


        def plot_ind_line_meteo_radiation(self,): 
            if self.plot_ind_line_meteo_radiation_TF[0]:

                checkpoint_to_logfile('plot_ind_line_meteo_radiation', self.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    meteo_col_dir_radiation = self.pvalloc_scen.WEAspec_meteo_col_dir_radiation
                    meteo_col_diff_radiation = self.pvalloc_scen.WEAspec_meteo_col_diff_radiation
                    meteo_col_temperature = self.pvalloc_scen.WEAspec_meteo_col_temperature

                    # import meteo data -----
                    meteo = pd.read_parquet(f'{self.data_path}/pvalloc/{scen}/meteo_ts.parquet')


                    # try to also get raw data to show how radidation is derived
                    try: 
                        meteo_raw = pd.read_parquet(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/meteo.parquet')
                        meteo_raw = meteo_raw.loc[meteo_raw['timestamp'].isin(meteo['timestamp'])]
                        meteo_raw[meteo_col_temperature] = meteo_raw[meteo_col_temperature].astype(float)
                    except Exception:
                        print('... no raw meteo data available')
                        
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    try:  # necessary to accomodate older code versions where radiation is not strictly split into direct and diffuse
                        fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo[['rad_direct', 'rad_diffuse']].sum(axis = 1), name='Radiation [W/m^2]'))
                    except Exception:
                        fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo['radiation'], name='Radiation [W/m^2]'))
                    fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo['temperature'], name='Temperature [°C]'), secondary_y=True)
                    
                    radiation_cols = [meteo_col_dir_radiation, meteo_col_diff_radiation]
                    try: 
                        for col in radiation_cols:
                            fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[col], name=f'Rad. raw data: {col}'))

                        fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[radiation_cols].sum(axis=1), name='Rad. raw data: sum of rad types'))
                        fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[meteo_col_temperature], name=f'Temp. raw data: {meteo_col_temperature}'), secondary_y=True)
                    except Exception:
                        pass

                    fig.update_layout(title_text = f'Meteo Data: Temperature and Radiation (if Direct & Diffuse. flat_diffuse_rad_factor: {self.pvalloc_scen.WEAspec_flat_diffuse_rad_factor})')
                    fig.update_xaxes(title_text='Time')
                    fig.update_yaxes(title_text='Radiation [W/m^2]', secondary_y=False)
                    fig.update_yaxes(title_text='Temperature [°C]', secondary_y=True)
                    
                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                    # fig = set_default_fig_zoom_hour(fig, default_zoom_hour)

                    if self.plot_show and self.plot_ind_line_meteo_radiation_TF[1]:
                        if self.plot_ind_line_meteo_radiation_TF[2]:
                            fig.show()
                        elif not self.plot_ind_line_meteo_radiation_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_line_meteo_radiation.html')
                    else:
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_line_meteo_radiation.html')
                    print_to_logfile(f'\texport: plot_ind_line_meteo_radiation.html (for: {scen})', self.log_name)




    # PLOT IND SCEN: pvalloc_MC_algorithm ----------------------------------------
    def plot_ALL_mcalgorithm(self,): 
        self.plot_ind_line_installedCap()
        self.plot_ind_line_PVproduction()
        self.plot_ind_line_productionHOY_per_node()
        self.plot_ind_hist_NPV_freepartitions()


    if True: 
        def plot_ind_line_installedCap(self, ): 
            if self.plot_ind_line_installedCap_TF[0]:

                checkpoint_to_logfile('plot_ind_line_installedCap', self.log_name)
                
                # available color palettes
                trace_color_dict = {
                    'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                    'Viridis': pc.sequential.Viridis, 'Plotly3': pc.sequential.Plotly3, 
                    'Turbo': pc.sequential.Turbo, 'solar': pc.sequential.solar, 
                    'RdBu': pc.diverging.RdBu, 'Rainbow': pc.sequential.Rainbow, 

                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Blackbody': pc.sequential.Blackbody, 
                }        

                fig_agg_pmonth = go.Figure()
                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.mc_data_path = glob.glob(f'{self.data_path}/pvalloc/{scen}/{self.MC_subdir_for_plot}')[0] 
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    topo = json.load(open(f'{self.mc_data_path}/topo_egid.json', 'r'))
                    Map_egid_pv = pd.read_parquet(f'{self.data_path}/pvalloc/{scen}/Map_egid_pv.parquet')

                    gm_shp = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson') 
                    pv_gdf = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/pv_gdf.geojson')
                    gwr_all_building_gdf = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_all_building_gdf.geojson')


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
    

                    gwr_buff_select_BUT_year = copy.deepcopy(gwr_all_building_gdf)
                    gwr_buff_select_BUT_year = gwr_buff_select_BUT_year.loc[(gwr_buff_select_BUT_year['GGDENR'].isin(self.pvalloc_scen.bfs_numbers)) &
                                                                            (gwr_buff_select_BUT_year['GSTAT'].isin(self.pvalloc_scen.GWRspec_GSTAT)) & 
                                                                            (gwr_buff_select_BUT_year['GKLAS'].isin(self.pvalloc_scen.GWRspec_GKLAS)) ]
                    # gwr_buff_select_BUT_year.set_crs("EPSG:32632", allow_override=True, inplace=True)
                    # gwr_buff_select_BUT_year['geometry'] = gwr_buff_select_BUT_year['geometry'].buffer(self.pvalloc_scen.SOLKAT_GWR_EGID_buffer_size)
                    gwr_buff_select_BUT_year = gwr_buff_select_BUT_year.to_crs(gm_shp.crs)
                    instcomp_df = instcomp_df.to_crs(gm_shp.crs)
                    
                    instcomp_mrg_df = gpd.sjoin(instcomp_df, gwr_buff_select_BUT_year, how='left', predicate='within')

                    instcomp_mrg_df = instcomp_mrg_df.merge(Map_egid_pv, on = 'xtf_id', how = 'inner')
                    instcomp_mrg_df = instcomp_mrg_df.loc[instcomp_mrg_df['TotalPower'] <= 30]




                    # plot ind - line: Installed Capacity per Month ===========================
                    if self.plot_ind_line_installedCap_TF[0]:  #['plot_ind_line_installedCap_per_month']:
                        checkpoint_to_logfile('plot_ind_line_installedCap_per_month', self.log_name)

                        capa_month_df = pvinst_df.copy()
                        capa_month_df['BeginOp_month'] = capa_month_df['BeginOp'].dt.to_period('M')
                        capa_month_df = capa_month_df.groupby(['BeginOp_month', 'info_source'])['TotalPower'].sum().reset_index().copy()
                        capa_month_df['BeginOp_month'] = capa_month_df['BeginOp_month'].dt.to_timestamp()
                        capa_month_built = capa_month_df.loc[capa_month_df['info_source'] == 'pv_df'].copy()
                        capa_month_predicted = capa_month_df.loc[capa_month_df['info_source'] == 'alloc_algorithm'].copy()

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

                        instcomp_month_df = copy.deepcopy(instcomp_df)
                        instcomp_month_df['BeginOp'] = pd.to_datetime(instcomp_month_df['BeginOp'], format='%Y-%m-%d')
                        instcomp_month_df['BeginOp_month'] = instcomp_month_df['BeginOp'].dt.to_period('M')
                        instcomp_month_df = instcomp_month_df.groupby(['BeginOp_month', ])['TotalPower'].sum().reset_index().copy()
                        instcomp_month_df['BeginOp_month'] = instcomp_month_df['BeginOp_month'].dt.to_timestamp()

                        instcomp_year_df = copy.deepcopy(instcomp_df)
                        instcomp_year_df['BeginOp'] = pd.to_datetime(instcomp_year_df['BeginOp'], format='%Y-%m-%d')
                        instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp'].dt.to_period('Y')
                        instcomp_year_df = instcomp_year_df.groupby(['BeginOp_year',])['TotalPower'].sum().reset_index().copy()
                        instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp_year'].dt.to_timestamp()
                        instcomp_year_df['Cumm_TotalPower'] = instcomp_year_df['TotalPower'].cumsum()
                        instcomp_year_df['growth_cumm_TotalPower'] = instcomp_year_df['Cumm_TotalPower'].diff() / instcomp_year_df['Cumm_TotalPower'].shift(1) 
                        instcomp_year_df[['Cumm_TotalPower', 'growth_cumm_TotalPower']] 
                        

                        # -- DEBUGGING ----------
                        capa_year_df.to_csv(f'{self.visual_path}/{scen}_capa_year_df.csv')
                        capa_cumm_year_df.to_csv(f'{self.visual_path}/{scen}_capa_cumm_year_df.csv')
                        instcomp_mrg_df.to_csv(f'{self.visual_path}/{scen}_instcomp_mrg_df.csv') 
                        # -- DEBUGGING ----------


                        # plot ----------------
                        fig1 = go.Figure()

                        fig1.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name='built (month)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'cornflowerblue'), name='predicted (month)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'navy'),name='built + predicted (month)', mode='lines+markers'))

                        fig1.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'], line = dict(color = 'lightgreen'), name='built (year)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'], line = dict(color = 'limegreen'), name='predicted (year)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'], line = dict(color = 'forestgreen'), name='built + predicted (year)', mode='lines+markers',))

                        fig1.add_trace(go.Scatter(x=capa_cumm_year_df['BeginOp_year'], y=capa_cumm_year_df['Cumm_TotalPower'], line = dict(color ='purple'), name='cumulative built (year)', mode='lines+markers'))

                        fig1.add_trace(go.Scatter(x=instcomp_month_df['BeginOp_month'], y=instcomp_month_df['TotalPower'], line = dict(color = 'orange'), name='pv_df (blw 30kwp month)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['TotalPower'], line = dict(color = 'darkorange'), name='pv_df (blw 30kwp year)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['Cumm_TotalPower'], line = dict(color = 'orangered'), name='pv_df (blw 30kwp cumulative)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['growth_cumm_TotalPower'], line = dict(color = 'orangered'), name='pv_df (blw 30kwp growth)', mode='lines+markers'))

                        fig1.update_layout(
                            xaxis_title='Time',
                            yaxis_title='Installed Capacity (kW)',
                            legend_title='Time steps',
                            title = f'Installed Capacity per Month (weather year: {self.pvalloc_scen.WEAspec_weather_year})'
                        )

                        # add T0 prediction
                        T0_prediction = self.pvalloc_scen.T0_prediction
                        date = '2008-01-01 00:00:00'
                        fig1.add_shape(
                            # Line Vertical
                            dict(
                                type="line",
                                x0=T0_prediction,
                                y0=0,
                                x1=T0_prediction,
                                y1=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),  # Dynamic height
                                line=dict(color="black", width=1, dash="dot"),
                            )
                        )
                        fig1.add_annotation(
                            x=  T0_prediction,
                            y=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),
                            text="T0 Prediction",
                            showarrow=False,
                            yshift=10
                        )

                        fig1 = self.add_scen_name_to_plot(fig1, scen, self.pvalloc_scen)
                        fig1 = self.set_default_fig_zoom_year(fig1, self.default_zoom_year, capa_year_df, 'BeginOp_year')

                        if self.plot_show and self.plot_ind_line_installedCap_TF[1]:
                            if self.plot_ind_line_installedCap_TF[2]:
                                fig1.show()
                            elif not self.plot_ind_line_installedCap_TF[2]:
                                fig1.show() if i_scen == 0 else None
                        if self.save_plot_by_scen_directory:
                            fig1.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_line_installedCap_per_month.html')
                        else:
                            fig1.write_html(f'{self.visual_path}/{scen}__plot_ind_line_installedCap_per_month.html')
                        print_to_logfile(f'\texport: plot_ind_line_installedCap_per_month.html (for: {scen})', self.log_name)                    
                        


                    # plot ind - line: Installed Capacity per BFS ===========================
                    if False: # self.plot_ind_line_installedCap_TF[0]:  #['plot_ind_line_installedCap_per_BFS']:
                        checkpoint_to_logfile(f'plot_ind_line_installedCap_per_BFS', self.log_name)
                        capa_bfs_df = pvinst_df.copy()
                        gm_gdf = gpd.read_file(f'{self.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
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
                        fig2 = self.set_default_fig_zoom_year(fig2, self.default_zoom_year, capa_bfs_year_df, 'BeginOp_year')
                        
                        if self.plot_show and self.plot_ind_line_installedCap_TF[1]:
                            if self.plot_ind_line_installedCap_TF[2]:
                                fig2.show()
                            elif not self.plot_ind_line_installedCap_TF[2]:
                                fig2.show() if i_scen == 0 else None
                        if self.save_plot_by_scen_directory:
                            fig2.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_line_installedCap_per_BFS.html')
                        else:
                            fig2.write_html(f'{self.visual_path}/{scen}__plot_ind_line_installedCap_per_BFS.html')
                        print_to_logfile(f'\texport: plot_ind_line_installedCap_per_BFS.html (for: {scen})', self.log_name)
                        

                    # plot add aggregated - line: Installed Capacity per Year ===========================
                    if self.plot_ind_line_installedCap_TF[0]:  #['plot_ind_line_installedCap_per_month']:
                    
                        color_allscen_list = [list(trace_color_dict.keys())[i_scen] for i_scen in range(len(self.pvalloc_scen_list))]
                        color_palette = trace_color_dict[list(trace_color_dict.keys())[i_scen]]

                        # fig_agg_pmonth.add_trace(go.Scatter(x=[0,], y=[0,],  name=f'',opacity=1, ))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],  name=f'',opacity=0, ))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'black'), name=f'{scen}',opacity=0, mode='lines+markers'))

                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'],          opacity = 0.75, line = dict(color = color_palette[0+0]),    name='-- built (month)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'],  opacity = 0.75, line = dict(color = color_palette[0+1]),    name='-- predicted (month)', mode='lines+markers'))
                        # fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],                opacity = 0.75, line = dict(color = color_palette[0+2]),    name='-- built + predicted (month)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'],             opacity = 0.75, line = dict(color = color_palette[0+2]),    name='-- built (year)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'],     opacity = 0.75, line = dict(color = color_palette[0+3]),    name='-- predicted (year)', mode='lines+markers'))
                        # fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'],                   opacity = 0.75, line = dict(color = color_palette[0+5]),    name='-- built + predicted (year)', mode='lines+markers',))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_cumm_year_df['BeginOp_year'], y=capa_cumm_year_df['Cumm_TotalPower'],    opacity = 0.75, line = dict(color = color_palette[0+4]),    name='-- cumulative built + pred (year)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=instcomp_month_df['BeginOp_month'], y=instcomp_month_df['TotalPower'],        opacity = 0.75, line = dict(color = color_palette[0+5]),    name='-- pv_df (blw 30kwp month)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['TotalPower'],           opacity = 0.75, line = dict(color = color_palette[0+6]),    name='-- pv_df (blw 30kwp year)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=instcomp_year_df['BeginOp_year'], y=instcomp_year_df['Cumm_TotalPower'],      opacity = 0.75, line = dict(color = color_palette[0+0]),    name='-- pv_df (blw 30kwp cumulative)', mode='lines+markers'))
                        
                        # export plot add aggregated - line: Installed Capacity per Year 
                        if i_scen == len(self.pvalloc_scen_list)-1:
                            fig_agg_pmonth.update_layout(
                            xaxis_title='Time',
                            yaxis_title='Installed Capacity (kW)',
                            legend_title='Time steps',
                            title = f'Installed Capacity per Month/Year, {len(self.pvalloc_scen_list)}scen (weather year: {self.pvalloc_scen.WEAspec_weather_year})'
                            )

                            # add T0 prediction
                            T0_prediction = self.pvalloc_scen.T0_prediction
                            date = '2008-01-01 00:00:00'
                            fig_agg_pmonth.add_shape(
                                # Line Vertical
                                dict(
                                    type="line",
                                    x0=T0_prediction,
                                    y0=0,
                                    x1=T0_prediction,
                                    y1=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),  # Dynamic height
                                    line=dict(color="black", width=1, dash="dot"),
                                )
                    )
                            fig_agg_pmonth.add_annotation(
                                x=  T0_prediction,
                                y=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),
                                text="T0 Prediction",
                                showarrow=False,
                                yshift=10
                            )

                            fig_agg_pmonth = self.set_default_fig_zoom_year(fig_agg_pmonth, self.default_zoom_year, capa_year_df, 'BeginOp_year')
                            
                            if self.plot_show and self.plot_ind_line_installedCap_TF[1]:
                                fig_agg_pmonth.show()

                            fig_agg_pmonth.write_html(f'{self.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html')
                            print_to_logfile(f'\texport: plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html', self.log_name)


        def plot_ind_line_PVproduction(self, ): 
            if self.plot_ind_line_PVproduction_TF[0]:

                checkpoint_to_logfile('plot_ind_line_PVproduction', self.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.mc_data_path = glob.glob(f'{self.data_path}/pvalloc/{scen}/{self.MC_subdir_for_plot}')[0]
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    topo = json.load(open(f'{self.mc_data_path}/topo_egid.json', 'r'))
                    topo_subdf_paths = glob.glob(f'{self.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
                    gridnode_df_paths = glob.glob(f'{self.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_*.parquet')
                    gridnode_df = pd.read_parquet(f'{self.mc_data_path}/gridnode_df.parquet')

                    # get installations of topo over time
                    egid_list, inst_TF_list, info_source_list, BeginOp_list, xtf_id_list, TotalPower_list, = [], [], [], [], [], []
                    k = list(topo.keys())[0]
                    for k, v in topo.items():
                        egid_list.append(k)
                        inst_TF_list.append(v['pv_inst']['inst_TF'])
                        info_source_list.append(v['pv_inst']['info_source'])
                        BeginOp_list.append(v['pv_inst']['BeginOp'])
                        xtf_id_list.append(v['pv_inst']['xtf_id'])
                        TotalPower_list.append(v['pv_inst']['TotalPower'])

                    pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                            'BeginOp': BeginOp_list, 'xtf_id': xtf_id_list, 'TotalPower': TotalPower_list,})
                    pvinst_df = pvinst_df.loc[pvinst_df['inst_TF'] == True]


                    pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')
                    pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
                    pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')


                    # attach annual production to each installation
                    pvinst_df['pvprod_kW'] = float(0)
                    aggdf_combo_list = []

                    for ipath, path in enumerate(topo_subdf_paths):
                        subdf = pd.read_parquet(path)
                        subdf = subdf.loc[subdf['EGID'].isin(pvinst_df['EGID'])]

                        agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG']).agg({'pvprod_kW': 'sum',}).reset_index() 
                        aggsub_npry = np.array(agg_subdf)


                        # attach production to each installation                
                        pvinst_egid_in_subdf = [egid for egid in pvinst_df['EGID'].unique() if egid in agg_subdf['EGID'].unique()]
                        egid = pvinst_egid_in_subdf[0]
                        for egid in pvinst_egid_in_subdf:
                            df_uid_combo = pvinst_df.loc[pvinst_df['EGID'] == egid]['xtf_id'].values[0].split('_')

                            if len(df_uid_combo) == 1:
                                pvinst_df.loc[pvinst_df['EGID'] == egid, 'pvprod_kW'] = agg_subdf.loc[agg_subdf['EGID'] == egid]['pvprod_kW'].values[0]
                            elif len(df_uid_combo) > 1:
                                pvinst_df.loc[pvinst_df['EGID'] == egid, 'pvprod_kW'] = agg_subdf.loc[agg_subdf['df_uid'].isin(df_uid_combo), 'pvprod_kW'].sum()
            

                    # aggregate pvinst_df to monthly values
                    prod_month_df = copy.deepcopy(pvinst_df)
                    prod_month_df['BeginOp_month'] = prod_month_df['BeginOp'].dt.to_period('M')
                    prod_month_df['BeginOp_month_str'] = prod_month_df['BeginOp_month'].astype(str)
                    prod_month_df['TotalPower_month'] = prod_month_df.groupby(['BeginOp_month'])['TotalPower'].transform('sum')
                    prod_month_df['pvprod_kW_month'] = prod_month_df.groupby(['BeginOp_month'])['pvprod_kW'].transform('sum')
                    prod_month_df['BeginOp_month'] = prod_month_df['BeginOp_month'].dt.to_timestamp()
                    prod_month_df.sort_values(by=['BeginOp_month'], inplace=True)


                    # aggregate gridnode_df to monthly values
                    BeginOp_month_list, feedin_all_list, feedin_taken_list, feedin_loss_list = [], [], [], []
                    
                    # ONLY KEEP THIS WHILE NOT ALL MONTHS ARE EXPORTED in PVALLOC
                    month_iters = [path.split('gridnode_df_')[1].split('.parquet')[0] for path in gridnode_df_paths]
                    gridnode_df_month_iters = [path.split('pred_gridprem_node_by_M\\')[1].split('.parquet')[0] for path in gridnode_df_paths]
                    prod_month_df = prod_month_df.loc[prod_month_df['BeginOp_month_str'].isin(month_iters)]

                    month = prod_month_df['BeginOp_month'].unique()[0]
                    for month in prod_month_df['BeginOp_month'].unique():
                        month_str = prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'BeginOp_month_str'].values[0]
                        grid_subdf = pd.read_parquet(f'{self.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_{month_str}.parquet')
                        
                        prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'feedin_kW'] = grid_subdf['feedin_kW'].sum()
                        prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'feedin_kW_taken'] = grid_subdf['feedin_kW_taken'].sum()
                        prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'feedin_kW_loss'] = grid_subdf['feedin_kW_loss'].sum()



                    # plot ----------------
                    # fig = go.Figure()
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['pvprod_kW_month'], name='EGID Prod kWh (total pvprod_kW)', ))
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['feedin_kW'], name='Grid feedin kWh (feedin_kwh)', ))
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['feedin_kW_taken'], name='Grid feedin take kWh (feedin_taken kWh)', ))
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['feedin_kW_loss'], name='Grid feedin loss kWh (feedin_loss kWh)', ))

                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['TotalPower_month'], name='Total installed capacity', line=dict(color='blue', width=2)), secondary_y=True)

                    fig.update_layout(
                        title=f'PV production per month',
                        xaxis_title='Month',
                        yaxis_title='Production [kW]',
                        yaxis2_title='Installed capacity [kW]',
                        legend_title='Legend',
                    )
                    fig.update_yaxes(title_text="Installed capacity [kW]", secondary_y=True)

                    if self.plot_show and self.plot_ind_line_PVproduction_TF[1]:
                        if self.plot_ind_line_PVproduction_TF[2]:
                            fig.show()
                        elif not self.plot_ind_line_PVproduction_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_line_PVproduction.html')
                    else:
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_line_PVproduction.html')
                    print_to_logfile(f'\texport: plot_ind_line_PVproduction.html (for: {scen})', self.log_name)


        def plot_ind_line_productionHOY_per_node(self, ): 
            if self.plot_ind_line_productionHOY_per_node_TF[0]:

                checkpoint_to_logfile('plot_ind_line_productionHOY_per_node', self.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import ----------
                    self.mc_data_path = glob.glob(f'{self.data_path}/pvalloc/{scen}/{self.MC_subdir_for_plot}')[0] # take first path if multiple apply, so code can still run properlyrly
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    self.node_selection

                    gridnode_df = pd.read_parquet(f'{self.mc_data_path}/gridnode_df.parquet')
                    gridnode_df['grid_node'].unique()
                    gridnode_df['t_int'] = gridnode_df['t'].str.extract(r't_(\d+)').astype(int)
                    gridnode_df.sort_values(by=['t_int'], inplace=True)

                    # plot ----------------
                    # unclear why if statement is necessary here? maybe older data versions featured col 'info_source'
                    if 'info_source' in gridnode_df.columns:
                        if isinstance(self.node_selection, list):
                            nodes = self.node_selection
                        elif self.node_selection == None:
                            nodes = gridnode_df['grid_node'].unique()
                            
                        pvsources = gridnode_df['info_source'].unique()
                        fig = go.Figure()

                        for node in nodes:
                            for source in pvsources:
                                if source != '':
                                # if True:
                                    filter_df = gridnode_df.loc[
                                        (gridnode_df['grid_node'] == node) & (gridnode_df['info_source'] == source)].copy()
                                    
                                    # fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['pvprod_kW'], name=f'Prod Node: {node}, Source: {source}'))
                                    fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW'], name=f'{node} - feedin (all),  Source: {source}'))
                                    fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_taken'], name= f'{node} - feedin_taken, Source: {source}'))
                                    fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_loss'], name=f'{node} - feedin_loss, Source: {source}'))

                        
                        # gridnode_total_df = gridnode_df.groupby(['t', 't_int'])['feedin_kW'].sum().reset_index()
                        gridnode_total_df = gridnode_df.groupby(['t', 't_int']).agg({'pvprod_kW': 'sum', 'feedin_kW': 'sum','feedin_kW_taken': 'sum','feedin_kW_loss': 'sum'}).reset_index()
                        gridnode_total_df.sort_values(by=['t_int'], inplace=True)
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['pvprod_kW'], name='Total production', line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW'], name='Total feedin', line=dict(color='black', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_taken'], name='Total feedin_taken', line=dict(color='green', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_loss'], name='Total feedin_loss', line=dict(color='red', width=2)))
                    
                    else:
                        if isinstance(self.node_selection, list):
                            nodes = self.node_selection
                        elif self.node_selection == None:
                            nodes = gridnode_df['grid_node'].unique()

                        fig = go.Figure()
                        for node in nodes:
                            filter_df = copy.deepcopy(gridnode_df.loc[gridnode_df['grid_node'] == node])
                            fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW'], name=f'{node} - feedin (all)'))
                            fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_taken'], name= f'{node} - feedin_taken'))
                            fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_loss'], name=f'{node} - feedin_loss'))

                        gridnode_total_df = gridnode_df.groupby(['t', 't_int']).agg({'pvprod_kW': 'sum', 'feedin_kW': 'sum','feedin_kW_taken': 'sum','feedin_kW_loss': 'sum'}).reset_index()
                        gridnode_total_df.sort_values(by=['t_int'], inplace=True)
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['pvprod_kW'], name='Total production', line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW'], name='Total feedin', line=dict(color='black', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_taken'], name='Total feedin_taken', line=dict(color='green', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_loss'], name='Total feedin_loss', line=dict(color='red', width=2)))
                                    

                    fig.update_layout(
                        xaxis_title='Hour of Year',
                        yaxis_title='Production / Feedin (kW)',
                        legend_title='Node ID',
                        title = f'Production per node (kW, weather year: {self.pvalloc_scen.WEAspec_weather_year}, self consum. rate: {self.pvalloc_scen.TECspec_self_consumption_ifapplicable})'
                    )

                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                    fig = self.set_default_fig_zoom_hour(fig, self.default_zoom_hour, gridnode_total_df, 't_int')

                    if self.plot_show and self.plot_ind_line_productionHOY_per_node_TF[1]:
                        if self.plot_ind_line_productionHOY_per_node_TF[2]:
                            fig.show()
                        elif not self.plot_ind_line_productionHOY_per_node_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_line_productionHOY_per_node.html')
                    else:   
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_line_productionHOY_per_node.html')
                    print_to_logfile(f'\texport: plot_ind_line_productionHOY_per_node.html (for: {scen})', self.log_name)


        def plot_ind_hist_NPV_freepartitions(self, ): 
            if self.plot_ind_hist_NPV_freepartitions_TF[0]:

                checkpoint_to_logfile('plot_ind_hist_NPV_freepartitions', self.log_name)

                fig_agg = go.Figure()

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    # setup + import ----------
                    self.mc_data_path = glob.glob(f'{self.data_path}/pvalloc/{scen}/{self.MC_subdir_for_plot}')[0]
                    self.get_pvallocscen_pickle_IN_SCEN_output(pvalloc_scen_name = scen)

                    
                    npv_df_paths = glob.glob(f'{self.mc_data_path}/pred_npv_inst_by_M/npv_df_*.parquet')
                    periods_list = [pd.to_datetime(path.split('npv_df_')[-1].split('.parquet')[0]) for path in npv_df_paths]
                    before_period, after_period = min(periods_list), max(periods_list)

                    npv_df_before = pd.read_parquet(f'{self.mc_data_path}/pred_npv_inst_by_M/npv_df_{before_period.to_period("M")}.parquet')
                    npv_df_after  = pd.read_parquet(f'{self.mc_data_path}/pred_npv_inst_by_M/npv_df_{after_period.to_period("M")}.parquet')

                    # plot ----------------
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name='Before Allocation Algorithm', opacity=0.5))
                    fig.add_trace(go.Histogram(x=npv_df_after['NPV_uid'], name='After Allocation Algorithm', opacity=0.5))

                    fig.update_layout(
                        xaxis_title=f'Net Present Value (NPV, interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr)',
                        yaxis_title='Frequency',
                        title = f'NPV Distribution of possible PV installations, first / last year (weather year: {self.pvalloc_scen.TECspec_weather_year})',
                        barmode = 'overlay')
                    fig.update_traces(bingroup=1, opacity=0.5)

                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                        
                    if self.plot_show and self.plot_ind_hist_NPV_freepartitions_TF[1]:
                        if self.plot_ind_hist_NPV_freepartitions_TF[2]:
                            fig.show()
                        elif not self.plot_ind_hist_NPV_freepartitions_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_path}/{scen}/{scen}__plot_ind_hist_NPV_freepartitions.html')
                    else:
                        fig.write_html(f'{self.visual_path}/{scen}__plot_ind_hist_NPV_freepartitions.html')
                        

                    # aggregate plot ----------------
                    fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
                    fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,)) 

                    fig_agg.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name=f'Before Allocation', opacity=0.7, xbins=dict(size=500)))
                    fig_agg.add_trace(go.Histogram(x=npv_df_after['NPV_uid'],  name=f'After Allocation',  opacity=0.7, xbins=dict(size=500)))

                fig_agg.update_layout(
                    xaxis_title=f'Net Present Value (NPV, interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr)',
                    yaxis_title='Frequency',
                    title = f'NPV Distribution of possible PV installations, first / last year ({len(self.pvalloc_scen_list)} scen, weather year: {self.pvalloc_scen.TECspec_weather_year})',
                    barmode = 'overlay')
                # fig_agg.update_traces(bingroup=1, opacity=0.75)

                if self.plot_show and self.plot_ind_hist_NPV_freepartitions_TF[1]:
                    fig_agg.show()
                fig_agg.write_html(f'{self.visual_path}/plot_agg_hist_NPV_freepartitions__{len(self.pvalloc_scen_list)}scen.html')

                






        # def plot_ind_line_gridPremiumHOY_per_node(self, ): 


        # def plot_ind_line_gridPremium_structure(self, ): 

        # def plot_ind_hist_NPV_freepartitions(self, ): 

        # def plot_ind_map_topo_egid(self, ): 

        # def plot_ind_map_node_connections(self, ): 

        # def plot_ind_map_omitted_egids(self, ): 

        # def plot_ind_lineband_contcharact_newinst(self, ): 




# -------------------------
# *** RUN VISUALIZATION ***
# -------------------------

if __name__ == '__main__':

    run_visualizations = VisualSetting(
        pvalloc_exclude_pattern_list = ['*.txt','*old_vers*', 
                                        'pvalloc_BLsml_40y_f1983_1mc_meth2.2_rnd',
                                        '*pvalloc_BLsml_20y*', 
                                        ], 
        save_plot_by_scen_directory        = False, 
        remove_old_plot_scen_directories   = True,  
        remove_old_plots_in_visualization = True,  )
    
    # run_visualizations.plot_ind_var_summary_stats()
    # run_visualizations.plot_ind_hist_pvcapaprod_sanitycheck() 
    # run_visualizations.plot_ind_boxp_radiation_rng_sanitycheck()
    # run_visualizations.plot_ind_charac_omitted_gwr()
    # run_visualizations.plot_ind_line_meteo_radiation()

    run_visualizations.plot_ind_line_installedCap()
    # run_visualizations.plot_ind_line_PVproduction()
    # run_visualizations.plot_ind_line_productionHOY_per_node()
    # run_visualizations.plot_ind_hist_NPV_freepartitions()



    print('end <if __main__> chunk')
            






class test_class:
    def __init__(self):
        self.a = 1
        self.b = 2

    def meth1(self,):
        print(f'meth1 - print a: {self.a}')

    def meth2(self,):
        print(f'meth2 - print a+b: {self.b + self.a}')

    def meth3(self,):

        self.meth1()
        self.meth2()

    def store_to_pickle(self,):
        with open('test_class.pkl', 'wb') as f:
            pickle.dump(self, f)

if False: 
    test = test_class()
    test.store_to_pickle()

    print('-- before load ---------')
    test_reload = pickle.load(open('test_class.pkl', 'rb'))
    print('-- after load ---------')
    test_reload.meth3()
              
