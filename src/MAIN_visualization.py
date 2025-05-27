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
import polars as pl
import copy
from shapely import union_all, Polygon, MultiPolygon, MultiPoint
from scipy.stats import gaussian_kde
from scipy.stats import skewnorm
from itertools import chain
from dataclasses import dataclass, field


from typing_extensions import List, Dict

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots


# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr, get_bfsnr_name_tuple_list
from src.MAIN_pvallocation import PVAllocScenario_Settings


@dataclass
class Visual_Settings: 
    pvalloc_scen_list : List[str]                = field(default_factory=lambda: [])
    pvalloc_exclude_pattern_list : List[str]     = field(default_factory=lambda: [
                                                    '*.txt',
                                                    '*old_vers*',
                                                    ]),
    pvalloc_include_pattern_list : List[str]      = field(default_factory=lambda: [])
    plot_show: bool                              = True
    save_plot_by_scen_directory: bool            = True
    remove_old_plot_scen_directories: bool       = False
    remove_old_plots_in_visualization: bool      = False
    MC_subdir_for_plot: str                      = '*MC*1'
    mc_plots_individual_traces: bool             = True

    default_zoom_year: List[int]                 = field(default_factory=lambda: [2002, 2030]) # field(default_factory=lambda: [2002, 2030]),
    default_zoom_hour: List[int]                 = field(default_factory=lambda: [2400, 2400+(24*7)]) # field(default_factory=lambda: [2400, 2400+(24*7)]),
    default_map_zoom: int                        = 11
    default_map_center: List[float]              = field(default_factory=lambda: [47.48, 7.57])
    node_selection_for_plots: List[str]          = field(default_factory=lambda: ['1', '3', '5'])

    # PLOT CHUCK 
    # for pvalloc_inital + sanitycheck -------------------------------------------------->  [run plot,  show plot,  show all scen]
    plot_ind_var_summary_stats_TF: List[bool]               = field(default_factory=lambda: [True,      True,       False])
    plot_ind_hist_pvcapaprod_sanitycheck_TF: List[bool]     = field(default_factory=lambda: [True,      True,       False])
    plot_ind_boxp_radiation_rng_sanitycheck_TF: List[bool]  = field(default_factory=lambda: [True,      True,       False])
    plot_ind_hist_pvprod_deviation_TF: List[bool]           = field(default_factory=lambda: [True,      True,       False])
    plot_ind_charac_omitted_gwr_TF: List[bool]              = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_meteo_radiation_TF: List[bool]            = field(default_factory=lambda: [True,      True,       False])

    plot_ind_hist_pvcapaprod_sanitycheck_specs: Dict        = field(default_factory=lambda:  {
        'xbins_hist_instcapa_abs': 0.5,
        'xbins_hist_instcapa_stand': 0.1,
        'xbins_hist_totalprodkwh_abs': 500, 
        'xbins_hist_totalprodkwh_stand': 0.05,
        'trace_color_palettes': ['Turbo', 'Viridis', 'Aggrnyl', 'Agsunset'],    #  ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'Mint', 'solar', 'Teal', 'Magenta', 'Plotly3', 'Viridis', 'Turbo', 'Blackbody']
        'trace_colval_max': 60,                            # max value for color scale; the higher the max value and the lower the increments, the more colors will be picked within the same color range of the palette
        'trace_colincr': 10,                                # increment for color scale
        'uniform_scencolor_and_KDE_TF': True,
        'export_spatial_data_for_prod0': True, 
        })
    plot_ind_charac_omitted_gwr_specs: Dict                 = field(default_factory=lambda:  {
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
            ('1004', 'Existing bldg.'),], 
        'gwr_code_name_tuples_GKAT': [
            ('1020', 'Bdg. w. exclusive living purpose'), 
            ('1030', 'other bldg. for living purpose (incl mixed use)'), 
            ('1040', 'Bdg. w. partial living purpose'), 
            ('1060', 'Bdg wo. living purpose'), 
            ('1080', 'special bdg'), 
        ]
    })

    # for pvalloc_MC_algorithm ---------------------------------------------------------->  [run plot,  show plot,  show all scen]
    plot_ind_line_installedCap_TF: List[bool]               = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_productionHOY_per_node_TF: List[bool]     = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_productionHOY_per_EGID_TF: List[bool]     = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_PVproduction_TF: List[bool]               = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_gridPremiumHOY_per_node_TF: List[bool]    = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_gridPremiumHOY_per_EGID_TF: List[bool]    = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_gridPremium_structure_TF: List[bool]      = field(default_factory=lambda: [True,      True,       False])
    plot_ind_hist_NPV_freepartitions_TF: List[bool]         = field(default_factory=lambda: [True,      True,       False])
    plot_ind_hist_pvcapaprod_TF: List[bool]                 = field(default_factory=lambda: [True,      True,       False])

    plot_ind_map_topo_egid_TF: List[bool]                   = field(default_factory=lambda: [True,      True,       False])
    plot_ind_map_topo_egid_incl_gridarea_TF: List[bool]     = field(default_factory=lambda: [True,      True,       False])

    plot_ind_map_node_connections_TF: List[bool]            = field(default_factory=lambda: [True,      True,       False])

    plot_ind_map_omitted_egids_TF: List[bool]               = field(default_factory=lambda: [True,      True,       False])
    plot_ind_lineband_contcharact_newinst_TF: List[bool]    = field(default_factory=lambda: [True,      True,       False])

    plot_ind_line_productionHOY_per_EGID_specs: Dict         = field(default_factory=lambda: {
        'include_EGID_traces_TF': True,
        'n_egid_for_info_source': 10,
        'grid_col_to_plot':         ['demand_kW', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW',
                                     'netfeedin_all_kW', 'netfeedin_all_taken_kW', 'netfeedin_all_loss_kW',
                                     'kW_threshold',], 
        'egid_col_to_plot':         ['demand_kW', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', ],
        'egid_col_only_first_few':  ['demand_kW', 'pvprod_kW', 'selfconsum_kW', ]
    })


    plot_ind_map_topo_egid_specs: Dict                      = field(default_factory=lambda: {
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
        'gridnode_area_geom_buffer': 0.5,
        'gridnode_area_palette': 'Turbo',
        'girdnode_egid_size': 20, 
        'girdnode_egid_opacity': 0.1,
        'gridnode_point_size': 15,
        'gridnode_point_opacity': 1,
        
    })
    plot_ind_map_node_connections_specs: Dict               = field(default_factory=lambda: {
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
    })
    plot_ind_map_omitted_egids_specs: Dict                  = field(default_factory=lambda: {
        'point_opacity': 0.7,
        'point_size_select_but_omitted': 10,
        'point_size_rest_not_selected': 1, # 4.5,
        'point_color_select_but_omitted': '#ed4242', # red
        'point_color_rest_not_selected': '#ff78ef',  # pink
        'export_gdfs_to_shp': True, 
    })
    plot_ind_line_contcharact_newinst_specs: Dict           = field(default_factory=lambda: {
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
        })

    # for aggregated MC_algorithms
    plot_mc_line_PVproduction: List[bool]                    = field(default_factory=lambda: [False,  True,    False])


class Visualization:
    def __init__(self, settings: Visual_Settings):
        self.visual_sett = settings

        # SETUP --------------------
        self.visual_sett.wd_path = os.getcwd()
        self.visual_sett.data_path = os.path.join(self.visual_sett.wd_path, 'data')
        self.visual_sett.visual_path = os.path.join(self.visual_sett.data_path, 'visualization')
        self.visual_sett.log_name = f'{self.visual_sett.visual_path}/visual_log.txt'

        os.makedirs(self.visual_sett.visual_path, exist_ok=True)

        # create a str list of scenarios in pvalloc to visualize (exclude by pattern recognition)
        scen_in_pvalloc_list = os.listdir(f'{self.visual_sett.data_path}/pvalloc')
        
        if not self.visual_sett.pvalloc_include_pattern_list == []:
            self.pvalloc_scen_list: list[str] = [
                scen for scen in scen_in_pvalloc_list
                if any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_include_pattern_list)
            ]
        else:
            self.pvalloc_scen_list: list[str] = [
                scen for scen in scen_in_pvalloc_list
                if not any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_exclude_pattern_list)
            ]     
        
        # create new visual directories per scenario (+ remove old ones)
        for scen in self.pvalloc_scen_list:
            visual_scen_path = f'{self.visual_sett.visual_path}/{scen}'
            if os.path.exists(visual_scen_path):
                n_same_names = len(glob.glob(f'{visual_scen_path}*/'))
                old_dir_rename = f'{visual_scen_path}_{n_same_names}_old_vers'
                os.rename(visual_scen_path, old_dir_rename)

            os.makedirs(visual_scen_path) if self.visual_sett.save_plot_by_scen_directory else None

        if self.visual_sett.remove_old_plot_scen_directories:
            old_plot_scen_dirs = glob.glob(f'{self.visual_sett.visual_path}/*old_vers')
            for dir in old_plot_scen_dirs:
                try:    
                    shutil.rmtree(dir)
                except Exception as e:
                    print(f'Could not remove {dir}: {e}')

        if self.visual_sett.remove_old_plots_in_visualization: 
            old_plots = glob.glob(f'{self.visual_sett.visual_path}/*.html')
            for file in old_plots:
                os.remove(file)


        chapter_to_logfile('start MASTER_visualization\n', self.visual_sett.log_name, overwrite_file=True)
        # print('end_setup')



    # ------------------------------------------------------------------------------------------------------
    # VISUALIZATION of PVAlloc_INITIALIZATION + SanityCHECK
    # ------------------------------------------------------------------------------------------------------
    def plot_ALL_init_sanitycheck(self, ):
        self.plot_ind_var_summary_stats()
        self.plot_ind_hist_pvcapaprod_sanitycheck()
        # self.plot_ind_boxp_radiation_rng_sanitycheck()
        self.plot_ind_charac_omitted_gwr()
        self.plot_ind_line_meteo_radiation()


    # ------------------------------------------------------------------------------------------------------
    # VISUALIZATION of PVAlloc_MC_ALGORITHM
    # ------------------------------------------------------------------------------------------------------
    def plot_ALL_mcalgorithm(self,): 
        self.plot_ind_line_installedCap()
        self.plot_ind_line_productionHOY_per_node()
        self.plot_ind_line_productionHOY_per_EGID()
        self.plot_ind_line_PVproduction()
        self.plot_ind_hist_NPV_freepartitions()
        self.plot_ind_line_gridPremiumHOY_per_node()
        self.plot_ind_line_gridPremium_structure()
        self.plot_ind_hist_NPV_freepartitions()

        self.plot_ind_map_topo_egid()
        self.plot_ind_map_topo_egid_incl_gridarea()
        # plot_ind_map_node_connections()
        # plot_ind_map_omitted_egids()
        # plot_ind_lineband_contcharact_newinst()
 

    # ------------------------------------------------------------------------------------------------------
    # PLOT-AUXILIARY FUNCTIONS
    # ------------------------------------------------------------------------------------------------------
    if True: 
        def get_pvalloc_sett_output(self,pvalloc_scen_name):
            # pickle_path = glob.glob(f'{self.data_path}/pvalloc/{pvalloc_scen_name}/*.pkl')[0]
            # with open(pickle_path, 'rb') as f:
            #     pvalloc_scen = pickle.load(f)
            sett_json_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{pvalloc_scen_name}/*pvalloc_sett*.json')[0]
            with open(sett_json_path, 'r') as f:
                sett_json = json.load(f)
            
            self.pvalloc_scen = PVAllocScenario_Settings()
            for key, value in sett_json.items():
                if hasattr(self.pvalloc_scen, key):
                    setattr(self.pvalloc_scen, key, value)

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

        def flatten_geometry(self, geom):
            if geom.has_z:
                if geom.geom_type == 'Polygon':
                    exterior = [(x, y) for x, y, z in geom.exterior.coords]
                    interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                    return Polygon(exterior, interiors)
                elif geom.geom_type == 'MultiPolygon':
                    return MultiPolygon([self.flatten_geometry(poly) for poly in geom.geoms])
            return geom

    # ------------------------------------------------------------------------------------------------------------------------
    # ALL AVAILABLE PLOTS 
    # ------------------------------------------------------------------------------------------------------------------------

    # PLOT IND SCEN: pvalloc_initalization + sanitycheck ----------------------------------------
    if True: 
        def plot_ind_var_summary_stats(self, ):
            if self.visual_sett.plot_ind_var_summary_stats_TF[0]:

                checkpoint_to_logfile(f'plot_ind_var_summary_stats', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    # total kWh by demandtypes ------------------------
                    demandtypes = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/demandtypes.parquet')
                    
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

                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_var_summary_stats_TF[1]:
                        if self.visual_sett.plot_ind_var_summary_stats_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_var_summary_stats_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_bar_totaldemand_by_type.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_bar_totaldemand_by_type.html')
                    print_to_logfile(f'\texport: plot_ind_bar_totaldemand_by_type.html (for: {scen})', self.visual_sett.log_name)



        def plot_ind_hist_pvcapaprod_sanitycheck(self,):
            if self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[0]:

                checkpoint_to_logfile(f'plot_ind_hist_pvcapaprod_sanitycheck', self.visual_sett.log_name)
                    
                # available color palettes
                trace_color_dict = {
                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
                    'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
                    'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                }        
                
                # visual settings
                plot_ind_hist_pvcapaprod_sanitycheck_specs= self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_specs
                xbins_hist_instcapa_abs, xbins_hist_instcapa_stand = plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_instcapa_abs'], plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_instcapa_stand']
                xbins_hist_totalprodkwh_abs, xbins_hist_totalprodkwh_stand = plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_totalprodkwh_abs'], plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_totalprodkwh_stand']
                
                trace_colval_max, trace_colincr, uniform_scencolor_and_KDE_TF = plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_colval_max'], plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_colincr'], plot_ind_hist_pvcapaprod_sanitycheck_specs['uniform_scencolor_and_KDE_TF']
                trace_color_palettes = plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_color_palettes']
                trace_color_palettes_list= [trace_color_dict[color] for color in trace_color_palettes]

                color_pv_df, color_solkat, color_rest = self.visual_sett.plot_ind_map_topo_egid_specs['point_color_pv_df'], self.visual_sett.plot_ind_map_topo_egid_specs['point_color_solkat'], self.visual_sett.plot_ind_map_topo_egid_specs['point_color_rest']
                
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
                        
                        self.get_pvalloc_sett_output(pvalloc_scen_name = scen)
                        panel_efficiency_print = 'dynamic' if self.pvalloc_scen.PEFspec_variable_panel_efficiency_TF else 'static'

                        # data import
                        self.visual_sett.sanity_scen_data_path = f'{self.visual_sett.data_path}/pvalloc/{scen}/sanity_check_byEGID'
                        pv = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/pv.parquet')
                        topo = json.load(open(f'{self.visual_sett.sanity_scen_data_path}/topo_egid.json', 'r'))
                        gwr_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')

                        egid_with_pvdf = [egid for egid in topo.keys() if topo[egid]['pv_inst']['info_source'] == 'pv_df']
                        xtf_in_topo = [topo[egid]['pv_inst']['xtf_id'] for egid in egid_with_pvdf]
                        topo_subdf_paths = glob.glob(f'{self.visual_sett.sanity_scen_data_path}/topo_subdf_*.parquet')
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

                            if self.visual_sett.plot_show and self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[1]:
                                if self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show()
                                elif not self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show() if i_scen == 0 else None
                            if self.visual_sett.save_plot_by_scen_directory:
                                fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__ind_hist_instCapa_kW.html')
                            else:
                                fig.write_html(f'{self.visual_sett.visual_path}/{scen}__ind_hist_instCapa_kW.html')    
                            print_to_logfile(f'\texport: plot_ind_hist_SanityCheck_instCapa_kW.html (for: {scen})', self.visual_sett.log_name)


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
                            

                            if self.visual_sett.plot_show and self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[1]:
                                if self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show()
                                elif not self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[2]:
                                    fig.show() if i_scen == 0 else None
                            if self.visual_sett.save_plot_by_scen_directory:
                                fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__ind_hist_annualPVprod_kWh.html')
                            else:
                                fig.write_html(f'{self.visual_sett.visual_path}/{scen}__ind_hist_annualPVprod_kWh.html')
                            print_to_logfile(f'\texport: plot_ind_hist_SanityCheck_annualPVprod_kWh.html (for: {scen})', self.visual_sett.log_name)


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


                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_hist_pvcapaprod_sanitycheck_TF[1]:
                            fig_agg_abs.show()
                            fig_agg_stand.show()
                        fig_agg_abs.write_html(f'{self.visual_sett.visual_path}/plot_agg_hist_pvCapaProd_abs_values__{len(self.pvalloc_scen_list)}scen_KDE{uniform_scencolor_and_KDE_TF}.html')   
                        fig_agg_stand.write_html(f'{self.visual_sett.visual_path}/plot_agg_hist_pvCapaProd_stand_values__{len(self.pvalloc_scen_list)}scen_KDE{uniform_scencolor_and_KDE_TF}.html')
                        print_to_logfile(f'\texport: plot_agg_hist_SanityCheck_instCapa_kW.html ({len(self.pvalloc_scen_list)} scens, KDE: {uniform_scencolor_and_KDE_TF})', self.visual_sett.log_name)
            

                    # Export shapes with 0 kWh annual production --------------------
                    if plot_ind_hist_pvcapaprod_sanitycheck_specs['export_spatial_data_for_prod0']:
                        os.makedirs(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_spatial_data', exist_ok=True)

                        # EGID_no_prod = aggdf_combo.loc[aggdf_combo['pvprod_kW'] == 0, 'EGID'].unique()
                        aggdf_combo_noprod = aggdf_combo.loc[aggdf_combo['pvprod_kW'] == 0]

                        # match GWR geom to gdf
                        aggdf_noprod_gwrgeom_gdf = aggdf_combo_noprod.merge(gwr_gdf, on='EGID', how='left')
                        aggdf_noprod_gwrgeom_gdf = gpd.GeoDataFrame(aggdf_noprod_gwrgeom_gdf, geometry='geometry')
                        aggdf_noprod_gwrgeom_gdf.set_crs(epsg=2056, inplace=True)
                        aggdf_noprod_gwrgeom_gdf.to_file(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_spatial_data/aggdf_noprod_gwrgeom_gdf.geojson', driver='GeoJSON')
                        print_to_logfile(f'\texport: aggdf_noprod_gwrgeom_gdf.geojson (scen: {scen}) for sanity check', self.visual_sett.log_name)

                        # try to match solkat geom to gdf
                        solkat_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/solkat_gdf.geojson')
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
                                    aggdf_noprod_gwrgeom_gdf.loc[i, 'geometry'] = union_all(dfuid_row_solkatgeom['geometry'])#  dfuid_row_solkatgeom.unary_union
                                elif len(dfuid_row_solkatgeom) == 0:
                                    aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = 'NA_dfuid_aggdf_combo_notin_solkat_gdf'

                        aggdf_noprod_solkatgeom_gdf.loc[aggdf_noprod_solkatgeom_gdf['geometry'] == 'NA', 'geometry'] = None
                        aggdf_noprod_solkatgeom_gdf = gpd.GeoDataFrame(aggdf_noprod_solkatgeom_gdf, geometry='geometry')
                        aggdf_noprod_solkatgeom_gdf.set_crs(epsg=2056, inplace=True)
                        aggdf_noprod_solkatgeom_gdf.to_file(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_spatial_data/aggdf_noprod_solkatgeom_gdf.geojson', driver='GeoJSON')
                        print_to_logfile(f'\texport: aggdf_noprod_solkatgeom_gdf.geojson (scen: {scen}) for sanity check', self.visual_sett.log_name)



        def plot_ind_boxp_radiation_rng_sanitycheck(self,): 
            if self.visual_sett.plot_ind_boxp_radiation_rng_sanitycheck_TF[0]:

                checkpoint_to_logfile(f'plot_ind_boxp_radiation_rng_sanitycheck', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
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

                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_boxp_radiation_rng_sanitycheck_TF[1]:
                        if self.visual_sett.plot_ind_boxp_radiation_rng_sanitycheck_TF[2]:
                            fig_onebox.show()
                        elif not self.visual_sett.plot_ind_boxp_radiation_rng_sanitycheck_TF[2]:
                            fig_onebox.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_onebox.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__ind_boxp_radiation_rng_sanitycheck.html')
                    else:
                        fig_onebox.write_html(f'{self.visual_sett.visual_path}/{scen}__ind_boxp_radiation_rng_sanitycheck.html')
                    print_to_logfile(f'\texport: plot_ind_boxp_radiation_rng_sanitycheck.html (for: {scen})', self.visual_sett.log_name)



        def plot_ind_charac_omitted_gwr(self, ): 
            if self.visual_sett.plot_ind_charac_omitted_gwr_TF[0]:
                plot_ind_charac_omitted_gwr_specs = self.visual_sett.plot_ind_charac_omitted_gwr_specs
                checkpoint_to_logfile(f'plot_ind_charac_omitted_gwr', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
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
                                    
                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_charac_omitted_gwr_TF[1]:
                        if self.visual_sett.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_pie_disc_charac_omitted_gwr.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_pie_disc_charac_omitted_gwr.html')
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
                    
                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_charac_omitted_gwr_TF[1]:
                        if self.visual_sett.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_charac_omitted_gwr_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_hist_cont_charac_omitted_gwr.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_hist_cont_charac_omitted_gwr.html')
                    print_to_logfile(f'\texport: plot_ind_hist_cont_charac_omitted_gwr.png (for: {scen})', self.visual_sett.log_name)
                


        def plot_ind_line_meteo_radiation(self,): 
            if self.visual_sett.plot_ind_line_meteo_radiation_TF[0]:

                checkpoint_to_logfile('plot_ind_line_meteo_radiation', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    meteo_col_dir_radiation = self.pvalloc_scen.WEAspec_meteo_col_dir_radiation
                    meteo_col_diff_radiation = self.pvalloc_scen.WEAspec_meteo_col_diff_radiation
                    meteo_col_temperature = self.pvalloc_scen.WEAspec_meteo_col_temperature

                    # import meteo data -----
                    meteo = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/meteo_ts.parquet')


                    # try to also get raw data to show how radidation is derived
                    try: 
                        meteo_raw = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/meteo.parquet')
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


                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_meteo_radiation_TF[1]:
                        if self.visual_sett.plot_ind_line_meteo_radiation_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_line_meteo_radiation_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_meteo_radiation.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_meteo_radiation.html')
                    print_to_logfile(f'\texport: plot_ind_line_meteo_radiation.html (for: {scen})', self.visual_sett.log_name)



    # PLOT IND SCEN: pvalloc_MC_algorithm ----------------------------------------
    if True: 
        def plot_ind_line_installedCap(self, ): 
            if self.visual_sett.plot_ind_line_installedCap_TF[0]:

                checkpoint_to_logfile('plot_ind_line_installedCap', self.visual_sett.log_name)
                
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
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
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
                    if self.visual_sett.plot_ind_line_installedCap_TF[0]:  #['plot_ind_line_installedCap_per_month']:
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
                        capa_year_df.to_csv(f'{self.visual_sett.visual_path}/{scen}_capa_year_df.csv')
                        capa_cumm_year_df.to_csv(f'{self.visual_sett.visual_path}/{scen}_capa_cumm_year_df.csv')
                        instcomp_mrg_df.to_csv(f'{self.visual_sett.visual_path}/{scen}_instcomp_mrg_df.csv') 
                        # -- DEBUGGING ----------


                        # plot ----------------
                        fig1 = go.Figure()

                        fig1.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name='built (month)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'cornflowerblue'), name='predicted (month)', mode='lines+markers'))
                        fig1.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'navy'),name='built + predicted (month)', mode='lines+markers'))

                        fig1.add_trace(go.Scatter(x=capa_cumm_month_df['BeginOp_month'], y=capa_cumm_month_df['Cumm_TotalPower'], line = dict(color ='blue'), name='cumulative built (month)', mode='lines+markers'))

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

                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_installedCap_TF[1]:
                            if self.visual_sett.plot_ind_line_installedCap_TF[2]:
                                fig1.show()
                            elif not self.visual_sett.plot_ind_line_installedCap_TF[2]:
                                fig1.show() if i_scen == 0 else None
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig1.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_installedCap_per_month.html')
                        else:
                            fig1.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_installedCap_per_month.html')
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
                        
                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_installedCap_TF[1]:
                            if self.visual_sett.plot_ind_line_installedCap_TF[2]:
                                fig2.show()
                            elif not self.visual_sett.plot_ind_line_installedCap_TF[2]:
                                fig2.show() if i_scen == 0 else None
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig2.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_installedCap_per_BFS.html')
                        else:
                            fig2.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_installedCap_per_BFS.html')
                        print_to_logfile(f'\texport: plot_ind_line_installedCap_per_BFS.html (for: {scen})', self.visual_sett.log_name)
                        

                    # plot add aggregated - line: Installed Capacity per Year ===========================
                    if self.visual_sett.plot_ind_line_installedCap_TF[0]:  #['plot_ind_line_installedCap_per_month']:
                    
                        color_allscen_list = [list(trace_color_dict.keys())[i_scen] for i_scen in range(len(self.pvalloc_scen_list))]
                        color_palette = trace_color_dict[list(trace_color_dict.keys())[i_scen]]

                        # fig_agg_pmonth.add_trace(go.Scatter(x=[0,], y=[0,],  name=f'',opacity=1, ))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],  name='',opacity=0, ))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'black'), name=f'{scen}',opacity=0, mode='lines+markers'))

                        # fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'],          opacity = 0.75, line = dict(color = color_palette[0+0]),    name='-- built (month)', mode='lines+markers'))
                        # fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'],  opacity = 0.75, line = dict(color = color_palette[0+1]),    name='-- predicted (month)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],                opacity = 0.75, line = dict(color = color_palette[0+2]),    name='-- built + predicted (month)', mode='lines+markers'))
                        # fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'],             opacity = 0.75, line = dict(color = color_palette[0+2]),    name='-- built (year)', mode='lines+markers'))
                        # fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'],     opacity = 0.75, line = dict(color = color_palette[0+3]),    name='-- predicted (year)', mode='lines+markers'))
                        fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'],                   opacity = 0.75, line = dict(color = color_palette[0+5]),    name='-- built + predicted (year)', mode='lines+markers',))
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
                            
                            if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_installedCap_TF[1]:
                                fig_agg_pmonth.show()

                            fig_agg_pmonth.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html')
                            print_to_logfile(f'\texport: plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)



        def plot_ind_line_productionHOY_per_node(self, ): 
            if self.visual_sett.plot_ind_line_productionHOY_per_node_TF[0]:

                checkpoint_to_logfile('plot_ind_line_productionHOY_per_node', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import ----------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0] # take first path if multiple apply, so code can still run properlyrly
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    self.visual_sett.node_selection_for_plots

                    gridnode_df = pd.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
                    gridnode_df['grid_node'].unique()
                    gridnode_df['t_int'] = gridnode_df['t'].str.extract(r't_(\d+)').astype(int)
                    gridnode_df.sort_values(by=['t_int'], inplace=True)

                    # plot ----------------
                    if any([n in self.visual_sett.node_selection_for_plots for n in gridnode_df['grid_node'].unique()]):
                        nodes = [n for n in gridnode_df['grid_node'].unique() if n in self.visual_sett.node_selection_for_plots]
                    else:
                        nodes = gridnode_df['grid_node'].unique()

                        fig = go.Figure()

                        for node in nodes:
                            filter_df = copy.deepcopy(gridnode_df.loc[gridnode_df['grid_node'] == node])
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['pvprod_kW'], name=f'{node} - pvprod_kW'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['selfconsum_kW'], name=f'{node} - selfconsum_kW'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_kW'], name=f'{node} - demand_kW'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netdemand_kW'], name=f'{node} - netdemand_kW'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_proxy_out_kW'], name=f'{node} - demand_proxy_out_kW'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_all_kW'], name=f'{node} - feedin (all)'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_all_taken_kW'], name= f'{node} - feedin_taken'))
                            fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_all_loss_kW'], name=f'{node} - feedin_loss'))

                        gridnode_total_df = gridnode_df.groupby(['t', 't_int']).agg({'pvprod_kW': 'sum', 'netfeedin_all_kW': 'sum','netfeedin_all_taken_kW': 'sum','netfeedin_all_loss_kW': 'sum'}).reset_index()
                        gridnode_total_df.sort_values(by=['t_int'], inplace=True)
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['pvprod_kW'], name='Total production', line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['netfeedin_all_kW'], name='Total feedin', line=dict(color='black', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['netfeedin_all_taken_kW'], name='Total feedin_taken', line=dict(color='green', width=2)))
                        fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['netfeedin_all_loss_kW'], name='Total feedin_loss', line=dict(color='red', width=2)))
                                    

                    fig.update_layout(
                        xaxis_title='Hour of Year',
                        yaxis_title='Production / Feedin (kW)',
                        legend_title='Node ID',
                        title = f'Production per node (kW, weather year: {self.pvalloc_scen.WEAspec_weather_year}, self consum. rate: {self.pvalloc_scen.TECspec_self_consumption_ifapplicable})'
                    )

                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                    fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)

                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_productionHOY_per_node_TF[1]:
                        if self.visual_sett.plot_ind_line_productionHOY_per_node_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_line_productionHOY_per_node_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_productionHOY_per_node.html')
                    else:   
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_productionHOY_per_node.html')
                    print_to_logfile(f'\texport: plot_ind_line_productionHOY_per_node.html (for: {scen})', self.visual_sett.log_name)



        def plot_ind_line_productionHOY_per_EGID(self, ):
            if self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[0]:

                checkpoint_to_logfile('plot_ind_line_productionHOY_per_EGID', self.visual_sett.log_name)

                n_egid_for_info_source  = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['n_egid_for_info_source']
                grid_col_to_plot        = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_col_to_plot']
                egid_col_to_plot        = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_col_to_plot']
                egid_col_only_first_few = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_col_only_first_few']

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import --------------------------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                    topo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
                    outtopo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/outtopo_time_subdf/*.parquet')
                    
                    gridnode_df = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
                    dsonodes_df = pl.read_parquet(f'{self.visual_sett.mc_data_path}/dsonodes_df.parquet')  



                    # select EGIDs by info_source --------------------------
                    egid_list, info_source_list, grid_node_list, df_uid_w_inst_list, xtf_id_list = [], [], [], [], []
                    for k, v in topo.items():
                        egid_list.append(k)
                        info_source_list.append(v['pv_inst']['info_source'])
                        grid_node_list.append(v['grid_node'])
                        df_uid_w_inst_list.append(v['pv_inst']['df_uid_w_inst'])
                        xtf_id_list.append(v['pv_inst']['xtf_id'])

                    topo_df = pd.DataFrame({'EGID': egid_list, 'info_source': info_source_list, 'grid_node': grid_node_list,
                                            'df_uid_w_inst': df_uid_w_inst_list, 'xtf_id': xtf_id_list}) 
                    
                    # select EGIDs to plot ----------
                    egid_info_list = []

                    # select all egids of smalles gridnode ----------
                    gridnodes_counts = topo_df['grid_node'].value_counts()
                    gridnodes_counts.sort_values(ascending=False, inplace=True)
                    gridnode_pick = gridnodes_counts.index[0]  # take the most common gridnode
                    egid_info = topo_df.loc[topo_df['grid_node'] == gridnode_pick].copy()

                    # # select n_egid for each info_source ----------
                    # egid_info_list = []
                    # for info in topo_df['info_source'].unique():
                    #     egid_info = topo_df.loc[topo_df['info_source'] == info].head(n_egid_for_info_source).copy()
                    #     egid_info_list.append(egid_info)
                    # egid_info = pd.concat(egid_info_list, axis=0)
                    # ----------

                    # iterate topo_time_subdf to extract production
                    k = 0
                    path, i, egid_info_row = topo_subdf_paths[k], k, egid_info.iloc[k]
                    subdf_prod_df_list = [] 
                    for i_path, path in enumerate(topo_subdf_paths):
                        subdf = pl.read_parquet(path)

                        subdf = subdf.join(dsonodes_df[['grid_node', 'kVA_threshold', ]], on='grid_node', how='left')
                        subdf = subdf.with_columns((pl.col('kVA_threshold') * self.pvalloc_scen.GRIDspec_perf_factor_1kVA_to_XkW).alias("kW_threshold"))

                        for i, egid_info_row in egid_info.iterrows():

                            # select and calculate feedin --------------------------
   
                            if egid_info_row["info_source"] == '':
                                subdf_prod = subdf.filter(
                                    (pl.col("EGID") == egid_info_row["EGID"]) 
                                )
                            elif isinstance(egid_info_row["df_uid_w_inst"], str):
                                subdf_prod = subdf.filter(
                                    (pl.col("EGID") == egid_info_row["EGID"]) &
                                    (pl.col("df_uid") == egid_info_row["df_uid_w_inst"])
                                )
                            elif isinstance(egid_info_row["df_uid_w_inst"], list):
                                subdf_prod = subdf.filter(
                                    (pl.col("EGID") == egid_info_row["EGID"]) &
                                    (pl.col("df_uid").is_in(egid_info_row["df_uid_w_inst"]))
                                )


                            subdf_prod = subdf_prod.with_columns(
                                pl.col("t").str.extract(r"t_(\d+)").cast(pl.Int32).alias("t_int")
                            )
                            subdf_prod = subdf_prod.group_by(["EGID", "t", "t_int"]).agg([
                                pl.col("grid_node").first(),
                                pl.col("pvprod_kW").sum(),
                                pl.col("demand_kW").first(), 
                                pl.col("kW_threshold").first(),
                            ])

                            selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) 

                            subdf_prod = subdf_prod.with_columns([        
                                selfconsum_expr.alias("selfconsum_kW"),
                                (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                                (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW"),
                            ])

                            subdf_prod_df_list.append(subdf_prod)

                    subdf_prod_df = pl.concat(subdf_prod_df_list)

                    
                    # import outtopo_time_subfs -----------------------------------------------------
                    agg_subdf_df_list = []
                    for i, path in enumerate(outtopo_subdf_paths):
                        outsubdf = pl.read_parquet(path)  
                        agg_outsubdf = outsubdf.group_by(["grid_node", "t"]).agg([
                            pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
                        ])
                        del outsubdf
                        agg_subdf_df_list.append(agg_outsubdf)
                        
                    agg_outsubdf_df = pl.concat(agg_subdf_df_list)
                    outtopo_gridnode_df = agg_outsubdf_df.group_by(['grid_node', 't']).agg([
                        pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
                    ])
                    subdf_prod_df = subdf_prod_df.join(outtopo_gridnode_df, on=['grid_node', 't'], how='left')

                        
                    # make arbitrary gridnode_df calculation --------------------------
                    subdf_prod_agg_byegid = subdf_prod_df.group_by(['EGID', 't', 't_int']).agg(
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('kW_threshold').first().alias('kW_threshold'),
                        pl.col('demand_kW').first().alias('demand_kW'),
                        pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                    )

                    subdf_prod_agg_bynode = subdf_prod_agg_byegid.group_by(['grid_node', 't', 't_int']).agg(
                        pl.col('kW_threshold').first().alias('kW_threshold'),
                        pl.col('demand_kW').sum().alias('demand_kW'),
                        pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'), 
                    )

                    # code replication from gridnode_updating
                    subdf_prod_agg_bynode = subdf_prod_agg_bynode.with_columns([
                        (pl.col('netfeedin_kW') - pl.col('netdemand_kW') - pl.col('demand_proxy_out_kW')).alias('netfeedin_all_kW'),
                    ])
                    subdf_prod_agg_bynode = subdf_prod_agg_bynode.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") < 0)
                        .then(0)
                        .otherwise(pl.col("netfeedin_all_kW"))
                        .alias("netfeedin_all_kW"),
                        ])
                    subdf_prod_agg_bynode = subdf_prod_agg_bynode.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") > pl.col("kW_threshold"))
                        .then(pl.col("kW_threshold"))
                        .otherwise(pl.col("netfeedin_all_kW"))
                        .alias("netfeedin_all_taken_kW"),
                        ])
                    subdf_prod_agg_bynode = subdf_prod_agg_bynode.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") > pl.col("kW_threshold"))
                        .then(pl.col("netfeedin_all_kW") - pl.col("kW_threshold"))
                        .otherwise(0)
                        .alias("netfeedin_all_loss_kW")
                    ])
                

                    # plot --------------------------
                    include_EGID_traces_TF = [True, False]
                    for incl_EGID_TF in include_EGID_traces_TF: 
                        subdf_prod_agg_byegid_df = subdf_prod_agg_byegid.to_pandas()
                        subdf_prod_agg_bynode_df = subdf_prod_agg_bynode.to_pandas()
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'grid_node_from_subdf {20*"-"}', opacity=0)) 
                        for node in subdf_prod_agg_bynode_df['grid_node'].unique():
                            node_subdf = subdf_prod_agg_bynode_df.loc[subdf_prod_agg_bynode_df['grid_node'] == node]
                            node_subdf = node_subdf.sort_values(by=['t_int'])   

                            for col in grid_col_to_plot:
                                fig.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[col], name=f'grid_node: {node} - {col}'))
                        
                        # sanity check ----------
                        # add values form gridnode_df to make in-plot sanity check
                        fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name='', opacity=0)) 
                        fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'gridnode_df_sanitycheck {20*"-"}', opacity=0)) 
                        if 't_int' not in gridnode_df.columns:
                            gridnode_df = gridnode_df.with_columns([
                                pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                            ])
                            gridnode_df = gridnode_df.sort("t_int", descending=False)            
                            gridnode_df = gridnode_df.filter(pl.col('grid_node') == gridnode_pick)

                        gridnode_df_pd = gridnode_df.to_pandas()
                        for col in grid_col_to_plot:
                            fig.add_trace(go.Scatter(x=gridnode_df_pd['t_int'], y=gridnode_df_pd[col], name=f'from gridnode_df: {gridnode_pick} - {col}', 
                                                    mode = 'lines+markers',))
                        fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name='', opacity=0)) 
                        # ----------

                        if incl_EGID_TF:
                            for i_egid, egid in enumerate(egid_info['EGID'].unique()):
                                egid_subdf = subdf_prod_agg_byegid_df.loc[subdf_prod_agg_byegid_df['EGID'] == egid]
                                egid_subdf = egid_subdf.sort_values(by=['t_int'])

                                for col in egid_col_to_plot:
                                    if col not in egid_col_only_first_few or i_egid < 3:  # only plot first few egids with all columns
                                        fig.add_trace(go.Scatter(x=egid_subdf['t_int'], y=egid_subdf[col], name=f'EGID: {egid:15} - {col}'))
                                        fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name='', opacity=0)) 


                        fig.update_layout(
                            title = 'Production per EGID / Grid Node (kW, Hour of Year)', 
                            xaxis_title='Hour of Year',
                            yaxis_title='Production / Feedin (kW)',
                            legend_title='EGID / Node ID',
                            template='plotly_white',
                            showlegend=True,
                        )
                        fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                        fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)


                        
                        # export plot --------------------------
                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[1]:
                            if self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[2]:
                                fig.show()
                            elif not self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[2]:
                                fig.show() if i_scen == 0 else None
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_productionHOY_per_EGID_inclEGID_{incl_EGID_TF}.html')
                        else:
                            fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_productionHOY_per_EGID_inclEGID_{incl_EGID_TF}.html')
                        print_to_logfile(f'\texport: plot_ind_line_productionHOY_per_EGID.html (for: {scen})', self.visual_sett.log_name)



        def plot_ind_line_PVproduction(self, ): 
            if self.visual_sett.plot_ind_line_PVproduction_TF[0]:

                checkpoint_to_logfile('plot_ind_line_PVproduction', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                    topo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
                    gridnode_df_paths = glob.glob(f'{self.visual_sett.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_*.parquet')
                    gridnode_df = pd.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')

                    # get installations of topo over time
                    egid_list, inst_TF_list, info_source_list, BeginOp_list, xtf_id_list, TotalPower_list, df_uid_list = [], [], [], [], [], [], []
                    k = list(topo.keys())[0]
                    for k, v in topo.items():
                        egid_list.append(k)
                        inst_TF_list.append(v['pv_inst']['inst_TF'])
                        info_source_list.append(v['pv_inst']['info_source'])
                        BeginOp_list.append(v['pv_inst']['BeginOp'])
                        xtf_id_list.append(v['pv_inst']['xtf_id'])
                        df_uid_list.append(v['pv_inst']['df_uid_w_inst'])
                        TotalPower_list.append(v['pv_inst']['TotalPower'])

                    pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                            'BeginOp': BeginOp_list, 'xtf_id': xtf_id_list, 'df_uid': df_uid_list,
                                            'TotalPower': TotalPower_list,})
                    pvinst_df = pvinst_df.loc[pvinst_df['inst_TF'] == True]


                    pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')
                    pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
                    pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')


                    # attach annual production to each installation
                    pvinst_df['pvprod_kW'] = float(0)
                    aggdf_combo_list = []

                    for ipath, path in enumerate(topo_subdf_paths):
                        subdf = pd.read_parquet(path)

                        # for i, row in pvinst_df.iterrows():
                            # select and calculate feedin --------------------------
                            # if row['info_source
                        subdf = subdf.loc[subdf['EGID'].isin(pvinst_df['EGID'])]

                        agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG']).agg({
                            'pvprod_kW': 'sum',
                        }).reset_index() 
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
                    # prod_month_df.sort_values(by=['BeginOp_month'], inplace=True)
                    # prod_month_df.sort_values(by=['BeginOp'])


                    # aggregate gridnode_df to monthly values
                    BeginOp_month_list, feedin_all_list, feedin_taken_list, feedin_loss_list = [], [], [], []
                    
                    # ONLY KEEP THIS WHILE NOT ALL MONTHS ARE EXPORTED in PVALLOC
                    # month_iters = [path.split('gridnode_df_')[1].split('.parquet')[0] for path in gridnode_df_paths]
                    # gridnode_df_month_iters = [path.split('pred_gridprem_node_by_M\\')[1].split('.parquet')[0] for path in gridnode_df_paths]
                    # prod_month_df = prod_month_df.loc[prod_month_df['BeginOp_month_str'].isin(month_iters)]

                    # month = prod_month_df['BeginOp_month'].unique()[0]
                    for month in prod_month_df['BeginOp_month'].unique():
                        month_str = prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'BeginOp_month_str'].values[0]
                        grid_subdf = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_{month_str}.parquet')
                        
                        prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'netfeedin_all_kW'] = grid_subdf['netfeedin_all_kW'].sum()
                        prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'netfeedin_all_taken_kW'] = grid_subdf['netfeedin_all_taken_kW'].sum()
                        prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'netfeedin_all_loss_kW'] = grid_subdf['netfeedin_all_loss_kW'].sum()



                    # plot ----------------
                    # fig = go.Figure()
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['pvprod_kW_month'], name='EGID Prod kWh (total pvprod_kW)', ))
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['netfeedin_all_kW'], name='Grid feedin kWh (feedin_kwh)', ))
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['netfeedin_all_taken_kW'], name='Grid feedin take kWh (feedin_taken kWh)', ))
                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['netfeedin_all_loss_kW'], name='Grid feedin loss kWh (feedin_loss kWh)', ))

                    fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['TotalPower_month'], name='Total installed capacity', line=dict(color='blue', width=2)), secondary_y=True)

                    fig.update_layout(
                        title=f'PV production per month',
                        xaxis_title='Month',
                        yaxis_title='Production [kW]',
                        yaxis2_title='Installed capacity [kW]',
                        legend_title='Legend',
                    )
                    fig.update_yaxes(title_text="Installed capacity [kW]", secondary_y=True)

                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_PVproduction_TF[1]:
                        if self.visual_sett.plot_ind_line_PVproduction_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_line_PVproduction_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_PVproduction.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_PVproduction.html')
                    print_to_logfile(f'\texport: plot_ind_line_PVproduction.html (for: {scen})', self.visual_sett.log_name)



        def plot_ind_hist_NPV_freepartitions(self, ): 
            if self.visual_sett.plot_ind_hist_NPV_freepartitions_TF[0]:

                checkpoint_to_logfile('plot_ind_hist_NPV_freepartitions', self.visual_sett.log_name)

                fig_agg = go.Figure()

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    # setup + import ----------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    
                    npv_df_paths = glob.glob(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_*.parquet')
                    # periods_list = [pd.to_datetime(path.split('npv_df_')[-1].split('.parquet')[0]) for path in npv_df_paths]
                    periods_list = [path.split('npv_df_')[-1].split('.parquet')[0] for path in npv_df_paths]
                    before_period, after_period = min(periods_list), max(periods_list)

                    npv_df_before = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_{before_period.to_period("M")}.parquet')
                    npv_df_after  = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_{after_period.to_period("M")}.parquet')

                    # plot ----------------
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name='Before Allocation Algorithm', opacity=0.5))
                    fig.add_trace(go.Histogram(x=npv_df_after['NPV_uid'], name='After Allocation Algorithm', opacity=0.5))

                    fig.update_layout(
                        xaxis_title=f'Net Present Value (NPV, interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr)',
                        yaxis_title='Frequency',
                        title = f'NPV Distribution of possible PV installations, first / last year (weather year: {self.pvalloc_scen.WEAspec_weather_year})',
                        barmode = 'overlay')
                    fig.update_traces(bingroup=1, opacity=0.5)

                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
                        
                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_hist_NPV_freepartitions_TF[1]:
                        if self.visual_sett.plot_ind_hist_NPV_freepartitions_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_hist_NPV_freepartitions_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_hist_NPV_freepartitions.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_hist_NPV_freepartitions.html')
                        

                    # aggregate plot ----------------
                    fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
                    fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,)) 

                    fig_agg.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name=f'Before Allocation', opacity=0.7, xbins=dict(size=500)))
                    fig_agg.add_trace(go.Histogram(x=npv_df_after['NPV_uid'],  name=f'After Allocation',  opacity=0.7, xbins=dict(size=500)))

                fig_agg.update_layout(
                    xaxis_title=f'Net Present Value (NPV, interest rate: {self.pvalloc_scen.TECspec_interest_rate}, maturity: {self.pvalloc_scen.TECspec_invst_maturity} yr)',
                    yaxis_title='Frequency',
                    title = f'NPV Distribution of possible PV installations, first / last year ({len(self.pvalloc_scen_list)} scen, weather year: {self.pvalloc_scen.WEAspec_weather_year})',
                    barmode = 'overlay')
                # fig_agg.update_traces(bingroup=1, opacity=0.75)

                if self.visual_sett.plot_show and self.visual_sett.plot_ind_hist_NPV_freepartitions_TF[1]:
                    fig_agg.show()
                fig_agg.write_html(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions__{len(self.pvalloc_scen_list)}scen.html')

               

        def plot_ind_line_gridPremiumHOY_per_node(self, ): 
            if self.visual_sett.plot_ind_line_gridPremiumHOY_per_node_TF[0]:

                checkpoint_to_logfile('plot_ind_line_gridPremiumHOY_per_node', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    # setup + import ----------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    gridprem_ts = pd.read_parquet(f'{self.visual_sett.mc_data_path}/gridprem_ts.parquet')
                    gridprem_ts['t_int'] = gridprem_ts['t'].str.extract(r't_(\d+)').astype(int)


                    # plot ----------------
                    fig = go.Figure()
                    for node in self.visual_sett.node_selection_for_plots:
                        gridprem_ts_node = gridprem_ts[gridprem_ts['grid_node'] == node]
                        gridprem_ts_node.sort_values(by=['t_int'], inplace=True)

                        fig.add_trace(go.Scatter(x=gridprem_ts_node['t_int'], y=gridprem_ts_node['prem_Rp_kWh'],
                                                 mode='lines', name=f'grid_premium node: {node}'))
                        
                    agg_gridprem = gridprem_ts.groupby('t_int').agg({
                        't': 'first',
                        'prem_Rp_kWh': ['mean', 'std']
                    }).reset_index()
                    agg_gridprem.columns = ['t_int', 't', 'prem_Rp_kWh_mean', 'prem_Rp_kWh_std']

                    fig.add_trace(go.Scatter
                        (x=agg_gridprem['t_int'], 
                         y=agg_gridprem['prem_Rp_kWh_mean'],
                         mode='lines',
                         name='grid premium Rp/kWh (mean)',
                         line=dict(color='black', width=2)
                        ))
                    fig.add_trace(go.Scatter
                        (x=agg_gridprem['t_int'],
                         y=agg_gridprem['prem_Rp_kWh_std'],
                         mode='lines',
                         name='grid premium Rp/kWh (std)',
                         line=dict(color='darkgrey', width=1,
                        dash='dash')
                        ))

                    # layout ----------------
                    fig.update_layout(
                        title=f'Grid premium Rp/kWh per node (scen: {scen})',
                        xaxis_title='Hour of year',
                        yaxis_title='Grid premium Rp/kWh',
                        showlegend=True,
                    )
                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen_list[i_scen])
                    fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)


                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_gridPremiumHOY_per_node_TF[1]:
                        if self.visual_sett.plot_ind_line_gridPremiumHOY_per_node_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_line_gridPremiumHOY_per_node_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_gridPremiumHOY_per_node.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_gridPremiumHOY_per_node.html')



        def plot_ind_line_gridPremium_structure(self, ): 
            if self.visual_sett.plot_ind_line_gridPremium_structure_TF[0]:
                
                checkpoint_to_logfile('plot_ind_line_gridPremium_structure', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    # setup + import ----------
                    tiers = self.pvalloc_scen.GRIDspec_tiers
                    tiers_rel_treshold_list, gridprem_Rp_kWh_list = [], []
                    for k,v in tiers.items():
                        tiers_rel_treshold_list.append(v[0])
                        gridprem_Rp_kWh_list.append(v[1])

                    gridprem_tiers_df = pd.DataFrame({'tiers_rel_treshold': tiers_rel_treshold_list, 'gridprem_Rp_kWh': gridprem_Rp_kWh_list})

                    # plot ----------
                    fig = go.Figure()

                    fig.add_trace(go.Scatter
                                    (x=gridprem_tiers_df['tiers_rel_treshold'], 
                                     y=gridprem_tiers_df['gridprem_Rp_kWh'],
                                     mode='lines+markers',
                                     name='gridprem, marginal feedin premium (Rp)',
                                     showlegend=True,
                                     ))
                    fig.update_layout(
                        title='Grid premium structure, feed-in premium for reaching relative grid node capacity (kVA)',
                        xaxis_title='Relative grid node capacity threshold',
                        yaxis_title='Feed-in premium (Rp/kWh)',
                    )

                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen_list[i_scen])

                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_gridPremium_structure_TF[1]:
                        if self.visual_sett.plot_ind_line_gridPremium_structure_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_line_gridPremium_structure_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_gridPremium_structure.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_gridPremium_structure.html')
                        

        def plot_ind_map_topo_egid(self, ): 
            if self.visual_sett.plot_ind_map_topo_egid_TF[0]:

                map_topo_egid_specs = self.visual_sett.plot_ind_map_topo_egid_specs
                checkpoint_to_logfile('plot_ind_map_topo_egid', self.visual_sett.log_name)

                trace_color_dict = {
                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
                    'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
                    'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                    'Rainbow': pc.sequential.Rainbow, 
                }     

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    # get pvinst_gdf ----------------
                    if True: 
                        self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]

                        # import
                        gwr_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                        gm_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
                        dsonodes_gdf = gpd.read_file(f'{self.visual_sett.data_path}/pvalloc/{scen}/dsonodes_gdf.geojson')

                        topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                        egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list= [], [], [], [], [], []
                        gklas_list, node_list, demand_type_list, pvtarif_list, elecpri_list, elecpri_info_list = [], [], [], [], [], []

                        for k,v, in topo.items():
                            egid_list.append(k)
                            inst_TF_list.append(v['pv_inst']['inst_TF'])
                            info_source_list.append(v['pv_inst']['info_source'])
                            BeginOp_list.append(v['pv_inst']['BeginOp'])
                            TotalPower_list.append(v['pv_inst']['TotalPower'])
                            bfs_list.append(v['gwr_info']['bfs'])
                            gklas_list.append(v['gwr_info']['gklas'])
                            node_list.append(v['grid_node'])
                            demand_type_list.append(v['demand_type'])
                            pvtarif_list.append(v['pvtarif_Rp_kWh'])
                            elecpri_list.append(v['elecpri_Rp_kWh'])
                            elecpri_info_list.append(v['elecpri_info'])

                        pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                                'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list, 
                                                'gklas': gklas_list, 'grid_node': node_list, 'demand_type': demand_type_list,
                                                'pvtarif': pvtarif_list, 'elecpri': elecpri_list, 'elecpri_info': elecpri_info_list })
                        pvinst_df = pvinst_df.merge(gwr_gdf[['geometry', 'EGID']], on='EGID', how='left')
                        pvinst_gdf = gpd.GeoDataFrame(pvinst_df, crs='EPSG:2056', geometry='geometry')
                    
                    # base map ----------------
                    if True:
                        # transformations
                        gm_gdf['BFS_NUMMER'] = gm_gdf['BFS_NUMMER'].astype(str)
                        gm_gdf = gm_gdf.loc[gm_gdf['BFS_NUMMER'].isin(pvinst_df['bfs'].unique())].copy()
                        date_cols = [col for col in gm_gdf.columns if (gm_gdf[col].dtype == 'datetime64[ns]') or (gm_gdf[col].dtype == 'datetime64[ms]')]
                        gm_gdf.drop(columns=date_cols, inplace=True)
                        
                        # add map relevant columns
                        gm_gdf['hover_text'] = gm_gdf.apply(lambda row: f"{row['NAME']}<br>BFS_NUMMER: {row['BFS_NUMMER']}", axis=1)

                        # geo transformations
                        gm_gdf = gm_gdf.to_crs('EPSG:4326')
                        gm_gdf['geometry'] = gm_gdf['geometry'].apply(self.flatten_geometry)

                        # geojson = gm_gdf.__geo_interface__
                        geojson = json.loads(gm_gdf.to_json())

                        # Plot using Plotly Express
                        fig_topobase = px.choropleth_mapbox(
                            gm_gdf,
                            geojson=geojson,
                            locations="BFS_NUMMER",  # Link BFS_NUMMER for color and location
                            featureidkey="properties.BFS_NUMMER",  # This must match the GeoJSON's property for BFS_NUMMER
                            color_discrete_sequence=[map_topo_egid_specs['uniform_municip_color']],  # Apply the single color to all shapes
                            hover_name="hover_text",  # Use the new column for hover text
                            mapbox_style="carto-positron",  # Basemap style
                            center={"lat": self.visual_sett.default_map_center[0], "lon": self.visual_sett.default_map_center[1]},  # Center the map on the region
                            zoom=self.visual_sett.default_map_zoom,  # Adjust zoom as needed
                            opacity=map_topo_egid_specs['shape_opacity'],   # Opacity to make shapes and basemap visible    
                        )   
                        # Update layout for borders and title
                        fig_topobase.update_layout(
                            mapbox=dict(
                                layers=[{
                                    'source': geojson,
                                    'type': 'line',
                                    'color': 'black',  # Set border color for polygons
                                    'opacity': 0.25,
                                }]
                            ),
                            title=f"Map of PV topology (scen: {scen})", 
                            legend=dict(
                                itemsizing='constant',
                                title='Legend',
                                traceorder='normal'
                            ),
                        )


                    # topo egid map: highlight EGIDs selected for summary ----------------
                    if True:
                        fig_topoegid = copy.deepcopy(fig_topobase)
                        pvinst_gdf = pvinst_gdf.to_crs('EPSG:4326')
                        pvinst_gdf['geometry'] = pvinst_gdf['geometry'].apply(self.flatten_geometry)

                        if len(glob.glob(f'{self.visual_sett.mc_data_path}/topo_egid_summary_byEGID/*.csv')) > 1:
                            files_sanity_check = glob.glob(f'{self.visual_sett.mc_data_path}/topo_egid_summary_byEGID/*.csv')
                            file = files_sanity_check[0]
                            egid_sanity_check = [file.split('summary_')[-1].split('.csv')[0] for file in files_sanity_check]

                            subinst4_gdf = pvinst_gdf.copy()
                            subinst4_gdf = subinst4_gdf.loc[subinst4_gdf['EGID'].isin(egid_sanity_check)]
                            
                            # Add the points using Scattermapbox
                            fig_topoegid.add_trace(go.Scattermapbox(lat=subinst4_gdf.geometry.y,lon=subinst4_gdf.geometry.x, mode='markers',
                                marker=dict(
                                    size=map_topo_egid_specs['point_size_sanity_check'],
                                    color=map_topo_egid_specs['point_color_sanity_check'],
                                    opacity=map_topo_egid_specs['point_opacity_sanity_check']
                                ),
                                name = 'EGIDs in sanity check xlsx',
                            ))
                        
                    # topo egid map: all buildings ----------------
                    if True:
                        # subset inst_gdf for different traces in map plot
                        pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>node: {row['grid_node']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)

                        subinst1_gdf, subinst2_gdf, subinst3_gdf  = pvinst_gdf.copy(), pvinst_gdf.copy(), pvinst_gdf.copy()
                        subinst1_gdf, subinst2_gdf, subinst3_gdf = subinst1_gdf.loc[(subinst1_gdf['inst_TF'] == True) & (subinst1_gdf['info_source'] == 'pv_df')], subinst2_gdf.loc[(subinst2_gdf['inst_TF'] == True) & (subinst2_gdf['info_source'] == 'alloc_algorithm')], subinst3_gdf.loc[(subinst3_gdf['inst_TF'] == False)]

                        # Add the points using Scattermapbox
                        fig_topoegid.add_trace(go.Scattermapbox(lat=subinst1_gdf.geometry.y,lon=subinst1_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_topo_egid_specs['point_size_pv'],
                                color=map_topo_egid_specs['point_color_pv_df'],
                                opacity=map_topo_egid_specs['point_opacity']
                            ),
                            name = 'house w pv (real)',
                            text=subinst1_gdf['hover_text'],
                            hoverinfo='text'
                        ))
                        fig_topoegid.add_trace(go.Scattermapbox(lat=subinst2_gdf.geometry.y,lon=subinst2_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_topo_egid_specs['point_size_pv'],
                                color=map_topo_egid_specs['point_color_alloc_algo'],
                                opacity=map_topo_egid_specs['point_opacity']
                            ),
                            name = 'house w pv (predicted)',
                            text=subinst2_gdf['hover_text'],
                            hoverinfo='text'
                        ))
                        fig_topoegid.add_trace(go.Scattermapbox(lat=subinst3_gdf.geometry.y,lon=subinst3_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_topo_egid_specs['point_size_rest'],
                                color=map_topo_egid_specs['point_color_rest'],
                                opacity=map_topo_egid_specs['point_opacity']
                            ),
                            name = 'house w/o pv',
                            text=subinst3_gdf['hover_text'],
                            hoverinfo='text'
                        ))
 
                    
                    # Update layout ----------------
                    fig_topoegid.update_layout(
                            title=f"Map of model PV Topology ({scen})",
                            mapbox=dict(
                                style="carto-positron",
                                center={"lat": self.visual_sett.default_map_center[0], "lon": self.visual_sett.default_map_center[1]},  # Center the map on the region
                                zoom=self.visual_sett.default_map_zoom
                            ))


                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_map_topo_egid_TF[1]:
                        if self.visual_sett.plot_ind_map_topo_egid_TF[2]:
                            fig_topoegid.show()
                        elif not self.visual_sett.plot_ind_map_topo_egid_TF[2]:
                            fig_topoegid.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_map_topo_egid.html')
                    else:
                        fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_map_topo_egid.html')


        def plot_ind_map_topo_egid_incl_gridarea(self, ): 
            if self.visual_sett.plot_ind_map_topo_egid_incl_gridarea_TF[0]:

                map_topo_egid_specs = self.visual_sett.plot_ind_map_topo_egid_specs
                checkpoint_to_logfile('plot_ind_map_topo_egid', self.visual_sett.log_name)

                trace_color_dict = {
                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
                    'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
                    'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                    'Rainbow': pc.sequential.Rainbow, 
                }     

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    # get pvinst_gdf ----------------
                    if True: 
                        self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]

                        # import
                        gwr_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                        gm_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
                        dsonodes_gdf = gpd.read_file(f'{self.visual_sett.data_path}/pvalloc/{scen}/dsonodes_gdf.geojson')

                        topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                        egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list= [], [], [], [], [], []
                        gklas_list, node_list, demand_type_list, pvtarif_list, elecpri_list, elecpri_info_list = [], [], [], [], [], []

                        for k,v, in topo.items():
                            egid_list.append(k)
                            inst_TF_list.append(v['pv_inst']['inst_TF'])
                            info_source_list.append(v['pv_inst']['info_source'])
                            BeginOp_list.append(v['pv_inst']['BeginOp'])
                            TotalPower_list.append(v['pv_inst']['TotalPower'])
                            bfs_list.append(v['gwr_info']['bfs'])
                            gklas_list.append(v['gwr_info']['gklas'])
                            node_list.append(v['grid_node'])
                            demand_type_list.append(v['demand_type'])
                            pvtarif_list.append(v['pvtarif_Rp_kWh'])
                            elecpri_list.append(v['elecpri_Rp_kWh'])
                            elecpri_info_list.append(v['elecpri_info'])

                        pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                                'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list, 
                                                'gklas': gklas_list, 'grid_node': node_list, 'demand_type': demand_type_list,
                                                'pvtarif': pvtarif_list, 'elecpri': elecpri_list, 'elecpri_info': elecpri_info_list })
                        pvinst_df = pvinst_df.merge(gwr_gdf[['geometry', 'EGID']], on='EGID', how='left')
                        pvinst_gdf = gpd.GeoDataFrame(pvinst_df, crs='EPSG:2056', geometry='geometry')
                    
                    # base map ----------------
                    if True:
                        # transformations
                        gm_gdf['BFS_NUMMER'] = gm_gdf['BFS_NUMMER'].astype(str)
                        gm_gdf = gm_gdf.loc[gm_gdf['BFS_NUMMER'].isin(pvinst_df['bfs'].unique())].copy()
                        date_cols = [col for col in gm_gdf.columns if (gm_gdf[col].dtype == 'datetime64[ns]') or (gm_gdf[col].dtype == 'datetime64[ms]')]
                        gm_gdf.drop(columns=date_cols, inplace=True)
                        
                        # add map relevant columns
                        gm_gdf['hover_text'] = gm_gdf.apply(lambda row: f"{row['NAME']}<br>BFS_NUMMER: {row['BFS_NUMMER']}", axis=1)

                        # geo transformations
                        gm_gdf = gm_gdf.to_crs('EPSG:4326')
                        gm_gdf['geometry'] = gm_gdf['geometry'].apply(self.flatten_geometry)

                        # geojson = gm_gdf.__geo_interface__
                        geojson = json.loads(gm_gdf.to_json())

                        # Plot using Plotly Express
                        fig_topobase = px.choropleth_mapbox(
                            gm_gdf,
                            geojson=geojson,
                            locations="BFS_NUMMER",  # Link BFS_NUMMER for color and location
                            featureidkey="properties.BFS_NUMMER",  # This must match the GeoJSON's property for BFS_NUMMER
                            color_discrete_sequence=[map_topo_egid_specs['uniform_municip_color']],  # Apply the single color to all shapes
                            hover_name="hover_text",  # Use the new column for hover text
                            mapbox_style="carto-positron",  # Basemap style
                            center={"lat": self.visual_sett.default_map_center[0], "lon": self.visual_sett.default_map_center[1]},  # Center the map on the region
                            zoom=self.visual_sett.default_map_zoom,  # Adjust zoom as needed
                            opacity=map_topo_egid_specs['shape_opacity'],   # Opacity to make shapes and basemap visible    
                        )   
                        # Update layout for borders and title
                        fig_topobase.update_layout(
                            mapbox=dict(
                                layers=[{
                                    'source': geojson,
                                    'type': 'line',
                                    'color': 'black',  # Set border color for polygons
                                    'opacity': 0.25,
                                }]
                            ),
                            title=f"Map of PV topology (scen: {scen})", 
                            legend=dict(
                                itemsizing='constant',
                                title='Legend',
                                traceorder='normal'
                            ),
                        )


                    # topo egid map: highlight EGIDs selected for summary ----------------
                    if True:
                        fig_topoegid = copy.deepcopy(fig_topobase)
                        pvinst_gdf = pvinst_gdf.to_crs('EPSG:4326')
                        pvinst_gdf['geometry'] = pvinst_gdf['geometry'].apply(self.flatten_geometry)

                        if len(glob.glob(f'{self.visual_sett.mc_data_path}/topo_egid_summary_byEGID/*.csv')) > 1:
                            files_sanity_check = glob.glob(f'{self.visual_sett.mc_data_path}/topo_egid_summary_byEGID/*.csv')
                            file = files_sanity_check[0]
                            egid_sanity_check = [file.split('summary_')[-1].split('.csv')[0] for file in files_sanity_check]

                            subinst4_gdf = pvinst_gdf.copy()
                            subinst4_gdf = subinst4_gdf.loc[subinst4_gdf['EGID'].isin(egid_sanity_check)]
                            
                            # Add the points using Scattermapbox
                            fig_topoegid.add_trace(go.Scattermapbox(lat=subinst4_gdf.geometry.y,lon=subinst4_gdf.geometry.x, mode='markers',
                                marker=dict(
                                    size=map_topo_egid_specs['point_size_sanity_check'],
                                    color=map_topo_egid_specs['point_color_sanity_check'],
                                    opacity=map_topo_egid_specs['point_opacity_sanity_check']
                                ),
                                name = 'EGIDs in sanity check xlsx',
                            ))
                        
                    # topo egid map: all buildings ----------------
                    if True:
                        # subset inst_gdf for different traces in map plot
                        pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>node: {row['grid_node']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)

                        subinst1_gdf, subinst2_gdf, subinst3_gdf  = pvinst_gdf.copy(), pvinst_gdf.copy(), pvinst_gdf.copy()
                        subinst1_gdf, subinst2_gdf, subinst3_gdf = subinst1_gdf.loc[(subinst1_gdf['inst_TF'] == True) & (subinst1_gdf['info_source'] == 'pv_df')], subinst2_gdf.loc[(subinst2_gdf['inst_TF'] == True) & (subinst2_gdf['info_source'] == 'alloc_algorithm')], subinst3_gdf.loc[(subinst3_gdf['inst_TF'] == False)]

                        # Add the points using Scattermapbox
                        fig_topoegid.add_trace(go.Scattermapbox(lat=subinst1_gdf.geometry.y,lon=subinst1_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_topo_egid_specs['point_size_pv'],
                                color=map_topo_egid_specs['point_color_pv_df'],
                                opacity=map_topo_egid_specs['point_opacity']
                            ),
                            name = 'house w pv (real)',
                            text=subinst1_gdf['hover_text'],
                            hoverinfo='text'
                        ))
                        fig_topoegid.add_trace(go.Scattermapbox(lat=subinst2_gdf.geometry.y,lon=subinst2_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_topo_egid_specs['point_size_pv'],
                                color=map_topo_egid_specs['point_color_alloc_algo'],
                                opacity=map_topo_egid_specs['point_opacity']
                            ),
                            name = 'house w pv (predicted)',
                            text=subinst2_gdf['hover_text'],
                            hoverinfo='text'
                        ))
                        fig_topoegid.add_trace(go.Scattermapbox(lat=subinst3_gdf.geometry.y,lon=subinst3_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_topo_egid_specs['point_size_rest'],
                                color=map_topo_egid_specs['point_color_rest'],
                                opacity=map_topo_egid_specs['point_opacity']
                            ),
                            name = 'house w/o pv',
                            text=subinst3_gdf['hover_text'],
                            hoverinfo='text'
                        ))


                    # topo egid map: add grid nodes ----------------
                    if True:
                        dsonodes_in_topo_gdf = dsonodes_gdf.loc[dsonodes_gdf['grid_node'].isin(pvinst_gdf['grid_node'].unique())].copy()
                        dsonodes_in_topo_gdf.to_crs('EPSG:4326', inplace=True)
                        dsonodes_in_topo_gdf['kW_threshold'] = dsonodes_in_topo_gdf['kVA_threshold'] * self.pvalloc_scen.GRIDspec_perf_factor_1kVA_to_XkW

                        # add color 
                        area_palette= trace_color_dict[map_topo_egid_specs['gridnode_area_palette']]
                        unique_nodes = sorted(pvinst_gdf['grid_node'].unique())
                        n_nodes = len(unique_nodes)
                        n_palette = len(area_palette)
                        palette_indices = np.linspace(0, n_palette - 1, n_nodes, dtype=int)

                        node_color_map = {node: area_palette[i] for node, i in zip(unique_nodes, palette_indices)}
                        pvinst_gdf['color'] = pvinst_gdf['grid_node'].map(node_color_map)
                        dsonodes_in_topo_gdf['color'] = dsonodes_in_topo_gdf['grid_node'].map(node_color_map)
    
                        fig_topoegid.add_trace(
                            go.Scattermapbox(
                                lat=dsonodes_in_topo_gdf.geometry.y,
                                lon=dsonodes_in_topo_gdf.geometry.x,
                                mode='markers+text',  # Add text next to markers
                                marker=dict(
                                    size=map_topo_egid_specs['gridnode_point_size'],
                                    color=dsonodes_in_topo_gdf['color'],
                                    opacity=map_topo_egid_specs['gridnode_point_opacity']
                                ),
                                text=dsonodes_in_topo_gdf['grid_node'],  # Or any other label (e.g. 'id')
                                textposition='top right',  # Position relative to marker
                                textfont=dict(
                                    size=10,  # Adjust font size as needed
                                    color='black'
                                ),
                                customdata=dsonodes_in_topo_gdf[['grid_node', 'kW_threshold']],
                                hovertemplate=(
                                    "Grid Node: %{customdata[0]}<br>" +
                                    "kW Limit: %{customdata[1]}<extra></extra>"
                                ),
                                name='Grid Nodes (Centroid)',
                                showlegend=True
                            )
                        )
                        
                        first_node_hull = True
                        for node in pvinst_gdf['grid_node'].unique():
                            subdf_node_egid = pvinst_gdf.loc[pvinst_gdf['grid_node'] == node].copy()
                            hull_subdf = MultiPoint(subdf_node_egid.geometry.tolist()).convex_hull
                            hull_subdf_gdf = gpd.GeoDataFrame(geometry=[hull_subdf], crs='EPSG:4326')
                            hull_subdf_geojson = json.loads(hull_subdf_gdf.to_json())
                            color = subdf_node_egid["color"].iloc[0]
                                                    
                            hull_trace = px.choropleth_mapbox(
                                hull_subdf_gdf,
                                geojson=hull_subdf_geojson,
                                locations=hull_subdf_gdf.index.astype(str),
                                color_discrete_sequence=[color], 
                                opacity=map_topo_egid_specs['girdnode_egid_opacity'],
                            ).data[0]
                            hull_trace.name = "Grid Node (EGID hull)"
                            hull_trace.legendgroup = "grid_node_area"  # <== GROUPING KEY
                            hull_trace.showlegend = first_node_hull     # Show only on first
                            # hull_trace.visible = True                   # Ensures all toggle together
                            first_node_hull = False
                            
                            fig_topoegid.add_trace(hull_trace)                          
 
                    
                    # Update layout ----------------
                    fig_topoegid.update_layout(
                            title=f"Map of model PV Topology ({scen})",
                            mapbox=dict(
                                style="carto-positron",
                                center={"lat": self.visual_sett.default_map_center[0], "lon": self.visual_sett.default_map_center[1]},  # Center the map on the region
                                zoom=self.visual_sett.default_map_zoom
                            ))


                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_map_topo_egid_incl_gridarea_TF[1]:
                        if self.visual_sett.plot_ind_map_topo_egid_incl_gridarea_TF[2]:
                            fig_topoegid.show()
                        elif not self.visual_sett.plot_ind_map_topo_egid_incl_gridarea_TF[2]:
                            fig_topoegid.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_map_topo_egid_incl_gridarea.html')
                    else:
                        fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_map_topo_egid_incl_gridarea.html')




        # def plot_ind_map_node_connections(self, ): 

        # def plot_ind_map_omitted_egids(self, ): 

        # def plot_ind_lineband_contcharact_newinst(self, ): 




# ======================================================================================================
# RUN VISUALIZATION
# ======================================================================================================


if __name__ == '__main__':
    # if False:

    visualization_list = [
        Visual_Settings(
            # pvalloc_exclude_pattern_list = [
            #     '*.txt','*.xlsx','*.csv','*.parquet',
            #     '*old_vers*', 
            #     'pvalloc_BLsml_10y_f2013_1mc_meth2.2*', 
            #     'pvalloc_mini_2m*',
            #     ], 
            pvalloc_include_pattern_list = [
                'pvalloc_BLsml_test3_2013_30y_16bfs',
                # 'pvalloc_mini_BYMONTH_rnd', 
            ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization = False,  
            ),        

    
    ]

    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        # visual_class.plot_ALL_init_sanitycheck()
        # visual_class.plot_ALL_mcalgorithm()

        # # -- def plot_ALL_init_sanitycheck(self, ): -------------
        # visual_class.plot_ind_var_summary_stats()                     # runs as intended
        # visual_class.plot_ind_hist_pvcapaprod_sanitycheck()           # runs as intended
        # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
        # visual_class.plot_ind_charac_omitted_gwr()                    # runs as intended
        # visual_class.plot_ind_line_meteo_radiation()                  # runs as intended

        # # -- def plot_ALL_mcalgorithm(self,): -------------
        # visual_class.plot_ind_line_installedCap()                     # runs as intended
        visual_class.plot_ind_line_productionHOY_per_node()           # runs as intended
        visual_class.plot_ind_line_productionHOY_per_EGID()           # runs as intended
        # visual_class.plot_ind_line_PVproduction()                     # runs
        # visual_class.plot_ind_hist_NPV_freepartitions()               # runs as intended
        # visual_class.plot_ind_line_gridPremiumHOY_per_node()          # runs 
        # visual_class.plot_ind_line_gridPremium_structure()            # runs 
        visual_class.plot_ind_map_topo_egid()                           # runs as intended
        # visual_class.plot_ind_map_topo_egid_incl_gridarea()            # runs as intended


        # plot_ind_map_node_connections()
        # plot_ind_map_omitted_egids()
        # plot_ind_lineband_contcharact_newinst()


            