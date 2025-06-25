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
                                                    '*testX',
                                                    '*pandas*',
                                                    '*text2_*', 
                                                    '*_BLsml_*', 
                                                    '*_mini_2m_*',
                                                    ])
    pvalloc_include_pattern_list : List[str]      = field(default_factory=lambda: [])
    plot_show: bool                              = True
    save_plot_by_scen_directory: bool            = True
    remove_old_plot_scen_directories: bool       = False
    remove_old_plots_in_visualization: bool      = False
    remove_old_csvs_in_visualization: bool       = False
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
            ('1122', 'Bldg. w three flat (incl double, row houses, w 3 flats)'),
            # ('1274', 'look up meaning'),
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
    plot_ind_line_gridPremiumHOY_per_node_TF: List[bool]    = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_gridPremiumHOY_per_EGID_TF: List[bool]    = field(default_factory=lambda: [True,      True,       False])
    plot_ind_cols_HOYagg_per_EGID_TF: List[bool]  = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_PVproduction_TF: List[bool]               = field(default_factory=lambda: [True,      True,       False])
    plot_ind_line_gridPremium_structure_TF: List[bool]      = field(default_factory=lambda: [True,      True,       False])
    plot_ind_hist_NPV_freepartitions_TF: List[bool]         = field(default_factory=lambda: [True,      True,       False])
    plot_ind_hist_pvcapaprod_TF: List[bool]                 = field(default_factory=lambda: [True,      True,       False])

    plot_ind_map_topo_egid_TF: List[bool]                   = field(default_factory=lambda: [True,      True,       False])
    plot_ind_map_topo_egid_incl_gridarea_TF: List[bool]     = field(default_factory=lambda: [True,      True,       False])
    plot_ind_map_node_connections_TF: List[bool]            = field(default_factory=lambda: [True,      True,       False])
    plot_ind_map_omitted_egids_TF: List[bool]               = field(default_factory=lambda: [True,      True,       False])
    plot_ind_lineband_contcharact_newinst_TF: List[bool]    = field(default_factory=lambda: [True,      True,       False])

    plot_ind_line_productionHOY_per_EGID_specs: Dict         = field(default_factory=lambda: {
        'grid_nodes_counts_minmax': (4,8),         # try to select grid node with nEGIDs for visualization
        'specific_gridnodes_egid_HOY': None,        # select specific gridn node by id number for visualization
        'egid_col_to_plot'          : ['demand_kW', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW' ],
        'grid_col_to_plot_tuples'   : [('netfeedin_kW', 'netfeedin_all_kW'),
                                       ('netfeedin_kW', 'demand_proxy_out_kW'),
                                    #    ('netfeedin_kW', 'netfeedin_all_taken_kW'),
                                    #    ('netfeedin_kW', 'netfeedin_all_loss_kW'),
                                    #    ('netfeedin_kW', 'kW_threshold'),
                                    ],
        'gridnode_pick_col_to_agg':  [  
                                        'demand_kW',
                                        'netfeedin_kW', 
                                        'netfeedin_all_kW', 
                                        # 'netfeedin_all_taken_kW', 
                                        # 'netfeedin_all_loss_kW', 
                                        'kW_threshold', 
                                        'demand_proxy_out_kW'
                                        ], 
        'egid_trace_opacity' : 0.4,
        'grid_trace_opacity' : 0.8,
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

    # old out of use plots -------------------------------------------------------------->  [run plot,  show plot,  show all scen]
    plot_ind_boxp_radiation_rng_sanitycheck_TF: List[bool]  = field(default_factory=lambda: [False,      True,       False])



class Visualization:
    def __init__(self, settings: Visual_Settings):
        self.visual_sett = settings

        # SETUP --------------------
        self.visual_sett.wd_path = os.getcwd()
        self.visual_sett.data_path = os.path.join(self.visual_sett.wd_path, 'data')
        self.visual_sett.visual_path = os.path.join(self.visual_sett.data_path, 'visualization')
        self.visual_sett.log_name = f'{self.visual_sett.visual_path}/visual_log.txt'

        os.makedirs(self.visual_sett.visual_path, exist_ok=True)

        # create a str list of scenarios in pvalloc to visualize (include/exclude by pattern recognition)
        filered_scen_list = os.listdir(f'{self.visual_sett.data_path}/pvalloc')
        
        if not self.visual_sett.pvalloc_include_pattern_list == []:
            filered_scen_list: list[str] = [
                scen for scen in filered_scen_list
                if any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_include_pattern_list)
            ]

        if not self.visual_sett.pvalloc_exclude_pattern_list == []:
            filered_scen_list: list[str] = [
                scen for scen in filered_scen_list 
                if not any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_exclude_pattern_list)
                # if not any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_exclude_pattern_list)
            ]

        self.pvalloc_scen_list = filered_scen_list
        
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

        if self.visual_sett.remove_old_csvs_in_visualization:
            old_files = glob.glob(f'{self.visual_sett.visual_path}/*.csv')
            for file in old_files:
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
        self.plot_ind_cols_HOYagg_per_EGID()
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

                    # import 
                    topo                     = json.load(open(f'{self.visual_sett.data_path}/pvalloc/{scen}/sanity_check_byEGID/topo_egid.json', 'r'))
                    swstore_arch_typ_factors = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/swstore_arch_typ_factors.parquet')                    
                    demandtypes_ts           = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/demandtypes_ts.parquet')

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
                                'GAREA': gwr_info.get('garea'),
                                'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                                'demand_arch_typ': v.get('demand_arch_typ'),
                                'demand_elec_dem_pGAREA': v.get('demand_elec_dem_pGAREA'),
                                'grid_node': v.get('grid_node'),

                                'inst_TF': pv_inst.get('inst_TF'),
                                'info_source': pv_inst.get('info_source'),
                                'pvid': pv_inst.get('xtf_id'),
                                'pv_tarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
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

                    topo_df = pl.DataFrame(rows)

                    demandtypes_mean_df = topo_df.group_by(['demand_arch_typ']).agg([
                        pl.col('GAREA').mean().alias('GAREA_mean'),
                        pl.col('GAREA').median().alias('GAREA_median'),
                    ])                    
                    demandtypes_mean_df = demandtypes_mean_df.with_columns(
                        pl.col('demand_arch_typ').str.extract(r'^(SFH|MFH)', 1).alias('sfhmfh_typ')
                    )

                    
                    # extend by demandtypes_ts + factors
                    swstore_arch_typ_factors = swstore_arch_typ_factors.rename({'arch_typ': 'demand_arch_typ',
                                                                                'elec_dem_ind_cecb': 'demand_elec_pGAREA'})
                    demandtypes_mean_df = demandtypes_mean_df.join(swstore_arch_typ_factors, on='demand_arch_typ', how='left')

                    demandtypes_unpivot = demandtypes_ts.unpivot(
                        on = ['SFH', 'MFH', ],
                        index=['t', 't_int'],  # col that stays unchanged
                        value_name='demand_profile',  # name of the column that will hold the values
                        variable_name='sfhmfh_typ'  # name of the column that will hold the original column names
                    )   
                    demandtypes_mean_df = demandtypes_mean_df.join(demandtypes_unpivot, on='sfhmfh_typ', how='left')

                    demandtypes_mean_df = demandtypes_mean_df.with_columns([
                        (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_mean") ).alias("demand_kW_mean"),  
                        (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_median")).alias("demand_kW_median")
                    ])


                    # aggregate demand per month
                    demandtypes_monthly = demandtypes_ts.with_columns([
                        pl.col("time").str.strptime(pl.Datetime, format="%d.%m.%y %H:%M").alias("parsed_time"),
                        pl.col("time").str.strptime(pl.Datetime, format="%d.%m.%y %H:%M").dt.truncate("1mo").alias("month")
                    ])
                    demandtypes_monthly = demandtypes_monthly.group_by("month").agg([
                        pl.col("MFH").sum().alias("MFH"),
                        pl.col("SFH").sum().alias("SFH"),
                        pl.col("t_int").min().alias("t_int")  # use min or first
                    ])
                    demandtypes_monthly = demandtypes_monthly.sort(['t_int',], descending=False, )  

                    demandtypes_monthly_unpivot = demandtypes_monthly.unpivot(
                        on = ['SFH', 'MFH', ],
                        index=['t_int'],  # col that stays unchanged
                        value_name='demand_profile',  # name of the column that will hold the values
                        variable_name='sfhmfh_typ'  # name of the column that will hold the original column names
                    )   
                    demandtypes_mean_monthly = demandtypes_mean_df[['demand_arch_typ', 'GAREA_mean', 'GAREA_median', 'sfhmfh_typ', 'demand_elec_pGAREA' ]].join(demandtypes_monthly_unpivot, on='sfhmfh_typ', how='left')
                    demandtypes_mean_monthly = demandtypes_mean_monthly.with_columns([
                        (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_mean") ).alias("demand_kW_mean"),  
                        (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_median")).alias("demand_kW_median")
                    ])

                    # plot
                    fig = go.Figure()
                    for i, arch_typ in enumerate(demandtypes_mean_df['demand_arch_typ'].unique()):
                        plotdf = demandtypes_mean_df.filter(pl.col('demand_arch_typ') == arch_typ)
                        plotdf = plotdf.with_columns([
                            pl.col('demand_kW_mean').sum().alias('total_demand_kW_mean'), 
                            pl.col('demand_kW_median').sum().alias('total_demand_kW_median')
                        ])
                        plotdf_monthly = demandtypes_mean_monthly.filter(pl.col('demand_arch_typ') == arch_typ)

                        plotdf = plotdf.to_pandas()
                        plotdf_monthly = plotdf_monthly.to_pandas()

                        fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['total_demand_kW_mean'], name= f'{arch_typ} total annual demand GAREA_mean',
                                                 mode='lines', yaxis = 'y2'))
                        fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['total_demand_kW_median'], name= f'{arch_typ} total annual demand by GAREA_median',
                                                 mode='lines', yaxis = 'y2'))
                        # fig.add_trace(go.Scatter(x=plotdf_monthly['t_int'], y=plotdf_monthly['demand_kW_mean'], name= f'{arch_typ} demand by GAREA_mean (monthly)',
                        #                          mode='lines+markers', yaxis='y2'))
                        # fig.add_trace(go.Scatter(x=plotdf_monthly['t_int'], y=plotdf_monthly['demand_kW_median'], name= f'{arch_typ} demand by GAREA_median (monthly)',
                        #                          mode='lines+markers', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['demand_kW_mean'], name= f'{arch_typ} demand by GAREA_mean', 
                                                 mode='lines', line=dict(dash='dot'), ))
                        fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['demand_kW_median'], name= f'{arch_typ} demand by GAREA_median',
                                                 mode='lines', line=dict(dash='dot'), ))
                        fig.add_trace(go.Scatter(x=[None,], y=[None,], name= '', mode='lines', opacity = 0))
                        
                    fig.update_layout(
                        title=f'Demand Profiles by Demand Arch Type (scen: {scen})',
                        xaxis_title='Time',
                        yaxis_title='Demand [kW]',
                        yaxis2=dict(
                            title='Total Demand [kWh]',
                            overlaying='y',
                            side='right',
                            showgrid=False,
                        ),
                        legend_title = 'Demand Arch Type',
                    )
                    fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen_list[i_scen])
                    fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)


                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_var_summary_stats_TF[1]:
                        if self.visual_sett.plot_ind_var_summary_stats_TF[2]:
                            fig.show()
                        elif not self.visual_sett.plot_ind_var_summary_stats_TF[2]:
                            fig.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_demand_by_arch_typ.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_demand_by_arch_typ.html')
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
                            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html'):
                                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen*.html'))
                                os.rename(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html', 
                                          f'{self.visual_sett.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
                            fig_agg_pmonth.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html')
                            print_to_logfile(f'\texport: plot_agg_line_installedCap__{len(self.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)



        def plot_ind_line_productionHOY_per_node(self, ): 
            if self.visual_sett.plot_ind_line_productionHOY_per_node_TF[0]:

                checkpoint_to_logfile('plot_ind_line_productionHOY_per_node', self.visual_sett.log_name)
                print_to_logfile(f'{self.pvalloc_scen_list}', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import ----------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0] # take first path if multiple apply, so code can still run properlyrly
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    gridnode_df = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
                    gridnode_df = gridnode_df.with_columns([
                        pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                    ])
                    gridnode_df = gridnode_df.sort("t_int", descending=False)

                    # aggregate to total production per HOY
                    gridnode_df = gridnode_df.with_columns([
                        ((pl.col('kW_threshold') - pl.col('netfeedin_all_taken_kW')) / pl.col('kW_threshold')).alias('holding_capacity')
                    ])

                    gridnode_total_df = gridnode_df.group_by(['t', 't_int']).agg([
                        pl.col('netfeedin_all_kW').sum().alias('netfeedin_all_kW'),
                        pl.col('netfeedin_all_taken_kW').sum().alias('netfeedin_all_taken_kW'),
                        pl.col('netfeedin_all_loss_kW').sum().alias('netfeedin_all_loss_kW'),
                        pl.col('kW_threshold').sum().alias('kW_threshold'),
                        pl.col('holding_capacity').mean().alias('holding_capacity_all'),
                    ])
                    gridnode_total_df = gridnode_total_df.sort("t_int", descending=False)

                    # plot ----------------
                    if any([n in self.visual_sett.node_selection_for_plots for n in gridnode_df['grid_node'].unique()]):
                        nodes = [n for n in gridnode_df['grid_node'].unique() if n in self.visual_sett.node_selection_for_plots]
                    elif len(gridnode_df['grid_node'].unique()) > 5: 
                        nodes = gridnode_df['grid_node'].unique()[:5]
                    else:
                        nodes = gridnode_df['grid_node'].unique()


                    fig = go.Figure()
                    gridnode_total_df = gridnode_total_df.to_pandas()
                    gridnode_df = gridnode_df.to_pandas()

                    fig.add_trace(go.Scatter(x=[None, ], y=[None, ], name=f'-- Total aggregation {10*"-"}', opacity = 0, ))
                    fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['netfeedin_all_kW'],       name='Total netfeedin_all_kW',       mode='lines+markers', line=dict(color='black', width=3), marker=dict(symbol='cross')))
                    fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['netfeedin_all_taken_kW'], name='Total netfeedin_all_taken_kW', mode='lines+markers', line=dict(color='green', width=3), marker=dict(symbol='cross')))
                    fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['netfeedin_all_loss_kW'],  name='Total netfeedin_all_loss_kW',  mode='lines+markers', line=dict(color='red',   width=3), marker=dict(symbol='cross')))
                    fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['kW_threshold'],           name='Total kW_threshold',           mode='lines+markers', line=dict(color='blue',  width=3), marker=dict(symbol='cross')))
                    fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['holding_capacity_all'],  name='Total holding_capacity_all',  mode='lines+markers', line=dict(color='purple', width=3), marker=dict(symbol='cross')))
                    fig.add_trace(go.Scatter(x=[None, ], y=[None, ], name='', opacity = 0, ))

                    for node in nodes:
                        filter_df = copy.deepcopy(gridnode_df.loc[gridnode_df['grid_node'] == node])
                        fig.add_trace(go.Scatter(x=[None,], y=[None,], name=f'-- grid_node: {node} {10*"-"}', opacity=0, mode='lines+markers', line=dict(color='black', width=1, dash='dash')))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_kW'], name=f'{node} - demand_kW'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['pvprod_kW'], name=f'{node} - pvprod_kW'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['selfconsum_kW'], name=f'{node} - selfconsum_kW'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_kW'], name=f'{node} - netfeedin_kW'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netdemand_kW'], name=f'{node} - netdemand_kW'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_proxy_out_kW'], name=f'{node} - demand_proxy_out_kW'))

                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_all_kW'], name=f'{node} - feedin_all'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_all_taken_kW'], name= f'{node} - feedin_all_taken'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_all_loss_kW'], name=f'{node} - feedin_all_loss'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['kW_threshold'], name=f'{node} - kW_threshold'))
                        

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

                grid_nodes_counts_minmax    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_nodes_counts_minmax']
                specific_gridnodes_egid_HOY = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['specific_gridnodes_egid_HOY']
                egid_col_to_plot            = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_col_to_plot']
                grid_col_to_plot_tuples     = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_col_to_plot_tuples']
                gridnode_pick_col_to_agg    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['gridnode_pick_col_to_agg']
                egid_trace_opacity          = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_trace_opacity']
                grid_trace_opacity          = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_trace_opacity']

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import --------------------------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                    topo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
                    outtopo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/outtopo_time_subdf/*.parquet')
                    
                    gridnode_df_import = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
                    dsonodes_df = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/dsonodes_df.parquet')  


                    # select EGIDs by EGID - DF_UID combo --------------------------
                    egid_list, dfuid_list, info_source_list, inst_TF_list, grid_node_list = [], [], [], [], []
                    for k,v in topo.items():
                        if v['pv_inst']['inst_TF']:
                            for dfuid_w_inst in v['pv_inst']['df_uid_w_inst']:
                                egid_list.append(k)
                                dfuid_list.append(dfuid_w_inst)
                                info_source_list.append(v['pv_inst']['info_source'])
                                inst_TF_list.append(v['pv_inst']['inst_TF'])   
                                grid_node_list.append(v['grid_node']) 
                        else: 
                            egid_list.append(k)
                            dfuid_list.append('')
                            info_source_list.append('')
                            inst_TF_list.append(False)
                            grid_node_list.append(v['grid_node'])

                    gridnode_freq         = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,  'grid_node': grid_node_list})
                    Map_pvinfo_topo_egid  = pl.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,  'grid_node': grid_node_list})


                    # select gridnodes / EGIDs to plot ----------
                    if specific_gridnodes_egid_HOY != None: 
                        gridnode_pick = specific_gridnodes_egid_HOY
                    else: 
                        gridnodes_counts =  gridnode_freq.groupby(['EGID', ]).agg({
                            'inst_TF': 'first',
                            'info_source': 'first',
                            'grid_node': 'first',})
                        egid_gridnode_counts = gridnodes_counts['grid_node'].value_counts()
                        egid_gridnode_counts.sort_values(ascending=False)
                        gridnode_selec = egid_gridnode_counts[(egid_gridnode_counts >= grid_nodes_counts_minmax[0]) & (egid_gridnode_counts <= grid_nodes_counts_minmax[1]) ]

                        if len(gridnode_selec) == 0:
                            egid_gridnode_counts.sort_values(ascending=True, inplace=True)
                            gridnode_pick = egid_gridnode_counts.index[0]  # take the smallest gridnode
                        else:
                            gridnode_selec.sort_values(ascending=True, inplace=True)
                            gridnode_pick = gridnode_selec.index[0]


                    Map_pvinfo_gridnode = gridnode_freq.loc[gridnode_freq['grid_node'] == gridnode_pick].copy()



                    agg_subdf_df_list, agg_egids_list = [], []
                    for i_path, path in enumerate(topo_subdf_paths):
                        subdf = pl.read_parquet(path)

                        # Only plot EGIDs for 1 node: 
                        subdf = subdf.filter(pl.col('grid_node') == gridnode_pick)  


                        # taken 1:1 from algo_update_gridnode_AND_gridprem_POLARS() =============================================
                        subdf_updated = subdf.clone()                                      
                        subdf_updated = subdf_updated.drop(['info_source', 'inst_TF'])                      

                        subdf_updated = subdf_updated.join(Map_pvinfo_topo_egid[['EGID', 'df_uid', 'info_source', 'inst_TF']], on=['EGID', 'df_uid'], how='left')         
                        # remove the nulls from the merged columns
                        subdf_updated = subdf_updated.with_columns([
                            pl.when(pl.col('inst_TF').is_null())
                                .then(False).otherwise(pl.col('inst_TF')).alias('inst_TF'),
                            pl.when(pl.col('info_source').is_null())
                                .then(pl.lit("")).otherwise(pl.col('info_source')).alias('info_source'),
                        ])

                        Map_pvinst_topo_egid = Map_pvinfo_topo_egid.filter(pl.col('inst_TF'))  # indifferetn => should give same result, Map_pvinfo_topo_egid.filter(pl.col('df_uid') != '')

                        subdf_no_inst = subdf_updated.join(
                            Map_pvinst_topo_egid[['EGID', 'df_uid']], 
                            on=['EGID', 'df_uid'], 
                            how='anti'
                        )
                        subdf_updated = subdf_updated.with_columns([
                            pl.when(
                                (pl.col('EGID').is_in(subdf_no_inst['EGID'])) &
                                (pl.col('df_uid').is_in(subdf_no_inst['df_uid']))
                            ).then(pl.lit(0.0)).otherwise(pl.col('pvprod_kW')).alias('pvprod_kW'),
                        ])


                        selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.pvalloc_scen.TECspec_self_consumption_ifapplicable

                        subdf_updated = subdf_updated.with_columns([        
                            selfconsum_expr.alias("selfconsum_kW"),
                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                        ])
                        
                        # necessary, not too exagerate demand per gridnode
                        agg_egids = subdf_updated.group_by(['EGID', 't']).agg([
                            pl.col('grid_node').first().alias('grid_node'),
                            pl.col('demand_kW').first().alias('demand_kW'),
                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                            pl.col('netdemand_kW').sum().alias('netdemand_kW')
                        ])

                        # only select egids for grid_node mentioned above
                        agg_egids = agg_egids.filter(pl.col('EGID').is_in(Map_pvinfo_gridnode['EGID'].to_list()))
                        agg_egids_list.append(agg_egids)
                        # -----

                        agg_subdf = agg_egids.group_by(["grid_node", "t"]).agg([
                            pl.col('demand_kW').sum().alias('demand_kW'),
                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                            pl.col('netdemand_kW').sum().alias('netdemand_kW')
                        ])

                        agg_subdf_df_list.append(agg_subdf)

                    
                    agg_subdf_df = pl.concat(agg_subdf_df_list)
                    topo_gridnode_df = agg_subdf_df.group_by(['grid_node', 't']).agg([
                        pl.col('demand_kW').first().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                    ])

                    # MAIN DF of this plot, all feedin TS by EGID ------------
                    topo_agg_egids_df = pl.concat(agg_egids_list)
                    topo_agg_egids_df = topo_agg_egids_df.with_columns([
                        pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                    ])
                    topo_agg_egids_df = topo_agg_egids_df.sort("t_int", descending=False)
                    # -----


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


                    # build gridnode_df -----------------------------------------------------
                    gridnode_df = topo_gridnode_df.join(outtopo_gridnode_df, on=['grid_node', 't'], how='left')
                    gridnode_df = gridnode_df.with_columns([
                        pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                    ])
                    
                    demand_proxy_out_kW = gridnode_df['demand_proxy_out_kW'].fill_null(0)
                    # netdemand_kW = gridnode_df['netdemand_kW'].fill_null(0)
                    netfeedin_kW = gridnode_df['netfeedin_kW'].fill_null(0)
                    gridnode_df = gridnode_df.with_columns([
                        (netfeedin_kW - demand_proxy_out_kW).alias('netfeedin_all_kW'),
                    ])

                    # sanity check
                    gridnode_df.group_by(['grid_node',]).agg([pl.len()])
                    gridnode_df.group_by(['t',]).agg([pl.len()])


                    # attach node thresholds 
                    gridnode_df = gridnode_df.join(dsonodes_df[['grid_node', 'kVA_threshold']], on='grid_node', how='left')
                    gridnode_df = gridnode_df.with_columns((pl.col("kVA_threshold") * self.pvalloc_scen.GRIDspec_perf_factor_1kVA_to_XkW).alias("kW_threshold"))
                    
                    gridnode_df = gridnode_df.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") < 0)
                        .then(0.0)
                        .otherwise(pl.col("netfeedin_all_kW"))
                        .alias("netfeedin_all_kW"),
                        ])
                    gridnode_df = gridnode_df.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") > pl.col("kW_threshold"))
                        .then(pl.col("kW_threshold"))
                        .otherwise(pl.col("netfeedin_all_kW"))
                        .alias("netfeedin_all_taken_kW"),
                        ])
                    gridnode_df = gridnode_df.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") > pl.col("kW_threshold"))
                        .then(pl.col("netfeedin_all_kW") - pl.col("kW_threshold"))
                        .otherwise(0.0)
                        .alias("netfeedin_all_loss_kW")
                    ])

                    # end 1:1 copy from pvallocation =============================================
                    gridnode_pick_df = gridnode_df.clone()


                        
                    # plot --------------------------
                    fig_demand, fig_pvprod, fig_selfconsum, fig_netdemand, fig_netfeedin, fig_netfeedin_all = go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
                    fig_dict = {
                    'fig_demand_kW':     fig_demand     ,
                    'fig_pvprod_kW':     fig_pvprod     ,
                    'fig_selfconsum_kW': fig_selfconsum ,
                    'fig_netdemand_kW':  fig_netdemand  ,
                    'fig_netfeedin_kW':  fig_netfeedin  ,
                    'fig_netfeedin_ALL_kW':  fig_netfeedin_all  ,

                    }
                    gridnode_pick_df = gridnode_pick_df.to_pandas()
                    topo_agg_egids_df = topo_agg_egids_df.to_pandas()

                    # Stack traces EGIDs in picked gridnode ----------
                    for col in egid_col_to_plot:
                        fig_sub = fig_dict[f'fig_{col}']

                        for egid in topo_agg_egids_df['EGID'].unique():
                            stack_subdf = topo_agg_egids_df.loc[topo_agg_egids_df['EGID'] == egid]
                            stack_subdf = stack_subdf.sort_values(by=['t_int'])
                            fig_sub.add_trace(go.Scatter(x=stack_subdf['t_int'], y=stack_subdf[col],
                                                        mode='lines', 
                                                        stackgroup= col, 
                                                        name=f'EGID: {egid} - {col}', 
                                                        hoverinfo='skip',
                                                        opacity=egid_trace_opacity
                                                        ))
                        fig_sub.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'topo_agg_egids_df from subdf {20*"-"}', opacity=0)) 


                    # gridnode_pick_df for comparison ----------
                    for col in egid_col_to_plot:
                        fig_sub = fig_dict[f'fig_{col}']

                        for node in gridnode_pick_df['grid_node'].unique():
                            grid_col_agg_tuples = [t[0] for t in grid_col_to_plot_tuples]
                            node_subdf = gridnode_pick_df.loc[gridnode_pick_df['grid_node'] == node]
                            node_subdf = node_subdf.sort_values(by=['t_int'])   
                            
                            # add aggregated traces 
                            if col in grid_col_agg_tuples:
                                agg_columns = [t[1] for t in grid_col_to_plot_tuples if t[0] == col]
                                for agg_col in agg_columns:
                                    fig_sub.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[agg_col],
                                                                name=f'grid_node: {node} - {agg_col} (agg)', 
                                                                mode='lines+markers', 
                                                                marker=dict(symbol='cross', ), 
                                                                hoverinfo='skip',
                                                                opacity=grid_trace_opacity
                                                                ))
                            # add col trace
                            fig_sub.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[col],
                                mode='lines+markers', name=f'grid_node: {node} - {col}', 
                                marker=dict(symbol='cross', ), 
                                hoverinfo='skip',
                                opacity=grid_trace_opacity
                                ))
                        fig_sub.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'gridnode_pick_df from subdf {20*"-"}', opacity=0))
                                                                

                    # # Sanity Check: picked node from gridnode_df +  ----------
                    # # add values form gridnode_df to make in-plot sanity check
                    for col in egid_col_to_plot:
                        fig_sub = fig_dict[f'fig_{col}']

                        for node in gridnode_pick_df['grid_node'].unique():
                            grid_col_agg_tuples = [t[0] for t in grid_col_to_plot_tuples]
                            gridnode_subdf = gridnode_df_import.filter(pl.col('grid_node') == node)
                            gridnode_subdf = gridnode_subdf.sort("t_int", descending=False)
                            gridnode_subdf = gridnode_subdf.to_pandas()

                            if col in grid_col_agg_tuples:
                                agg_columns = [t[1] for t in grid_col_to_plot_tuples if t[0] == col]
                                for agg_col in agg_columns:
                                    fig_sub.add_trace(go.Scatter(x=gridnode_subdf['t_int'], y=gridnode_subdf[agg_col],
                                                name=f'grid_node: {node} - {agg_col} (gridnode_df)',
                                                mode='lines+markers', 
                                                marker=dict(symbol='cross', ), 
                                                hoverinfo='skip',
                                                opacity=grid_trace_opacity
                                                ))
                            fig_sub.add_trace(go.Scatter(x=gridnode_subdf['t_int'], y=gridnode_subdf[col],
                                                    name=f'grid_node: {node} - {col} (gridnode_df)',
                                                    mode='lines+markers', 
                                                    hoverinfo='skip',
                                                    opacity=grid_trace_opacity
                                                    ))
                        fig_sub.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'picked node from gridnode_df {20*"-"}', opacity=0))
                    # ----------


                    # plot fig_netfeedin_ALL ----------
                    gridnode_pick_df['netfeedin_all_kW+demand_proxy_out_kW'] = gridnode_pick_df['netfeedin_all_kW'] + gridnode_pick_df['demand_proxy_out_kW']

                    for node in gridnode_pick_df['grid_node'].unique():
                        node_subdf = gridnode_pick_df.loc[gridnode_pick_df['grid_node'] == node]
                        node_subdf = node_subdf.sort_values(by=['t_int'])   

                        fig_netfeedin_all.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'gridnode_pick_df from subdf {20*"-"}', opacity=0))
                        for agg_col in gridnode_pick_col_to_agg:    
                            fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[agg_col],           name= f'grid_node: {node} - {agg_col} (agg subdf)',             mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity))
                        # fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['netfeedin_kW'],           name= f'grid_node: {node} - netfeedin_kW (agg subdf)',             mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity))
                        # fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['netfeedin_all_kW'],       name= f'grid_node: {node} - netfeedin_all_kW (agg subdf)',         mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity, stackgroup='netfeedin_all_kW'))
                        # fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['demand_proxy_out_kW' ],   name= f'grid_node: {node} - demand_proxy_out_kW (agg subdf)',      mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity, stackgroup='netfeedin_all_kW'))
                        # fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['netfeedin_all_taken_kW'], name= f'grid_node: {node} - netfeedin_all_taken_kW (agg subdf)',   mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity))
                        # fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['netfeedin_all_loss_kW'],  name= f'grid_node: {node} - netfeedin_all_loss_kW (agg subdf)',    mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity))
                        # fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['kW_threshold'],           name= f'grid_node: {node} - kW_threshold (agg subdf)',             mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity))
                        fig_netfeedin_all.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf['netfeedin_all_kW+demand_proxy_out_kW'],           name= f'grid_node: {node} - netfeedin_all_kW+demand_proxy_out_kW (agg subdf)',             mode = 'lines+markers', hoverinfo = 'skip', opacity= grid_trace_opacity))
                        
                        
                        gridnode_subdf = gridnode_df_import.filter(pl.col('grid_node') == node)
                        gridnode_subdf = gridnode_subdf.sort("t_int", descending=False)
                        gridnode_subdf = gridnode_subdf.to_pandas()

                        fig_netfeedin_all.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'picked node from gridnode_df {20*"-"}', opacity=0))
                        for agg_col in gridnode_pick_col_to_agg:    
                            fig_netfeedin_all.add_trace(go.Scatter(x=gridnode_subdf['t_int'], y=gridnode_subdf[col],
                                name=f'grid_node: {node} - {agg_col} (gridnode_df)',
                                mode='lines+markers', 
                                hoverinfo='skip',
                                opacity=grid_trace_opacity
                                ))  


                    # export plot --------------------------
                    for k,v in fig_dict.items():
                        fig_sub = v
                        col_name = k.split('fig_')[1]

                        fig_sub.update_layout(
                            title = f'Production per EGID / Grid Node (col: {col_name}, Hour of Year)', 
                            xaxis_title='Hour of Year',
                            yaxis_title='Production / Feedin (kW)',
                            legend_title='EGID / Node ID',
                            template='plotly_white',
                            showlegend=True,
                        )
                        fig_sub = self.add_scen_name_to_plot(fig_sub, scen, self.pvalloc_scen)
                        fig_sub = self.set_default_fig_zoom_hour(fig_sub, self.visual_sett.default_zoom_hour)

                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[1]:
                            if self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[2]:
                                fig_sub.show()
                            elif not self.visual_sett.plot_ind_line_productionHOY_per_EGID_TF[2]:
                                fig_sub.show() if i_scen == 0 else None
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig_sub.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_line_productionHOY_per_EGID_inclEGID_{col_name}.html')
                        else:
                            fig_sub.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_line_productionHOY_per_EGID_inclEGID_{col_name}.html')
                    
                    print_to_logfile(f'\texport: plot_ind_line_productionHOY_per_EGID.html (for: {scen})', self.visual_sett.log_name)



        def plot_ind_cols_HOYagg_per_EGID(self, ):
            if self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[0]:

                checkpoint_to_logfile('plot_ind_cols_HOYagg_per_EGID', self.visual_sett.log_name)

                grid_nodes_counts_minmax    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_nodes_counts_minmax']
                specific_gridnodes_egid_HOY = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['specific_gridnodes_egid_HOY']
                egid_col_to_plot            = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_col_to_plot']
                grid_col_to_plot_tuples     = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_col_to_plot_tuples']
                gridnode_pick_col_to_agg    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['gridnode_pick_col_to_agg']
                egid_trace_opacity          = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_trace_opacity']
                grid_trace_opacity          = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_trace_opacity']

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import --------------------------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                    topo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
                    outtopo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/outtopo_time_subdf/*.parquet')
                    
                    gridnode_df_import = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
                    dsonodes_df = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/dsonodes_df.parquet')  


                    # select EGIDs by EGID - DF_UID combo --------------------------
                    egid_list, dfuid_list, info_source_list, inst_TF_list, grid_node_list = [], [], [], [], []
                    for k,v in topo.items():
                        if v['pv_inst']['inst_TF']:
                            for dfuid_w_inst in v['pv_inst']['df_uid_w_inst']:
                                egid_list.append(k)
                                dfuid_list.append(dfuid_w_inst)
                                info_source_list.append(v['pv_inst']['info_source'])
                                inst_TF_list.append(v['pv_inst']['inst_TF'])   
                                grid_node_list.append(v['grid_node']) 
                        else: 
                            egid_list.append(k)
                            dfuid_list.append('')
                            info_source_list.append('')
                            inst_TF_list.append(False)
                            grid_node_list.append(v['grid_node'])

                    gridnode_freq         = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,  'grid_node': grid_node_list})
                    Map_pvinfo_topo_egid  = pl.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,  'grid_node': grid_node_list})


                    # select gridnodes / EGIDs to plot ----------
                    if specific_gridnodes_egid_HOY != None: 
                        gridnode_pick = specific_gridnodes_egid_HOY
                    else: 
                        gridnodes_counts =  gridnode_freq.groupby(['EGID', ]).agg({
                            'inst_TF': 'first',
                            'info_source': 'first',
                            'grid_node': 'first',})
                        egid_gridnode_counts = gridnodes_counts['grid_node'].value_counts()
                        egid_gridnode_counts.sort_values(ascending=False)
                        gridnode_selec = egid_gridnode_counts[(egid_gridnode_counts >= grid_nodes_counts_minmax[0]) & (egid_gridnode_counts <= grid_nodes_counts_minmax[1]) ]

                        if len(gridnode_selec) == 0:
                            egid_gridnode_counts.sort_values(ascending=True, inplace=True)
                            gridnode_pick = egid_gridnode_counts.index[0]  # take the smallest gridnode
                        else:
                            gridnode_selec.sort_values(ascending=True, inplace=True)
                            gridnode_pick = gridnode_selec.index[0]


                    Map_pvinfo_gridnode = gridnode_freq.loc[gridnode_freq['grid_node'] == gridnode_pick].copy()



                    agg_subdf_df_list, agg_egids_list = [], []
                    for i_path, path in enumerate(topo_subdf_paths):
                        subdf = pl.read_parquet(path)

                        # Only plot EGIDs for 1 node: 
                        # subdf = subdf.filter(pl.col('grid_node') == gridnode_pick)  


                        # taken 1:1 from algo_update_gridnode_AND_gridprem_POLARS() =============================================
                        subdf_updated = subdf.clone()                                      
                        subdf_updated = subdf_updated.drop(['info_source', 'inst_TF'])                      

                        subdf_updated = subdf_updated.join(Map_pvinfo_topo_egid[['EGID', 'df_uid', 'info_source', 'inst_TF']], on=['EGID', 'df_uid'], how='left')         
                        # remove the nulls from the merged columns
                        subdf_updated = subdf_updated.with_columns([
                            pl.when(pl.col('inst_TF').is_null())
                                .then(False).otherwise(pl.col('inst_TF')).alias('inst_TF'),
                            pl.when(pl.col('info_source').is_null())
                                .then(pl.lit("")).otherwise(pl.col('info_source')).alias('info_source'),
                        ])

                        Map_pvinst_topo_egid = Map_pvinfo_topo_egid.filter(pl.col('inst_TF'))  # indifferetn => should give same result, Map_pvinfo_topo_egid.filter(pl.col('df_uid') != '')

                        subdf_no_inst = subdf_updated.join(
                            Map_pvinst_topo_egid[['EGID', 'df_uid']], 
                            on=['EGID', 'df_uid'], 
                            how='anti'
                        )
                        subdf_updated = subdf_updated.with_columns([
                            pl.when(
                                (pl.col('EGID').is_in(subdf_no_inst['EGID'])) &
                                (pl.col('df_uid').is_in(subdf_no_inst['df_uid']))
                            ).then(pl.lit(0.0)).otherwise(pl.col('pvprod_kW')).alias('pvprod_kW'),
                        ])


                        selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.pvalloc_scen.TECspec_self_consumption_ifapplicable

                        subdf_updated = subdf_updated.with_columns([        
                            selfconsum_expr.alias("selfconsum_kW"),
                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                        ])
                        
                        # necessary, not too exagerate demand per gridnode
                        agg_egids = subdf_updated.group_by(['EGID', 't']).agg([
                            pl.col('grid_node').first().alias('grid_node'),
                            pl.col('demand_kW').first().alias('demand_kW'),
                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                            pl.col('netdemand_kW').sum().alias('netdemand_kW')
                        ])

                        # only select egids for grid_node mentioned above
                        # agg_egids = agg_egids.filter(pl.col('EGID').is_in(Map_pvinfo_gridnode['EGID'].to_list()))
                        agg_egids_list.append(agg_egids)
                        # -----

                        agg_subdf = agg_egids.group_by(["grid_node", "t"]).agg([
                            pl.col('demand_kW').sum().alias('demand_kW'),
                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                            pl.col('netdemand_kW').sum().alias('netdemand_kW')
                        ])

                        agg_subdf_df_list.append(agg_subdf)

                    
                    agg_subdf_df = pl.concat(agg_subdf_df_list)
                    topo_gridnode_df = agg_subdf_df.group_by(['grid_node', 't']).agg([
                        pl.col('demand_kW').first().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                    ])

                    # MAIN DF of this plot, all feedin TS by EGID ------------
                    topo_agg_egids_df = pl.concat(agg_egids_list)
                    topo_agg_egids_df = topo_agg_egids_df.with_columns([
                        pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                    ])
                    topo_agg_egids_df = topo_agg_egids_df.sort("t_int", descending=False)
                    # -----


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


                    # build gridnode_df -----------------------------------------------------
                    gridnode_df = topo_gridnode_df.join(outtopo_gridnode_df, on=['grid_node', 't'], how='left')
                    gridnode_df = gridnode_df.with_columns([
                        pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                    ])
                    
                    demand_proxy_out_kW = gridnode_df['demand_proxy_out_kW'].fill_null(0)
                    # netdemand_kW = gridnode_df['netdemand_kW'].fill_null(0)
                    netfeedin_kW = gridnode_df['netfeedin_kW'].fill_null(0)
                    gridnode_df = gridnode_df.with_columns([
                        (netfeedin_kW - demand_proxy_out_kW).alias('netfeedin_all_kW'),
                    ])

                    # sanity check
                    gridnode_df.group_by(['grid_node',]).agg([pl.len()])
                    gridnode_df.group_by(['t',]).agg([pl.len()])


                    # attach node thresholds 
                    gridnode_df = gridnode_df.join(dsonodes_df[['grid_node', 'kVA_threshold']], on='grid_node', how='left')
                    gridnode_df = gridnode_df.with_columns((pl.col("kVA_threshold") * self.pvalloc_scen.GRIDspec_perf_factor_1kVA_to_XkW).alias("kW_threshold"))
                    
                    gridnode_df = gridnode_df.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") < 0)
                        .then(0.0)
                        .otherwise(pl.col("netfeedin_all_kW"))
                        .alias("netfeedin_all_kW"),
                        ])
                    gridnode_df = gridnode_df.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") > pl.col("kW_threshold"))
                        .then(pl.col("kW_threshold"))
                        .otherwise(pl.col("netfeedin_all_kW"))
                        .alias("netfeedin_all_taken_kW"),
                        ])
                    gridnode_df = gridnode_df.with_columns([
                        pl.when(pl.col("netfeedin_all_kW") > pl.col("kW_threshold"))
                        .then(pl.col("netfeedin_all_kW") - pl.col("kW_threshold"))
                        .otherwise(0.0)
                        .alias("netfeedin_all_loss_kW")
                    ])

                    # end 1:1 copy from pvallocation =============================================
                    gridnode_pick_df = gridnode_df.clone()


                    gridnode_pick_df = gridnode_pick_df.to_pandas()
                    topo_agg_egids_df = topo_agg_egids_df.to_pandas()

                    # plot line --------------------------
                    if True:
                        fig = go.Figure()

                        # subset topo_agg_egids_df for line plot
                        topo_agg_egids_gridnode_pick_df = topo_agg_egids_df.loc[topo_agg_egids_df['grid_node'] == gridnode_pick]


                        # stack egid traces for comparison ----------
                        selfconsumHOY_cols_to_plot = ['selfconsum_kW', ]
                        for col in selfconsumHOY_cols_to_plot:
                            for egid in topo_agg_egids_gridnode_pick_df['EGID'].unique():
                                stack_subdf = topo_agg_egids_gridnode_pick_df.loc[topo_agg_egids_gridnode_pick_df['EGID'] == egid]
                                stack_subdf = stack_subdf.sort_values(by=['t_int'])
                                fig.add_trace(go.Scatter(x=stack_subdf['t_int'], y=stack_subdf[col],
                                                            mode='lines', 
                                                            stackgroup= col, 
                                                            name=f'EGID: {egid} - {col}', 
                                                            hoverinfo='skip',
                                                            opacity=egid_trace_opacity
                                                            ))
                            fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'topo_agg_egids_df from subdf {20*"-"}', opacity=0)) 


                        # aggregated traces per month ----------
                        for col in selfconsumHOY_cols_to_plot:
                            for egid in topo_agg_egids_gridnode_pick_df['EGID'].unique():
                                dated_subdf =  topo_agg_egids_gridnode_pick_df.loc[topo_agg_egids_gridnode_pick_df['EGID'] == egid].copy()
                                start_year = pd.Timestamp(f'{self.pvalloc_scen.T0_year_prediction}-01-01')
                                dated_subdf['datetime'] =start_year + pd.to_timedelta(dated_subdf['t_int'] -1 , unit ='h' )
                                dated_subdf['month'] = dated_subdf['datetime'].dt.month

                                subdf_agg = dated_subdf.groupby(['EGID', 'month', ]).agg({
                                    't': 'first', 
                                    't_int': 'first',
                                    col: 'sum',
                                })

                                fig.add_trace(go.Scatter(x=subdf_agg['t_int'], y=subdf_agg[col],
                                                            mode='lines', 
                                                            name=f'EGID: {egid} - {col} (agg month)', 
                                                            hoverinfo='skip',       
                                                            opacity=egid_trace_opacity
                                                            ))
                            fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'agg month from subdf {20*"-"}', opacity=0))
                                                        

                        # aggregated traces per year ----------
                        for col in selfconsumHOY_cols_to_plot:
                            for node in gridnode_pick_df['grid_node'].unique():
                                dated_subdf =  topo_agg_egids_gridnode_pick_df.loc[topo_agg_egids_gridnode_pick_df['EGID'] == egid].copy()
                                start_year = pd.Timestamp(f'{self.pvalloc_scen.T0_year_prediction}-01-01')
                                dated_subdf['datetime'] =start_year + pd.to_timedelta(dated_subdf['t_int'] -1 , unit ='h' )
                                dated_subdf['year'] = dated_subdf['datetime'].dt.year

                                subdf_agg = dated_subdf.groupby(['EGID', 'year', ]).agg({
                                    't': 'first', 
                                    't_int': 'first',
                                    col: 'sum',
                                })
                                xarray = [min(topo_agg_egids_gridnode_pick_df['t_int']), max(topo_agg_egids_gridnode_pick_df['t_int'])]
                                yarray = [subdf_agg[col].values[0], subdf_agg[col].values[0]]
                                fig.add_trace(go.Scatter(x=xarray, y=yarray,
                                                            mode='lines', 
                                                            name=f'node: {node} - {col} (agg year)', 
                                                            hoverinfo='skip',       
                                                            opacity=egid_trace_opacity
                                                            ))
                            fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'agg month from subdf {20*"-"}', opacity=0))
                                        
                        # export plot 
                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[1]:
                            if self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[2]:
                                fig.show()
                            elif not self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[2]:
                                fig.show() if i_scen == 0 else None
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_cols_HOYagg_per_EGID.html')
                        else:
                            fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_cols_HOYagg_per_EGID.html')



                    # plot hist --------------------------
                    if True:
                        fig_hist = go.Figure()

                        # subset topo_agg_egids_df for line plot
                        topo_agg_egids_df
                        topo_agg_egids_gridnode_pick_df = topo_agg_egids_df.loc[topo_agg_egids_df['grid_node'] == gridnode_pick]

                        # aggregate subset + all cols by EGID over year
                        topo_agg_egids_hist = topo_agg_egids_df.groupby(['EGID', 'grid_node']).agg({
                            'demand_kW'     : 'sum' ,
                            'pvprod_kW'     : 'sum' ,
                            'selfconsum_kW' : 'sum' ,
                            'netdemand_kW'  : 'sum' ,
                            'netfeedin_kW'  : 'sum' ,
                        }).reset_index()

                        topo_agg_egids_gridnode_pick_df_hist = topo_agg_egids_gridnode_pick_df.groupby(['EGID', 'grid_node']).agg({
                            'demand_kW'     : 'sum' ,
                            'pvprod_kW'     : 'sum' ,
                            'selfconsum_kW' : 'sum' ,
                            'netdemand_kW'  : 'sum' ,
                            'netfeedin_kW'  : 'sum' ,
                        }).reset_index()

                        # plot hist traces
                        hist_cols_to_plot = [
                            'demand_kW', 
                            'pvprod_kW', 
                            'selfconsum_kW', 
                            'netdemand_kW', 
                            'netfeedin_kW', ]
        

                        # adjust binsize in standardized way
                        for col in hist_cols_to_plot:
                            global_min = min(topo_agg_egids_gridnode_pick_df_hist[col].min(), topo_agg_egids_hist[col].min())
                            global_max = max(topo_agg_egids_gridnode_pick_df_hist[col].max(), topo_agg_egids_hist[col].max())
                            bin_size = (global_max - global_min) / 50  # 50 bins for example
                            xbins_config = dict(
                                start=global_min,
                                end=global_max,
                                size=bin_size
                            )

                        # subset EGID hist traces
                        fig_hist.add_trace(go.Histogram(
                            x=[None, ],
                            name=f'EGIDs in gridnode_pick_df {20*"-"}',
                            opacity=0,
                        ))

                        for node in topo_agg_egids_gridnode_pick_df_hist['grid_node'].unique():
                            node_subdf = topo_agg_egids_gridnode_pick_df_hist.loc[topo_agg_egids_gridnode_pick_df_hist['grid_node'] == node]
                            
                            for col in hist_cols_to_plot:
                                fig_hist.add_trace(go.Histogram(
                                        x=node_subdf[col],
                                        name=f'{col} select grid_node {node}',
                                        histnorm='percent',
                                        opacity=0.75,
                                        xbins=xbins_config
                                ))

                        # all EGID hist traces
                        fig_hist.add_trace(go.Histogram(
                            x=[None, ],
                            name=f'All EGIDs in topo  {20*"-"}',
                            opacity=0,
                        ))

                        for col in hist_cols_to_plot:
                            fig_hist.add_trace(go.Histogram(
                                x=topo_agg_egids_hist[col],
                                name=f'{col} all EGIDs in topo',
                                histnorm='percent',
                                opacity=0.75,
                                xbins=xbins_config
                            ))

                        fig_hist.update_layout(
                            title=f'Histogramm Annual Numbers per EGID', 
                            xaxis_title='Total (sum kWh per year)',
                            yaxis_title='Frequency',
                            barmode='overlay',
                            template='plotly_white',
                        )

                    # export plot --------------------------
                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[1]:
                        if self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[2]:
                            fig_hist.show()
                        elif not self.visual_sett.plot_ind_cols_HOYagg_per_EGID_TF[2]:
                            fig_hist.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_hist.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_hist_selfconsumHOYagg_per_EGID.html')
                    else:
                        fig_hist.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_hist_selfconsumHOYagg_per_EGID.html')

               

        def plot_ind_line_PVproduction(self, ): 
            if self.visual_sett.plot_ind_line_PVproduction_TF[0]:

                checkpoint_to_logfile('plot_ind_line_PVproduction', self.visual_sett.log_name)

                trace_color_dict = {
                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
                    'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
                    'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                    'Rainbow': pc.sequential.Rainbow, 
                }     

                fig_agg_color_palettes = ['Oranges', 'Purples', 'Mint', 'Greys', 'Blues', 'Greens', 'Reds',  ]
                fig_agg = go.Figure()

                for i_scen, scen in enumerate(self.pvalloc_scen_list):

                    # setup + import --------------------------
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                    self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                    gridnode_df_paths = glob.glob(f'{self.visual_sett.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_*.parquet')
                    trange_prediction = pd.read_parquet(f'{self.visual_sett.mc_data_path}/trange_prediction.parquet')
                    topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                    
                    # TOBE DELETED EVENTUALLY .................
                    if 'n_iter' not in trange_prediction.columns:
                        trange_prediction['n_iter'] = trange_prediction.index+1
                    # .................


                    # girdnode_df: transform and prep ------------------
                    gridnode_df_by_iter_list = []
                    for path in gridnode_df_paths:
                        n_iter = int(path.split('gridnode_df_')[1].split('.parquet')[0])
                        
                        # taken 1:1 from plot_ind_line_productionHOY_per_node() ==========
                        gridnode_df = pl.read_parquet(path)
                        gridnode_df = gridnode_df.with_columns([
                            pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                        ])
                        gridnode_df = gridnode_df.sort("t_int", descending=False)

                        # calc holding capacity
                        gridnode_df = gridnode_df.with_columns([
                            ((pl.col('kW_threshold') - pl.col('netfeedin_all_taken_kW')) / pl.col('kW_threshold')).alias('holding_capacity')
                        ])

                        # aggregate to total production per HOY
                        gridnode_total_df = gridnode_df.group_by(['t', 't_int']).agg([
                            pl.col('netfeedin_all_kW').sum().alias('netfeedin_all_kW'),
                            pl.col('netfeedin_all_taken_kW').sum().alias('netfeedin_all_taken_kW'),
                            pl.col('netfeedin_all_loss_kW').sum().alias('netfeedin_all_loss_kW'),
                            pl.col('kW_threshold').sum().alias('kW_threshold'),

                            pl.col('holding_capacity').min().alias('holding_capacity_min'),
                            pl.col('holding_capacity').max().alias('holding_capacity_max'),
                            pl.col('holding_capacity').mean().alias('holding_capacity_mean'),
                        ])
                        gridnode_total_df = gridnode_total_df.sort("t_int", descending=False)
                        # ==========

                        agg_gridnode_row = gridnode_total_df.select([
                            pl.sum('netfeedin_all_kW').alias('netfeedin_all_kW'), 
                            pl.sum('netfeedin_all_taken_kW').alias('netfeedin_all_taken_kW'), 
                            pl.sum('netfeedin_all_loss_kW').alias('netfeedin_all_loss_kW'), 
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

                        gridnode_df_by_iter_list.append(agg_gridnode_row)

                    gridnode_df_by_iter = pl.concat(gridnode_df_by_iter_list, how="vertical")
                    gridnode_df_by_iter = gridnode_df_by_iter.with_columns([
                        (pl.col('netfeedin_all_loss_kW') / pl.col('netfeedin_all_taken_kW')).alias('netfeedin_loss_ratio')
                    ])
                    gridnode_df_by_iter = gridnode_df_by_iter.sort("n_iter", descending=False)


                    # topo_df: transform and prep ------------------
                    egid_list, dfuid_list, info_source_list, inst_TF_list, grid_node_list, BeginOp_list = [], [], [], [], [], []
                    for k, v in topo.items():
                        egid_list.append(k)
                        dfuid_list.append(v['pv_inst']['df_uid_w_inst'])
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
                    
                    topo_df_iter['n_EGID_pvinst_cum'] = topo_df_iter['n_EGID_pvinst'].cumsum()
                    topo_df_iter['ratio_EGID_pvinst'] = topo_df_iter['n_EGID_pvinst_cum'] / len(topo)
                        


                    # plot ind line ----------------
                    fig = go.Figure()
                    gridnode_total_df = gridnode_total_df.to_pandas()

                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_all_kW'],       name='Total netfeedin_all_kW',       mode='lines+markers',  line=dict(color='black',), marker=dict(symbol='cross',), ))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_all_taken_kW'], name='Total netfeedin_all_taken_kW', mode='lines+markers',  line=dict(color='green',), marker=dict(symbol='cross',), ))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_all_loss_kW'],  name='Total netfeedin_all_loss_kW',  mode='lines+markers',  line=dict(color='red',  ),   marker=dict(symbol='cross',), ))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['kW_threshold'],           name='Total kW_threshold',           mode='lines+markers',  line=dict(color='blue', ),  marker=dict(symbol='cross',),  ))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_loss_ratio'],   name='netfeedin_loss_ratio',        mode='lines+markers',  line=dict(color='orange', dash = 'dot'), opacity = 1     , yaxis = 'y2'))

                    fig.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['ratio_EGID_pvinst'],             name='ratio pvinst cumulative',     mode='lines+markers',  line=dict(color='purple',)                               , yaxis ='y2'))
                    fig.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['n_EGID_pvinst'],                 name='n pvinst insample',           mode='lines+markers',  line=dict(color='purple', dash = 'dot'), opacity = 0.5   , ))
                    fig.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['n_EGID_pvinst_cum'],             name='n pvinst cumulative',         mode='lines+markers',  line=dict(color='purple', dash = 'dash'),opacity = 0.8   , ))
                    
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_max_abs'],       name='Holding Capac max abs',      mode='lines+markers',  line=dict(color='goldenrod', dash = 'dot'),  opacity = 0.5, yaxis = 'y2'))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_max_of_mean'],   name='Holding Capac max of mean',  mode='lines+markers',  line=dict(color='goldenrod', dash = 'dash'), opacity = 0.8, yaxis = 'y2'))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_mean_of_mean'],  name='Holding Capac mean of mean', mode='lines+markers',  line=dict(color='goldenrod', ),              opacity = 1.0, yaxis = 'y2'))
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_min_of_mean'],   name='Holding Capac min of mean',  mode='lines+markers',  line=dict(color='goldenrod', dash = 'dash'), opacity = 0.8, yaxis = 'y2'))     
                    fig.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_min_abs'],       name='Holding Capac min abs',      mode='lines+markers',  line=dict(color='goldenrod', dash = 'dot'),  opacity = 0.5, yaxis = 'y2'))

                    fig.update_layout(
                        xaxis_title='Time',
                        yaxis_title='Energy (kWh Feedin / Loss)',
                        yaxis=dict(
                            title='Energy Feedin / Loss (kW)',
                            range=[0, max(gridnode_df_by_iter['netfeedin_all_kW']) * 1.2],  # Adjust range as needed
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


                    # plot AGGREGATION ----------------
                    color_pal = trace_color_dict[fig_agg_color_palettes[i_scen]]
                    color_pal_idx = len(color_pal)
                    
                    fig_agg.add_trace(go.Scatter(x=[None,], y=[None,], name=scen, opacity=0, ))
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_all_kW'],                name='Total netfeedin_all_kW',       mode='lines',         line=dict(color=color_pal[color_pal_idx-1],),                 marker=dict(symbol='cross',),   ))
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_all_taken_kW'],          name='Total netfeedin_all_taken_kW', mode='lines',         line=dict(color=color_pal[color_pal_idx-2],),                 marker=dict(symbol='cross',),   ))
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_all_loss_kW'],           name='Total netfeedin_all_loss_kW',  mode='lines',         line=dict(color=color_pal[color_pal_idx-3],),                 marker=dict(symbol='cross',),   ))
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['netfeedin_loss_ratio'],            name='netfeedin_loss_ratio',         mode='lines',         line=dict(color=color_pal[color_pal_idx-4],),                 marker=dict(symbol='circle',),  yaxis = 'y2'))
                    fig_agg.add_trace(go.Scatter(x=topo_df_iter['n_iter'],        y=topo_df_iter['ratio_EGID_pvinst'],                      name='ratio pvinst cumulative',      mode='lines+markers', line=dict(color=color_pal[color_pal_idx-5],),                 marker=dict(symbol='circle',),  yaxis = 'y2'))                    
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_mean_of_mean'],   name='Holding Capac mean of mean',   mode='lines+markers', line=dict(color=color_pal[color_pal_idx-6],),                 marker=dict(symbol='diamond',), yaxis = 'y2'))
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_max_abs'],        name='Holding Capac max abs',        mode='lines',         line=dict(color=color_pal[color_pal_idx-6], dash = 'dash'),                                   yaxis = 'y2'))
                    fig_agg.add_trace(go.Scatter(x=gridnode_df_by_iter['n_iter'], y=gridnode_df_by_iter['holding_capacity_min_abs'],        name='Holding Capac min abs',        mode='lines',         line=dict(color=color_pal[color_pal_idx-6], dash = 'dash'),                                   yaxis = 'y2'))

                    fig_agg.add_trace(go.Scatter(x=[None,], y=[None,], name='', opacity=0, ))

                
                # export AGGREGATION ----------------
                fig_agg.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Energy (kWh Feedin / Loss)',
                    yaxis=dict(
                        title='Energy Feedin / Loss (kW)',
                        range=[0, max(gridnode_df_by_iter['netfeedin_all_kW']) * 1.2],  # Adjust range as needed
                    ),  
                    yaxis2=dict(
                        title='Holding Capacity',
                        overlaying='y',
                        side='right',
                        range=[0, max(gridnode_df_by_iter['holding_capacity_mean_of_mean']) * 1.2]  # Adjust range as needed
                    ),
                    legend_title='Legend',
                    title=f'Development of Energy Feedin / Loss for 1 Year (weather year: {self.pvalloc_scen.WEAspec_weather_year}) Over Time',
                    template='plotly_white',
                )

                if self.visual_sett.plot_show and self.visual_sett.plot_ind_line_PVproduction_TF[1]:
                    fig_agg.show()
                if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction__{len(self.pvalloc_scen_list)}scen.html'):
                    n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction__{len(self.pvalloc_scen_list)}scen*.html'))
                    os.rename(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction__{len(self.pvalloc_scen_list)}scen.html', 
                              f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction__{len(self.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
                fig_agg.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_PVproduction__{len(self.pvalloc_scen_list)}scen.html')
                print_to_logfile(f'\texport: plot_agg_line_PVproduction__{len(self.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)



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
                    periods_list = [int(path.split('npv_df_')[-1].split('.parquet')[0]) for path in npv_df_paths]
                    before_period, after_period = min(periods_list), max(periods_list)

                    npv_df_before = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_{before_period}.parquet')
                    npv_df_after  = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/npv_df_{after_period}.parquet')

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
                if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions__{len(self.pvalloc_scen_list)}scen.html'):
                    n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions__{len(self.pvalloc_scen_list)}scen*.html'))
                    os.rename(f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions__{len(self.pvalloc_scen_list)}scen.html', 
                              f'{self.visual_sett.visual_path}/plot_agg_hist_NPV_freepartitions__{len(self.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
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


        def plot_ind_lineband_contcharact_newinst(self, ):
            if self.visual_sett.plot_ind_lineband_contcharact_newinst_TF[0]:
                checkpoint_to_logfile('plot_ind_line_gridPremium_structure', self.visual_sett.log_name)

                # available color palettes
                trace_color_dict = {
                    'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
                    'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
                    'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
                    'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
                    'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
                }   

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                        self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]
                        self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                        # setup + import ----------
                        colnams_charac_AND_numerator = self.visual_sett.plot_ind_line_contcharact_newinst_specs['colnames_cont_charact_installations_AND_numerator']
                        trace_color_palette = self.visual_sett.plot_ind_line_contcharact_newinst_specs['trace_color_palette']
                        # col_colors = [val / len(colnams_charac_AND_numerator) for val in range(1,len(colnams_charac_AND_numerator)+1)]
                        col_colors = list(range(1,len(colnams_charac_AND_numerator)+1))
                        palette = trace_color_dict[trace_color_palette]
                    
                        topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                        predinst_all= pd.read_parquet( f'{self.visual_sett.mc_data_path}/pred_inst_df.parquet')
                        predinst_absdf = copy.deepcopy(predinst_all)

                        agg_dict ={}
                        for col_tuple in colnams_charac_AND_numerator:
                            agg_dict[f'{col_tuple[0]}'] = ['mean', 'std']
                            predinst_absdf[f'{col_tuple[0]}'] = predinst_absdf[f'{col_tuple[0]}'] / col_tuple[1]

                        agg_predinst_absdf = predinst_absdf.groupby('iter_round').agg(agg_dict)
                        agg_predinst_absdf['iter_round'] = agg_predinst_absdf.index

                        agg_predinst_absdf.replace(np.nan, 0, inplace=True) # replace NaNs with 0, needed if no deviation in std

                        # plot ----------------
                        fig = go.Figure()
                        i_col = 3
                        col, col_numerator = colnams_charac_AND_numerator[i_col][0], colnams_charac_AND_numerator[i_col][1]
                        for i_col, col_tuple in enumerate(colnams_charac_AND_numerator):
                            col = col_tuple[0]
                            col_numerator = col_tuple[1]

                            xaxis   =           agg_predinst_absdf['iter_round']
                            y_mean  =           agg_predinst_absdf[col]['mean']
                            y_lower, y_upper =  agg_predinst_absdf[col]['mean'] - agg_predinst_absdf[col]['std'], agg_predinst_absdf[col]['mean'] + agg_predinst_absdf[col]['std']
                            trace_color = palette[col_colors[i_col % len(col_colors)]]

                            # mean trace
                            fig.add_trace(go.Scatter(x=xaxis, y=y_mean,
                                                    name=f'{col} mean (1/{col_numerator})',
                                                    legendgroup=f'{col}',
                                                    line=dict(color=trace_color),
                                                    mode='lines+markers', showlegend=True))
                            # upper / lower bound band
                            fig.add_trace(go.Scatter(
                                x=xaxis.tolist() + xaxis.tolist()[::-1],  # Concatenate xaxis with its reverse
                                y=y_upper.tolist() + y_lower.tolist()[::-1],  # Concatenate y_upper with reversed y_lower
                                fill='toself',
                                fillcolor=trace_color,  # Dynamic color with 50% transparency
                                opacity=0.2,
                                line=dict(color='rgba(255,255,255,0)'),  # No boundary line
                                hoverinfo="skip",  # Don't show info on hover
                                showlegend=False,  # Do not show this trace in the legend
                                legendgroup=f'{col}',  # Group with the mean line
                                visible=True  # Make this visible/toggleable with the mean line
                            ))
                        
                        # add nEGID count trace
                        total_EGID_in_topo  = len(topo)
                        agg_predinst_counts = predinst_absdf.groupby('iter_round').size().reset_index(name='nEGID_count')
                        agg_predinst_counts['nEGID_cumm'] = agg_predinst_counts['nEGID_count'].cumsum()
                        agg_predinst_counts['nEGID_cumm_rel'] = agg_predinst_counts['nEGID_cumm'] / total_EGID_in_topo 
                        fig.add_trace(go.Scatter(x=agg_predinst_counts['iter_round'],
                                                y=agg_predinst_counts['nEGID_cumm_rel'],
                                                name='nEGID count (rel total EGID in topo)',
                                                legendgroup='nEGID count',
                                                line=dict(color='black', width=2),
                                                mode='lines+markers', showlegend=True))
                                                 

                        fig.update_layout(
                            xaxis_title='Iteration Round',
                            yaxis_title='Mean (+/- 1 std)',
                            legend_title='Scenarios',
                            title = f'Agg. Cont. Charact. of Newly Installed Buildings per Iteration Round (iter unit: {self.pvalloc_scen.CSTRspec_iter_time_unit})', 
                                uirevision='constant'  # Maintain the state of the plot when interacting

                        )
                        fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen_list[i_scen])

                        if self.visual_sett.plot_show and self.visual_sett.plot_ind_lineband_contcharact_newinst_TF[1]:
                            if self.visual_sett.plot_ind_lineband_contcharact_newinst_TF[2]:
                                fig.show()
                            elif not self.visual_sett.plot_ind_lineband_contcharact_newinst_TF[2]:
                                fig.show() if i_scen == 0 else None
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_lineband_contcharact_newinst.html')
                        else:
                            fig.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_lineband_contcharact_newinst.html')



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
                        gwr_gdf =       gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                        gm_gdf =        gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')

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
                            demand_type_list.append(v['demand_arch_typ'])
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
                        gwr_gdf         = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                        gm_gdf          = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
                        dsonodes_gdf    = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/dsonodes_gdf.geojson')

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
                            demand_type_list.append(v['demand_arch_typ'])
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


        def plot_ind_map_node_connections(self, ): 
            if self.visual_sett.plot_ind_map_node_connections_TF[0]:

                map_topo_egid_specs = self.visual_sett.plot_ind_map_topo_egid_specs
                map_node_connections_specs = self.visual_sett.plot_ind_map_node_connections_specs
                checkpoint_to_logfile('plot_ind_map_node_connections', self.visual_sett.log_name)

                for i_scen, scen in enumerate(self.pvalloc_scen_list):
                    self.visual_sett.mc_data_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}')[0]

                    # import
                    gwr_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                    gm_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
                    dsonodes_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/dsonodes_gdf.geojson')
                    Map_egid_dsonode = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/Map_egid_dsonode.parquet')
                    topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))

                    # transformations
                    egid_in_topo = [k for k in topo.keys()]
                    gwr_gdf = copy.deepcopy(gwr_gdf.loc[gwr_gdf['EGID'].isin(egid_in_topo)])
                    Map_egid_dsonode.reset_index(drop=True, inplace=True)
                    gwr_gdf = gwr_gdf.merge(Map_egid_dsonode, on='EGID', how='left')

                    # pv_instdf_creation for base map
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
                        demand_type_list.append(v['demand_arch_typ'])
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


                    # dsonode map ----------
                    if True:
                        fig_dsonodes = copy.deepcopy(fig_topobase)
                        gwr_gdf = gwr_gdf.set_crs('EPSG:2056', allow_override=True)
                        gwr_gdf = gwr_gdf.to_crs('EPSG:4326')
                        gwr_gdf['geometry'] = gwr_gdf['geometry'].apply(self.flatten_geometry)

                        dsonodes_gdf = dsonodes_gdf.set_crs('EPSG:2056', allow_override=True)
                        dsonodes_gdf = dsonodes_gdf.to_crs('EPSG:4326')
                        dsonodes_gdf['geometry'] = dsonodes_gdf['geometry'].apply(self.flatten_geometry)

                        # define point coloring
                        unique_nodes = gwr_gdf['grid_node'].unique()
                        colors = pc.sample_colorscale(map_node_connections_specs['point_color_palette'], [n/(len(unique_nodes)) for n in range(len(unique_nodes))])
                        node_colors = [colors[c] for c in range(len(unique_nodes))]
                        colors_df = pd.DataFrame({'grid_node': unique_nodes, 'node_color': node_colors})
                        
                        gwr_gdf = gwr_gdf.merge(colors_df, on='grid_node', how='left')
                        dsonodes_gdf = dsonodes_gdf.merge(colors_df, on='grid_node', how='left')

                        # plot points as Scattermapbox
                        gwr_gdf['hover_text'] = gwr_gdf['EGID'].apply(lambda egid: f'EGID: {egid}')

                        fig_dsonodes.add_trace(go.Scattermapbox(lat=gwr_gdf.geometry.y,lon=gwr_gdf.geometry.x, mode='markers',
                            marker=dict(
                                size=map_node_connections_specs['point_size_all'],
                                color=map_node_connections_specs['point_color_all'],
                                opacity=map_node_connections_specs['point_opacity_all']
                                ),
                                text=gwr_gdf['hover_text'],
                                hoverinfo='text',
                                showlegend=False
                                ))

                        for un in unique_nodes:
                            # node center / trafo location
                            dsonodes_gdf_node = dsonodes_gdf.loc[dsonodes_gdf['grid_node'] == un]
                            fig_dsonodes.add_trace(go.Scattermapbox(lat=dsonodes_gdf_node.geometry.y,lon=dsonodes_gdf_node.geometry.x, mode='markers',
                                # marker_symbol = 'cross', 
                                marker=dict(
                                    size=map_node_connections_specs['point_size_dsonode_loc'],
                                    color=dsonodes_gdf_node['node_color'],
                                    opacity=map_node_connections_specs['point_opacity_dsonode_loc']
                                    ),
                                    name= f'trafo: {un}',
                                    text=f'node: {un}, kVA_thres: {dsonodes_gdf_node["kVA_threshold"].sum()}',
                                    hoverinfo='text',
                                    legendgroup='trafo',
                                    legendgrouptitle=dict(text='Trafo Locations'),
                                    showlegend=True
                                    ))

                            # all buildings
                            gwr_gdf_node = gwr_gdf.loc[gwr_gdf['grid_node'] == un]
                            fig_dsonodes.add_trace(go.Scattermapbox(lat=gwr_gdf_node.geometry.y,lon=gwr_gdf_node.geometry.x, mode='markers',
                                marker=dict(
                                    size=map_node_connections_specs['point_size_bynode'],
                                    color=gwr_gdf_node['node_color'],
                                    opacity=map_node_connections_specs['point_opacity_bynode']
                                    ),
                                    name= f'{un}',
                                    text=gwr_gdf_node['grid_node'],
                                    hoverinfo='text',
                                    showlegend=True
                                    ))


                    if self.visual_sett.plot_show and self.visual_sett.plot_ind_map_node_connections_TF[1]:
                        if self.visual_sett.plot_ind_map_node_connections_TF[2]:
                            fig_dsonodes.show()
                        elif not self.visual_sett.plot_ind_map_node_connections_TF[2]:
                            fig_dsonodes.show() if i_scen == 0 else None
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_dsonodes.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_map_node_connections.html')
                    else:
                        fig_dsonodes.write_html(f'{self.visual_sett.visual_path}/{scen}__plot_ind_map_node_connections.html')
                            

        # def plot_ind_map_omitted_egids(self, ): 

        # def plot_ind_lineband_contcharact_newinst(self, ): 




# ======================================================================================================
# RUN VISUALIZATION
# ======================================================================================================


if __name__ == '__main__':
    # if False:

    visualization_list = [
        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*.xlsx','*.csv','*.parquet',
                '*old_vers*',
                ], 
            pvalloc_include_pattern_list = [
                'pvalloc_mini_rnd',
                # 'pvalloc_RUR_test2c_default_max', 
          ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = True,  
            remove_old_plots_in_visualization  = False,  
            remove_old_csvs_in_visualization   = False, 
    )        
    ]

    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        plot_method_names = [

            # # -- def plot_ALL_init_sanitycheck(self, ): -------------
            # "plot_ind_var_summary_stats",                     # runs as intended
            # # "plot_ind_hist_pvcapaprod_sanitycheck",           # runs as intended
            # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
            # "plot_ind_charac_omitted_gwr",                     # runs as intended
            # # "plot_ind_line_meteo_radiation",                   # runs as intended

            # # # -- def plot_ALL_mcalgorithm(self,): -------------
            # # "plot_ind_line_installedCap",                     # runs as intended
            # "plot_ind_line_productionHOY_per_node",           # runs as intended
            # "plot_ind_line_productionHOY_per_EGID",           # runs as intended
            "plot_ind_cols_HOYagg_per_EGID", 
            # "plot_ind_line_PVproduction",                   # runs — optional, uncomment if needed
            # # "plot_ind_hist_NPV_freepartitions",               # runs as intended
            # # # "plot_ind_line_gridPremiumHOY_per_node",          # runs
            # # # "plot_ind_line_gridPremium_structure",            # runs
            # # "plot_ind_lineband_contcharact_newinst",          # status not noted
            # "plot_ind_map_topo_egid",                         # runs as intended
            # # "plot_ind_map_topo_egid_incl_gridarea",         # runs as intended — optional
            # # # "plot_ind_map_node_connections"                   # status not noted        
    
            # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
            # # plot_ind_map_node_connections()
            # # plot_ind_map_omitted_egids()
        ]

        for plot_method in plot_method_names:
            # try:
            method = getattr(visual_class, plot_method)
            method()
            # except Exception as e:
            #     print(f"Error in {plot_method}: {e}")
 
    print('end MAIN_visualization.py')


            