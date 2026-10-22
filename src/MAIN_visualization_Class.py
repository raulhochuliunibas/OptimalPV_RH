# python integrated packages
import sys
import os as os
import glob
import shutil
import traceback
from dataclasses import dataclass, field, replace
from typing_extensions import List, Dict

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile
from src.visualization.plot_ind_var_summary_stats import PlotIndVarSummaryStats
from src.visualization.plot_demand_profiles import PlotDemandProfiles
from src.visualization.plot_ind_hist_pvcapaprod_sanitycheck import PlotIndHistPvcapaprodSanitycheck
from src.visualization.plot_ind_boxp_radiation_rng_sanitycheck import PlotIndBoxpRadiationRngSanitycheck
from src.visualization.plot_ind_charac_omitted_gwr import PlotIndCharacOmittedGwr
from src.visualization.plot_ind_line_meteo_radiation import PlotIndLineMeteoRadiation
from src.visualization.plot_ind_line_installedCap import PlotIndLineInstalledCap
from src.visualization.plot_ind_line_productionHOY_per_node import PlotIndLineProductionHOYPerNode
from src.visualization.plot_ind_line_productionHOY_per_node_byiter import PlotIndLineProductionHOYPerNodeByiter
from src.visualization.plot_ind_line_productionHOY_per_EGID import PlotIndLineProductionHOYPerEGID
from src.visualization.plot_ind_hist_cols_HOYagg_per_EGID import PlotIndHistColsHOYaggPerEGID
from src.visualization.plot_ind_line_PVproduction import PlotIndLinePVproduction
from src.visualization.plot_ind_hist_NPV_freepartitions import PlotIndHistNPVFreepartitions
from src.visualization.plot_ind_line_gridPremiumHOY_per_node import PlotIndLineGridPremiumHOYPerNode
from src.visualization.plot_ind_line_gridPremium_structure import PlotIndLineGridPremiumStructure
from src.visualization.plot_ind_lineband_contcharact_newinst import PlotIndLinebandContcharactNewinst
from src.visualization.plot_ind_map_topo_egid import PlotIndMapTopoEgid
from src.visualization.plot_ind_map_topo_egid_incl_gridarea import PlotIndMapTopoEgidInclGridarea
from src.visualization.plot_ind_map_node_connections import PlotIndMapNodeConnections
from src.visualization.plot_ind_mapline_prodHOY_EGIDrfcombo import PlotIndMaplineProdHOYEGIDrfcombo
from src.visualization.plot_ind_jsctr_ausricht_flaech_newinst import PlotIndJsctrAusrichtFlaechNewinst
from src.visualization.plot_ind_hist_contcharact_newinst import PlotIndHistContcharactNewinst
from src.visualization.plot_ind_bar_catgcharact_newinst import PlotIndBarCatgcharactNewinst
from src.visualization.plot_ind_summary_stats_by_node import PlotIndSummaryStatsByNode
from src.visualization.visual_export_utils import VisualizationExportUtils


@dataclass
class Visual_Settings:
    pvalloc_scen_list : List[str]                = field(default_factory=lambda: [])
    pvalloc_exclude_pattern_list : List[str]     = field(default_factory=lambda: [
                                                    '*.txt',
                                                    '*.tar.gz',
                                                    '*old_vers*',
                                                    '*old*',
                                                    'x_*',
                                                    '.xlsx'
                                                    'archiv'
                                                    ])
    pvalloc_include_pattern_list : List[str]      = field(default_factory=lambda: [])
    save_plot_by_scen_directory: bool            = True
    remove_old_plot_scen_directories: bool       = False
    remove_old_plots_in_visualization: bool      = False
    remove_old_csvs_in_visualization: bool       = False
    export_only_agg_comparison_plots: bool       = False
    MC_subdir_for_plot: str                      = 'zMC1*'
    mc_plots_individual_traces: bool             = True
    cut_timeseries_to_zoom_hour: bool            = False
    add_day_night_HOY_bands: bool                = False
    reduce_information_content: bool             = False


    default_zoom_year: List[int]                 = field(default_factory=lambda: [2002, 2030]) # field(default_factory=lambda: [2002, 2030]),
    default_zoom_hour: List[int]                 = field(default_factory=lambda: [212*24, (212*24)+(24*7)]) # field(default_factory=lambda: [2400, 2400+(24*7)]),
    default_map_zoom: int                        = 13
    default_map_center: List[float]              = field(default_factory=lambda: [47.38, 7.65])
    node_selection_for_plots: List[str]          = field(default_factory=lambda: ['1', '3', '5'])

    # PLOT SPECS: one dict per plot method. Whether a plot runs is decided by calling its method (or not);
    # showing / exporting by the plot_show / plot_export kwargs of that method.
    # for pvalloc_inital + sanitycheck plots
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

    # for pvalloc_MC_algorithm plots
    plot_ind_line_productionHOY_per_EGID_specs: Dict        = field(default_factory=lambda: {
        'grid_nodes_counts_minmax': (5,999),         # try to select grid node with nEGIDs for visualization
        'specific_gridnodes_egid_HOY': None,         # select specific gridn node by id number for visualization
        'egid_col_to_plot'          : ['demand_kW',
                                       'radiation',
                                       'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW' ],
        'grid_col_to_plot_tuples'   : [('netfeedin_kW', 'feedin_atnode_kW'),
                                       ('netfeedin_kW', 'demand_proxy_out_kW'),
                                    #    ('netfeedin_kW', 'feedin_atnode_taken_kW'),
                                    #    ('netfeedin_kW', 'feedin_atnode_loss_kW'),
                                    #    ('netfeedin_kW', 'kW_threshold'),
                                    ],
        'gridnode_pick_col_to_agg':  [
                                        'demand_kW',
                                        'netfeedin_kW',
                                        'feedin_atnode_kW',
                                        # 'feedin_atnode_taken_kW',
                                        # 'feedin_atnode_loss_kW',
                                        'kW_threshold',
                                        'demand_proxy_out_kW'
                                        ],
        'egid_trace_opacity' : 0.9,
        'grid_trace_opacity' : 0.8,
    })
    plot_ind_hist_cols_HOYagg_per_EGID_specs: Dict          = field(default_factory=lambda: {
        'hist_cols_to_plot': [
                            'demand_kW',
                            'pvprod_kW',
                            'selfconsum_kW',
                            'netdemand_kW',
                            'netfeedin_kW',
                            'selfconsum_pvprod_ratio', ],
        'n_hist_bins_bycol': {
                            'demand_kW':                20,
                            'pvprod_kW':                50,
                            'selfconsum_kW':            20,
                            'netdemand_kW':             20,
                            'netfeedin_kW':             50,
                            'selfconsum_pvprod_ratio':  20,
                        },
        'show_hist_traces_TF': False,
        'hist_opacity': 0.85,
        'subset_trace_color_palette': 'Turbo',
        'all_egid_trace_color_palette': 'Viridis',

        'xy_tpl_cols_to_plot_joint_scatter': [
            ('TotalPower', 'dfuidPower'),
            # ('TotalPower', 'radiation'),
            # ('TotalPower', 'pvprod_kW'),
            ],
    })
    plot_ind_line_PVproduction_bynode_specs: Dict        = field(default_factory=lambda: {
        'select_nodes_stacked_traces': [],
        'n_top_loss_nodes': 15,}
        , )
    plot_ind_hist_NPV_freepartitions_specs: Dict        = field(default_factory=lambda: {
        'show_traces': ['hist', 'kde'],
        'exclude_filter_tags': [
            'filter_tag__south_nr', 'filter_tag__south_1r',
            'filter_tag__eastwest_2r', 'filter_tag__eastwest_nr',
            'subs_filter_tags_chf',

            # 'filter_tag__eastwest_80pr',
            'filter_tag__eastwest_70pr',
            'filter_tag__eastORwest_50pr',
            'filter_tag__eastORwest_40pr',
            # 'filter_tag__south_50pr',
            # 'filter_tag__south_40pr',
            'filt_dfuid_east_50pr',
            'filt_dfuid_east_40pr',
            'filt_dfuid_east_35pr',
            'filt_dfuid_west_50pr',
            'filt_dfuid_west_40pr',
            'filt_dfuid_west_35pr',
            'filt_dfuid_south_50pr',
            'filt_dfuid_south_40pr',
        ],
        'nbins': 25,
        'tag_subsidy_tuples':{
            ('filter_tag__eastwest_80pr', 0),
            # ('filter_tag__eastwest_80pr', 2500),
            # ('filter_tag__eastwest_80pr', 5000),
            # ('filter_tag__eastwest_80pr', 7500),
        }
    })


    plot_ind_map_topo_egid_specs: Dict                      = field(default_factory=lambda: {
        'uniform_municip_color': '#fff2ae',
        'shape_opacity': 0.2,
        'point_opacity': 0.7,
        'point_opacity_sanity_check': 0.4,
        'point_size_pv': 6,
        'point_size_rest': 4.5,
        'point_size_sanity_check': 20,
        'point_size_outsample': 3,
        'point_color_pv_df': '#54f533',      # green
        'point_color_solkat': '#f06a1d',     # turquoise
        'point_color_alloc_algo': '#ffa600', # yellow
        'point_color_rest': '#383838',       # dark grey
        'point_color_sanity_check': '#0041c2', # blue
        'point_color_outsample': '#ed4242',
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
    plot_ind_mapline_prodHOY_EGIDrfcombo_specs:Dict        = field(default_factory=lambda: {
        'specific_selected_EGIDs': [],
        # 'specific_selected_EGIDs': [
        #     '190113872', # heatpump
        #     '190057138', # no heatpump
        # ],
        'specific_exclude_EGIDs': [],
        'run_selected_plot_parts': [
                                'map',
                                'timeseries',
                                # 'summary',
                                # 'econ_func',
                                'joint_scatter',

                                # 'all_combo_TS',
                                # 'actual_inst_TS',
                                'outsample_TS',
                                # 'recalc_opt_inst',
                                ],
        'traces_in_timeseries_plot': [
                                    'full_partition_combo',
                                    'actual_pv_inst',
                                    'recalc_pv_inst',
                                    ],
        'rndm_sample_seed': None,
        'n_rndm_egid_winst_pvdf': 1,
        'n_rndm_egid_winst_alloc': 1,
        'n_rndm_egid_woinst': 1,
        'n_rndm_egid_outsample': 0,
        'n_partition_pegid_minmax': (1,4),
        'roofpartition_color': '#fff2ae',
        'actual_ts_trace_marker': 'cross',
        'summary_cols_only_in_legend': ['EGID', 'inst_TF', 'info_source', 'grid_node',
                                        'GKLAS', 'GAREA', 'are_typ', 'sfhmfh_typ',
                                        'TotalPower', 'demand_elec_pGAREA',
                                        'pvtarif_Rp_kWh', 'elecpri_Rp_kWh', ],
        'recalc_opt_max_flaeche_factor': None, #  1.5,
        'recalc_interest_rate_list': [
            0.02, ],
        'tryout_generic_pvtariff_elecpri_Rp_kWh': (None, None), # e.g. other generic setting (5.0, 25.0),
        'generic_pvinstcost_coeff_total': {
            'use_generic_instcost_coeff_total_TF': False,
            'coeff_total_dict_list': [
                {'a': 1193.8, 'b': 4860.3, 'c': 0.7291},  #  estimates pvinst_coefficients
                # {'a': 6000, 'b': 5000, 'c': 0.65},  #  higher fix cost, more curvutre, lower cost for large inst
                # {'a': 4000, 'b': 3300, 'c': 0.83},  #  higher fix cost, then much closes to estim function, higher cost for large inst
                # {'a': 7000, 'b': 1500, 'c': 0.85},  #  much higher fix costs, much less change in curvature, almost linear, much lower cost for large inst
                ],
            },
        'noisy_demand_TF': False,
        'noisy_demand_factor': 0.0,
            })
    plot_ind_lineband_contcharact_newinst_specs: Dict           = field(default_factory=lambda: {
            'trace_color_palette': 'Turbo',
            'upper_lower_bound_interval': [0.05, 0.95],
            'colnames_cont_charact_installations_AND_numerator': [
            ('FLAECHE',                1,    ),
            ('selfconsum_kW',          1000, ),
            ('netdemand_kW',           1000, ),
            ('dfuidPower',             1,    ),
            # ('elecpri_Rp_kWh',        1,   ),
            # ('pvtarif_Rp_kWh',        1,   ),
            # ('estim_pvinstcost_chf',   100,),
            ],
        })
    plot_ind_XX_charact_newinst_specs: Dict           = field(default_factory=lambda: {
        'n_cols_cont_charact': 2,
        'cols_cont_charact_inst_binsize': [
            ('FLAECHE',                 35, ),
            ('GAREA',                   35, ),
            # ('AUSRICHTUNG_original',    25, ),
            # ('AUSRICHTUNG_share_used',  25, ),
            # ('NEIGUNG_original',        10, ),
            # ('NEIGUNG_share_used',      10, ),
            ],
        'n_cols_catg_charact': 2,
        'max_n_rows_catg_charact': 20,
        'cols_catg_charact_inst': [
            'GKLAS',
            # 'GSTAT',
            # 'GBAUJ_cat',
            'are_typ',
            # 'sfhmfh_typ',
            # 'grid_node',
            'heatpump_TF',
            'filter_tag',
            ],
        'cols_as_filter_tags': [
            # ascending order so that "larger" tags overwritte smaller ones (e.g. south50 is also in south40, so will overwrite it if later in the queue)
            'filter_tag__eastwest_70pr',
            'filter_tag__eastwest_80pr',
            'filter_tag__south_40pr',
            'filter_tag__south_50pr',
        ]
        })
    plot_ind_summary_stats_by_node_specs: Dict              = field(default_factory=lambda:  {
        'node_summary_list': [],
        'cont_col_plot': [
            'AUSRICHTUNG_original',
            'AUSRICHTUNG_FLAECHE',
            'FLAECHE',
            'GAREA',
            'pvdf_inst_ratio',
        ]
    })

    # old out of use plots
    plot_ind_map_omitted_egids_specs: Dict                  = field(default_factory=lambda: {
            'point_opacity': 0.7,
            'point_size_select_but_omitted': 12,
            'point_size_rest_not_selected': 1, # 4.5,
            'point_color_select_but_omitted': '#ed4242', # red
            'point_color_rest_not_selected': '#ff78ef',  # pink
            'export_gdfs_to_shp': True,
        })


class Visualization(
    PlotIndVarSummaryStats,
    PlotDemandProfiles,
    PlotIndHistPvcapaprodSanitycheck,
    PlotIndBoxpRadiationRngSanitycheck,
    PlotIndCharacOmittedGwr,
    PlotIndLineMeteoRadiation,
    PlotIndLineInstalledCap,
    PlotIndLineProductionHOYPerNode,
    PlotIndLineProductionHOYPerNodeByiter,
    PlotIndLineProductionHOYPerEGID,
    PlotIndHistColsHOYaggPerEGID,
    PlotIndLinePVproduction,
    PlotIndHistNPVFreepartitions,
    PlotIndLineGridPremiumHOYPerNode,
    PlotIndLineGridPremiumStructure,
    PlotIndLinebandContcharactNewinst,
    PlotIndMapTopoEgid,
    PlotIndMapTopoEgidInclGridarea,
    PlotIndMapNodeConnections,
    PlotIndMaplineProdHOYEGIDrfcombo,
    PlotIndJsctrAusrichtFlaechNewinst,
    PlotIndHistContcharactNewinst,
    PlotIndBarCatgcharactNewinst,
    PlotIndSummaryStatsByNode,
    VisualizationExportUtils,
):
    """
    Entry point of a visualization run. Inherits the plot methods from the classes in src/visualization/plot_*.py
    (all built on VisualizationCore), so a plot is run by calling its method, e.g.

        visual_class = Visualization(Visual_Settings(pvalloc_include_pattern_list = ['pvalloc_LRG3_max']))
        visual_class.plot_ind_hist_contcharact_newinst(plot_show = True, plot_export = True)

    Every plot method takes the kwargs
        plot_show:   open the figure(s) in the browser
        plot_export: write the figure(s) (html) and plot data (csv) to disk; False writes nothing
    """

    def __init__(self, settings: Visual_Settings):
        super().__init__(settings)

        # create new visual directories per scenario (+ remove old ones)
        for scen in self.visual_sett.pvalloc_scen_list:
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
            # old_files = glob.glob(f'{self.visual_sett.visual_path}/*.csv')
            old_files = glob.glob(f'{self.visual_sett.visual_static_export_path}/*.csv')
            for file in old_files:
                os.remove(file)


        chapter_to_logfile('start MASTER_visualization\n', self.visual_sett.log_name, overwrite_file=True)
        subchapter_to_logfile('scenarios to plot:', self.visual_sett.log_name)
        for scen in self.visual_sett.pvalloc_scen_list:
            print_to_logfile(f'- {scen}', self.visual_sett.log_name)

        print_to_logfile('\n\n', self.visual_sett.log_name)


    # ------------------------------------------------------------------------------------------------------
    # RUN (many) PLOTS AT ONCE
    # ------------------------------------------------------------------------------------------------------
    def plot_ALL(self, plot_show: bool = True, plot_export: bool = True, 
                 main_plots_only: bool = False,
                 main_plot_list = None
        ):
        default_main_plot_list = [
                     'plot_ind_line_PVproduction', 
                     'plot_ind_line_installedCap', 
                     'plot_ind_hist_contcharact_newinst', 
                     'plot_ind_bar_catgcharact_newinst', 
                     'plot_ind_var_summary_stats', 
                     'plot_ind_hist_NPV_freepartitions', 
                     'plot_ind_map_topo_egid',
                     ]
        main_plot_list = main_plot_list if main_plot_list is not None else default_main_plot_list


        # self.export_csvs_for_static_plots()

        plots_dict = {
            # init + sanity check plots
            'plot_ind_var_summary_stats':                   self.plot_ind_var_summary_stats,
            'plot_demand_profiles':                         self.plot_demand_profiles,
            'plot_ind_hist_pvcapaprod_sanitycheck':         self.plot_ind_hist_pvcapaprod_sanitycheck,

            # mc algorithm plots
            'plot_ind_line_installedCap':                  self.plot_ind_line_installedCap,
            'plot_ind_line_productionHOY_per_node':        self.plot_ind_line_productionHOY_per_node,
            'plot_ind_line_productionHOY_per_node_byiter': self.plot_ind_line_productionHOY_per_node_byiter,
            'plot_ind_line_productionHOY_per_EGID':        self.plot_ind_line_productionHOY_per_EGID,
            'plot_ind_hist_cols_HOYagg_per_EGID':          self.plot_ind_hist_cols_HOYagg_per_EGID,
            'plot_ind_line_PVproduction':                  self.plot_ind_line_PVproduction,
            'plot_ind_hist_NPV_freepartitions':            self.plot_ind_hist_NPV_freepartitions,
            'plot_ind_line_gridPremiumHOY_per_node':       self.plot_ind_line_gridPremiumHOY_per_node,
            'plot_ind_line_gridPremium_structure':         self.plot_ind_line_gridPremium_structure,            


            # mc algorithm plots - maps
            'plot_ind_map_topo_egid':                      self.plot_ind_map_topo_egid,
            'plot_ind_map_topo_egid_incl_gridarea':        self.plot_ind_map_topo_egid_incl_gridarea,
            'plot_ind_mapline_prodHOY_EGIDrfcombo':        self.plot_ind_mapline_prodHOY_EGIDrfcombo,
        
            # mc algorithm plots - characteristics of new installations
            'plot_ind_hist_contcharact_newinst':           self.plot_ind_hist_contcharact_newinst,
            'plot_ind_bar_catgcharact_newinst':            self.plot_ind_bar_catgcharact_newinst,
            'plot_ind_summary_stats_by_node':              self.plot_ind_summary_stats_by_node,

            # # not relevant / used
            # self.plot_ind_charac_omitted_gwr,
            # self.plot_ind_line_meteo_radiation,
            # self.plot_ind_jsctr_ausricht_flaech_newinst,
            # self.plot_ind_lineband_contcharact_newinst,

            # # not yet / no longer working
            # # self.plot_ind_boxp_radiation_rng_sanitycheck,
        }

        # decide what runs - and say it, so that "nothing ran" cannot go unnoticed ----------
        plot_names_all = list(plots_dict)
        main_names     = [name for name in main_plot_list if name in plot_names_all]
        unknown_names  = [name for name in main_plot_list if name not in plot_names_all]
        other_names    = [] if main_plots_only else [name for name in plot_names_all if name not in main_names]
        plots_to_run   = main_names + other_names

        if unknown_names:
            print_to_logfile(f'WARNING plot_ALL: not a plot name, ignored: {unknown_names}', self.visual_sett.log_name)
        if not plots_to_run:
            print_to_logfile('WARNING plot_ALL: no plot to run', self.visual_sett.log_name)
        print_to_logfile(f'plot_ALL: running {len(plots_to_run)} plots for {len(self.visual_sett.pvalloc_scen_list)} scenarios', self.visual_sett.log_name)

        plots_failed = []
        for plot_name in plots_to_run:
            try:
                plots_dict[plot_name](plot_show = plot_show, plot_export = plot_export)
            except Exception:
                plots_failed.append(plot_name)
                print_to_logfile(f'Error in {plot_name}:\n{traceback.format_exc()}', self.visual_sett.log_name)

        print_to_logfile(f'plot_ALL finished: {len(plots_to_run) - len(plots_failed)} of {len(plots_to_run)} plots without error'
                         + (f' | FAILED: {plots_failed}' if plots_failed else ''), self.visual_sett.log_name)



def make_visual_scenario(default_visual_sett, **overrides):
    return replace(default_visual_sett, **overrides)



# ======================================================================================================
# RUN VISUALIZATION
# ======================================================================================================
if __name__ == '__main__':

    # SCENARIO DEFINITION --------------------------------------------------------------------------
    # derive variations of the default with: make_visual_scenario(visual_default, pvalloc_include_pattern_list = [...])
    visual_default = Visual_Settings(
        # pvalloc_scen_list                 = [
        #     'pvalloc_26nbfs_LRG_max', 

        #     # 'pvalloc_26nbfs_LRG_max_nokev', 
        #     # 'pvalloc_26nbfs_LRG_max_As8p0',
        #     # 'pvalloc_26nbfs_LRG_max_Bs0p8', 
        #     'pvalloc_26nbfs_LRG_max_Cs6p8', 
        #     'pvalloc_26nbfs_LRG_max_dynelecpri2023', 
        #     # 'pvalloc_26nbfs_LRG_rnd', 
        # ], 
        pvalloc_include_pattern_list       = [
            # 'pvalloc_26nbfs_LRG_*', 
            'pvalloc_LRG4*', 
                                            ],
        pvalloc_exclude_pattern_list       = [
                                                '*.txt', '*.tar.gz', '*old_vers*',
                                                # '*1hll*',
                                            ],
        save_plot_by_scen_directory        = False,
        # remove_old_plot_scen_directories   = True,
        # remove_old_plots_in_visualization  = True,
        # remove_old_csvs_in_visualization   = True,
        # export_only_agg_comparison_plots   = False,
        # reduce_information_content         = True,

        cut_timeseries_to_zoom_hour        = True,
        add_day_night_HOY_bands            = True,
        default_map_zoom                   = 10,
        default_map_center                 = [47.46, 7.58],
    )


    # RUN PLOTS ------------------------------------------------------------------------------------
    # a plot is run by calling its method (comment the call out to skip it)
    # for visual_sett in [visual_default, visual_OLD_LRG_scens]:
    visual_class = Visualization(visual_default)

    if 'scicore' in os.getcwd() or 'SCICORE' in os.getcwd():
        visual_class.plot_ALL(plot_show = False, plot_export = True)

    else:
        visual_class.plot_ALL(plot_show = False, plot_export = True, main_plots_only = True)


    print(f'... end of <{os.path.basename(__file__)}> ...')
