
import sys
from pathlib import Path

# Add parent directory to path to find src module
sys.path.insert(0, str(Path(__file__).parent.parent))

# from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [

        # Visual_Settings(
        #     pvalloc_exclude_pattern_list = [
        #         '*.txt','*old_vers*',
        #     ], 
        #     pvalloc_include_pattern_list = [
        #         'pvalloc_29nbfs_30y_max',
        #     ],
            
        #     # export_only_agg_comparison_plots   = True,# <===
            # remove_old_plot_scen_directories   = True,  
            # remove_old_plots_in_visualization  = True, 
            # remove_old_csvs_in_visualization   = True, 

        #     cut_timeseries_to_zoom_hour        = True,

        #     plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False]          ,
        # ),

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt', "*old_vers*",
            ], 
            pvalloc_include_pattern_list = [      
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_epzb1',
                'pvalloc_LRG3_max_epzb025',
                'pvalloc_LRG3_max_hist01',
                'pvalloc_LRG3_max_elecpri60rp',
                'pvalloc_LRG3_max_elecpri15rp',
                'pvalloc_LRG3_rnd',
                'pvalloc_LRG3_max_1hll',
                'pvalloc_LRG3_max_gridoptim',
                'pvalloc_LRG3_max_pksh',
                'pvalloc_LRG3_max_nokev', 


                'pvalloc_LRG3_max_sAs4p0',
                'pvalloc_LRG3_max_sAs6p0',
                'pvalloc_LRG3_max_sAs8p0',

                'pvalloc_LRG3_max_sBs0p4',
                'pvalloc_LRG3_max_sBs0p6',
                'pvalloc_LRG3_max_sBs0p8',

                'pvalloc_LRG3_max_sCs4p4',
                'pvalloc_LRG3_max_sCs4p6',
                'pvalloc_LRG3_max_sCs4p8',
                'pvalloc_LRG3_max_sCs6p4',
                'pvalloc_LRG3_max_sCs6p6',
                'pvalloc_LRG3_max_sCs6p8',
                'pvalloc_LRG3_max_sCs8p4',
                'pvalloc_LRG3_max_sCs8p6',
                'pvalloc_LRG3_max_sCs8p8',

                ],

            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_installedCap_TF                   = [True,      True,       False]    ,
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_productionHOY_per_node_byiter_TF  = [True,      True,      False],
            # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 

            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
                # plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
                # plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 

            # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],

            # plot_ind_hist_cols_HOYagg_per_EGID_TF         = [True,      True,      False],
            ), 
        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt', "*old_vers*",
            ], 
            pvalloc_include_pattern_list = [      
                'pvalloc_LRG3_max',
                # 'pvalloc_LRG3_max_gridoptim',
                # 'pvalloc_LRG3_gridoptim_max',

                'pvalloc_LRG3_max_sAs4p0',
                'pvalloc_LRG3_max_sAs6p0',
                'pvalloc_LRG3_max_sAs8p0',

                'pvalloc_LRG3_max_sBs0p4',
                'pvalloc_LRG3_max_sBs0p6',
                'pvalloc_LRG3_max_sBs0p8',

                'pvalloc_LRG3_max_sCs4p4',
                'pvalloc_LRG3_max_sCs4p6',
                'pvalloc_LRG3_max_sCs4p8',
                'pvalloc_LRG3_max_sCs6p4',
                'pvalloc_LRG3_max_sCs6p6',
                'pvalloc_LRG3_max_sCs6p8',
                'pvalloc_LRG3_max_sCs8p4',
                'pvalloc_LRG3_max_sCs8p6',
                'pvalloc_LRG3_max_sCs8p8',

                ],

            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            # plot_ind_line_installedCap_TF                   = [True,      True,       False]    ,
            # plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            # plot_ind_line_productionHOY_per_node_byiter_TF  = [True,      True,      False],
            # # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,      False],
            # plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 

            # # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            # plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 

            # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],

            # plot_ind_hist_cols_HOYagg_per_EGID_TF         = [True,      True,      False],
            ), 

  
]




EconLunch_plots = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt', "*old_vers*",
            ], 
            pvalloc_include_pattern_list = [      
                'pvalloc_29nbfs_LRG2_max', 
                
                'pvalloc_29nbfs_LRG2_rnd', 
                'pvalloc_29nbfs_LRG2_max_1hll', 

                'pvalloc_29nbfs_LRG2_gridoptim_max',

                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
            
                'pvalloc_29nbfs_LRG2_max_sCs*p6',
                'pvalloc_29nbfs_LRG2_max_sCs*p8',

                'pvalloc_29nbfs_LRG2_max_epzb1', 
                'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1',
                'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2',
                'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3',

                'pvalloc_29nbfs_LRG2_max_pksh', 
                'pvalloc_29nbfs_LRG2_max_pksh991', 

                'pvalloc_29nbfs_LRG2_max_pksh_sCs2p8', 
                'pvalloc_29nbfs_LRG2_max_pksh_sCs4p8', 
                'pvalloc_29nbfs_LRG2_max_pksh_sCs6p8', 
                ],

            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_installedCap_TF                   = [True,      True,       False]    ,
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_productionHOY_per_node_byiter_TF  = [True,      True,      False],
            # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 

            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 

            plot_ind_summary_stats_by_node_TF               = [True,      True,       True],

            plot_ind_hist_cols_HOYagg_per_EGID_TF         = [True,      True,      False],
            ), 

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt', "*old_vers*",
            ], 
            pvalloc_include_pattern_list = [      
                'pvalloc_29nbfs_LRG2_max', 
                
                'pvalloc_29nbfs_LRG2_max_1hll', 

                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
            
                'pvalloc_29nbfs_LRG2_max_sCs*p6',
                'pvalloc_29nbfs_LRG2_max_sCs*p8',

                'pvalloc_29nbfs_LRG2_max_epzb1', 
                'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1',
                'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2',
                'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3',

                'pvalloc_29nbfs_LRG2_max_pksh', 

                'pvalloc_29nbfs_LRG2_max_pksh_sCs2p8', 
                'pvalloc_29nbfs_LRG2_max_pksh_sCs4p8', 
                'pvalloc_29nbfs_LRG2_max_pksh_sCs6p8', 
                ],

            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },

            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 

            ), 

]

sust_future_lunch_plots = [
        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [      
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_gridoptim_max',
                
                'pvalloc_29nbfs_LRG2_rnd',
                'pvalloc_29nbfs_LRG2_max_epzb0_50', 
                'pvalloc_29nbfs_LRG2_max_epzb0_75', 
                
                # 'pvalloc_29nbfs_LRG2_max_sAs2p0',
                'pvalloc_29nbfs_LRG2_max_sAs4p0',
                # 'pvalloc_29nbfs_LRG2_max_sAs6p0',

                # 'pvalloc_29nbfs_LRG2_max_sBs0p4',
                # 'pvalloc_29nbfs_LRG2_max_sBs0p6',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',

                # 'pvalloc_29nbfs_LRG2_max_sCs2p4',
                # 'pvalloc_29nbfs_LRG2_max_sCs2p6',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                
                'pvalloc_29nbfs_LRG2_max_sCs4p4',
                'pvalloc_29nbfs_LRG2_max_sCs4p6',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p8',

                # 'pvalloc_29nbfs_LRG2_max_sCs6p4',
                # 'pvalloc_29nbfs_LRG2_max_sCs6p6',
                'pvalloc_29nbfs_LRG2_max_sCs6p8',

            ],

            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_productionHOY_per_node_byiter_TF  = [True,      True,      False],
            # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 

            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 

            # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],

            # plot_ind_hist_cols_HOYagg_per_EGID_TF         = [True,      True,      False],
            ),  

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [      
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_1hll',
                'pvalloc_29nbfs_LRG2_gridoptim_max',
                'pvalloc_29nbfs_LRG2_gridoptim_max_1hll',

                'pvalloc_29nbfs_LRG2_rnd', 
                'pvalloc_29nbfs_LRG2_max_epzb0_50',
                'pvalloc_29nbfs_LRG2_max_epzb0_75',

                # 'pvalloc_29nbfs_LRG2_max_1hll_sAs2p0',
                'pvalloc_29nbfs_LRG2_max_1hll_sAs4p0',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sAs6p0',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sBs0p4',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sBs0p6',
                'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sCs2p4',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sCs2p6',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sCs2p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs4p4',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sCs4p8',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sCs6p4',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sCs6p6',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs6p8',

            ],

            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_productionHOY_per_node_byiter_TF  = [True,      True,      False],
            # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 

            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 

            # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],

            # plot_ind_hist_cols_HOYagg_per_EGID_TF         = [True,      True,      False],
            ),  

    ]   

asdf = [
        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [       
                # 'DEV_pvalloc_*'                
            'DEV_pvalloc_16nbfs_RUR_max', 
            'DEV_pvalloc_16nbfs_RUR_max_sCs4p6', 
            'DEV_pvalloc_16nbfs_RUR_gridoptim_max', 

            'DEV_pvalloc_10nbfs_SUB_max', 
            'DEV_pvalloc_10nbfs_SUB_max_sCs4p6', 
            'DEV_pvalloc_10nbfs_SUB_gridoptim_max', 

            'DEV_pvalloc_29nbfs_LRG_max', 
            ],
            remove_old_plot_scen_directories   = True,  
            remove_old_plots_in_visualization  = True, 
            remove_old_csvs_in_visualization   = True, 

            cut_timeseries_to_zoom_hour        = True,
            # add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            # plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 
            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            # plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            # plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            # # plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 
            # # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],
            # plot_ind_line_productionHOY_per_node_byiter_TF = [True,      True,      False],
            ),  

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [       
                # 'DEV_pvalloc_*'                
            'DEV_pvalloc_16nbfs_RUR_max', 
            'DEV_pvalloc_16nbfs_RUR_max_sCs4p6', 
            'DEV_pvalloc_16nbfs_RUR_gridoptim_max', 

            'DEV_pvalloc_16nbfs_RUR_max_OLDpreprep', 
            'DEV_pvalloc_16nbfs_RUR_max_sCs4p6_OLDpreprep', 
            'DEV_pvalloc_16nbfs_RUR_gridoptim_max_OLDpreprep', 
            ],

            cut_timeseries_to_zoom_hour        = True,
            # add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 
            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            # # plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 
            # # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],
            # plot_ind_line_productionHOY_per_node_byiter_TF = [True,      True,      False],
            ),  

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [       
                # 'DEV_pvalloc_*'                
            'DEV_pvalloc_10nbfs_SUB_max', 
            'DEV_pvalloc_10nbfs_SUB_max_sCs4p6', 
            'DEV_pvalloc_10nbfs_SUB_gridoptim_max', 
            
            'DEV_pvalloc_10nbfs_SUB_max_OLDpreprep', 
            'DEV_pvalloc_10nbfs_SUB_max_sCs4p6_OLDpreprep', 
            'DEV_pvalloc_10nbfs_SUB_gridoptim_max_OLDpreprep', 
            ],

            cut_timeseries_to_zoom_hour        = True,
            # add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 
            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            # # plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 
            # # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],
            # plot_ind_line_productionHOY_per_node_byiter_TF = [True,      True,      False],

            ),  
    

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [       
                'pvalloc_29nbfs_LRG_max', 
                'pvalloc_29nbfs_LRG_max_sA*', 
                'pvalloc_29nbfs_LRG_max_sB*', 
                'pvalloc_29nbfs_LRG_max_sC*', 
                'pvalloc_29nbfs_LRG_max_epzb*'


            ],

            cut_timeseries_to_zoom_hour        = True,
            # add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 
            # plot_ind_map_topo_egid_TF                       = [True,      True,       False]  ,
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            # # plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 
            # # plot_ind_summary_stats_by_node_TF               = [True,      True,       True],
            # plot_ind_line_productionHOY_per_node_byiter_TF = [True,      True,      False],

            ),  
    ]    


if __name__ == '__main__':

    # visualization ---------------------
    for visual_scen in visualization_list:

        visual_class = Visualization(visual_scen)
        visual_class.plot_ALL()
    
    print('end main_visualization.py')

