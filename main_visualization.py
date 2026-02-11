
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [
            # export_only_agg_comparison_plots   = True,# <===
            # remove_old_plot_scen_directories   = True,  
            # remove_old_plots_in_visualization  = True, 
            # remove_old_csvs_in_visualization   = True, 
        # Visual_Settings(
        #     pvalloc_exclude_pattern_list = [
        #         '*.txt','*old_vers*',
        #     ], 
        #     pvalloc_include_pattern_list = [
        #         'pvalloc_29nbfs_30y_max',
        #     ],
            
        #     # export_only_agg_comparison_plots   = True,# <===
        #     remove_old_plot_scen_directories   = True,  
        #     remove_old_plots_in_visualization  = True, 
        #     remove_old_csvs_in_visualization   = True, 

        #     cut_timeseries_to_zoom_hour        = True,
        #     add_day_night_HOY_bands            = True,
        #     plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False]          ,
        # ),


        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [
                # 'pvalloc_16nbfs_RUR_max', 
                # 'pvalloc_16nbfs_RUR_max_gridoptim', 
                # 'pvalloc_16nbfs_RUR_max_sCs4p6', 
                # 'pvalloc_16nbfs_RUR_max_1hll', 
                # 'pvalloc_16nbfs_RUR_max_gridoptim_1hll', 
                # 'pvalloc_16nbfs_RUR_max_1hll_sCs4p6', 
                'pvalloc_29nbfs_30y_max',
                'pvalloc_29nbfs_30y_max_1hll',
                'pvalloc_29nbfs_30y_gridoptim_max',
                'pvalloc_29nbfs_30y_gridoptim_max_1hll',
                'pvalloc_29nbfs_30y_rnd',
                'pvalloc_29nbfs_30y_rnd_1hll',
                'pvalloc_29nbfs_30y_max_sAs4p0',
                'pvalloc_29nbfs_30y_max_1hll_sAs4',
                'pvalloc_29nbfs_30y_max_sCs4p6',
                'pvalloc_29nbfs_30y_max_1hll_sCs4p6',               
            ],
            
            # export_only_agg_comparison_plots   = True,# <===
            # remove_old_plot_scen_directories   = True,  
            # remove_old_plots_in_visualization  = True, 
            # remove_old_csvs_in_visualization   = True, 

            cut_timeseries_to_zoom_hour        = True,
            # add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 
            plot_ind_summary_stats_by_node_TF               = [True,      True,       True],
            # plot_ind_hist_NPV_freepartitions_TF             = [True,      True,       False],

            plot_ind_line_productionHOY_per_node_byiter_TF = [True,      True,      False],

            ), 

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
            ], 
            pvalloc_include_pattern_list = [
                'pvalloc_47nbfs_30y_max',
                'pvalloc_47nbfs_30y_max_1hll',
                'pvalloc_47nbfs_30y_gridoptim_max',
                'pvalloc_47nbfs_30y_gridoptim_max_1hll',
                'pvalloc_47nbfs_30y_rnd',
                'pvalloc_47nbfs_30y_rnd_1hll',
                'pvalloc_47nbfs_30y_max_sAs4p0',
                'pvalloc_47nbfs_30y_max_1hll_sAs4',
                'pvalloc_47nbfs_30y_max_sCs4p6',
                'pvalloc_47nbfs_30y_max_1hll_sCs4p6',               
            ],
            
            # export_only_agg_comparison_plots   = True,# <===
            # remove_old_plot_scen_directories   = True,  
            # remove_old_plots_in_visualization  = True, 
            # remove_old_csvs_in_visualization   = True, 

            cut_timeseries_to_zoom_hour        = True,
            # add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 

            plot_ind_line_PVproduction_bynode_specs  = {
                'select_nodes_stacked_traces': [], 
                'n_top_loss_nodes': 0,
                },
            
            plot_ind_line_productionHOY_per_node_TF         = [True,      True,      False],
            plot_ind_line_PVproduction_TF                   = [True,      True,       False]    , 
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]  ,
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  , 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  , 
            plot_ind_summary_stats_by_node_TF               = [True,      True,       True],
            # plot_ind_hist_NPV_freepartitions_TF             = [True,      True,       False],

            plot_ind_line_productionHOY_per_node_byiter_TF = [True,      True,      False],

            ), 

 
    ]    


if __name__ == '__main__':

    # visualization ---------------------
    for visual_scen in visualization_list:

        visual_class = Visualization(visual_scen)
        visual_class.plot_ALL()
    
    print('end main_visualization.py')

