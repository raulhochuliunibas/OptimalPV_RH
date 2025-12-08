
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [



        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
                '*old*',
                'x_*',
                # 'pvalloc_47nbfs_30y_1hll',
                ], 
            pvalloc_include_pattern_list = [
                'pvalloc_47nbfs_30y*'
            ],
            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 
            # remove_old_plot_scen_directories   = True,  
            # remove_old_plots_in_visualization  = True,  
            # remove_old_csvs_in_visualization   = True,
            
            plot_ind_var_summary_stats_TF                   = [True,      True,      False], 
            plot_ind_line_productionHOY_per_node_TF         = [True,      False,     False],      
            plot_ind_line_PVproduction_TF                   = [True,      False,     False],      
            plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,      True],     
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,      True], 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,      True], 
            plot_ind_summary_stats_by_node_TF               = [True,      True,      True],
            ), 

        # Visual_Settings(
        #     pvalloc_exclude_pattern_list = [
        #         '*.txt','*old_vers*',
        #         '*old*',
        #         'x_*',
        #         # 'pvalloc_47nbfs_30y_1hll',
        #         ], 
        #     pvalloc_include_pattern_list = [
        #         # 'pvalloc_47nbfs_30y*'
        #         'pvalloc_29nbfs_30y3*',
        #     ],
        #     cut_timeseries_to_zoom_hour        = True,
        #     add_day_night_HOY_bands            = True,
        #     save_plot_by_scen_directory        = False, 
            
        #     plot_ind_var_summary_stats_TF                   = [True,      True,      False], 
        #     plot_ind_line_productionHOY_per_node_TF         = [True,      False,     False],      
        #     plot_ind_line_PVproduction_TF                   = [True,      False,     False],      
        #     plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,      True],     
        #     plot_ind_hist_contcharact_newinst_TF            = [True,      True,      True], 
        #     plot_ind_bar_catgcharact_newinst_TF             = [True,      True,      True], 
        #     plot_ind_summary_stats_by_node_TF               = [True,      True,      True],
        #     ), 

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
                '*old*',
                'x_*',
                ], 
            pvalloc_include_pattern_list = [
                'pvalloc_29nbfs_30y3',
                'pvalloc_29nbfs_30y3_1hll',
                'pvalloc_29nbfs_30y3_1hll_ew1pool',
                'pvalloc_47nbfs_30y',
                'pvalloc_47nbfs_30y_1hll',
                'pvalloc_47nbfs_30y_1hll_ew1pool',
                'pvalloc_47nbfs_30y_1hll_ew2pool',
            ],

            # export_only_agg_comparison_plots   = True,# <===
            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 
            plot_ind_var_summary_stats_TF                   = [True,      True,      False], 
            plot_ind_line_PVproduction_TF                   = [True,      False,     False],      
            plot_ind_hist_contcharact_newinst_TF            = [True,      True,      True], 
            plot_ind_bar_catgcharact_newinst_TF             = [True,      True,      True], 
            ), 

    
        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*',
                '*old*',
                'x_*',
                ], 
            pvalloc_include_pattern_list = [
                'pvalloc_16nbfs_RUR', 
                'pvalloc_10nbfs_SUB',
            ],
            cut_timeseries_to_zoom_hour        = True,
            add_day_night_HOY_bands            = True,
            save_plot_by_scen_directory        = False, 
            
            plot_ind_var_summary_stats_TF                   = [True,      True,      False], 
            # plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,      True],     
            # plot_ind_line_PVproduction_TF                   = [True,      False,     False],   
            # plot_ind_hist_contcharact_newinst_TF            = [True,      True,      True], 
            # plot_ind_bar_catgcharact_newinst_TF             = [True,      True,      True], 
            # plot_ind_summary_stats_by_node_TF               = [True,      True,      True],
            ), 

        # Visual_Settings(
        #     pvalloc_exclude_pattern_list = [
        #         '*.txt','*old_vers*',
        #         '*old*',
        #         'x_*',
        #         ], 
        #     pvalloc_include_pattern_list = [
        #         'pvalloc_29nbfs_30y3',
        #         'pvalloc_29nbfs_30y3_ew1pool', 
        #         'pvalloc_29nbfs_30y3_1hll_ew1pool', 
        #         'pvalloc_29nbfs_30y2_ew1first', 
        #         'pvalloc_29nbfs_30y2_1hll_ew1first', 
        #     ],
        #     cut_timeseries_to_zoom_hour        = True,
        #     add_day_night_HOY_bands            = True,
        #     save_plot_by_scen_directory        = False, 
        #     # remove_old_plot_scen_directories   = True,  
        #     # remove_old_plots_in_visualization  = True,  
        #     # remove_old_csvs_in_visualization   = True,
            
        #     plot_ind_var_summary_stats_TF                   = [True,      True,      False], 
        #     plot_ind_line_productionHOY_per_node_TF         = [True,      False,     False],      
        #     plot_ind_line_PVproduction_TF                   = [True,      False,     False],      
        #     plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,      True],     
        #     plot_ind_hist_contcharact_newinst_TF            = [True,      True,      True], 
        #     plot_ind_bar_catgcharact_newinst_TF             = [True,      True,      True], 
        #     plot_ind_summary_stats_by_node_TF               = [True,      True,      True],
        #     ), 

 
    ]    


if __name__ == '__main__':

    # visualization ---------------------
    for visual_scen in visualization_list:

        # # # -- def plot_ALL_init_sanitycheck(self, ): --- [run plot,  show plot,  show all scen] ---------
        # visual_scen.plot_ind_var_summary_stats_TF                   = [True,      True,       False]
        # # visual_scen.plot_ind_hist_pvcapaprod_sanitycheck_TF         = [True,      True,       False]
        # # visual_scen.plot_ind_hist_pvprod_deviation_TF               = [True,      True,       False]
        # visual_scen.plot_ind_charac_omitted_gwr_TF                  = [True,      True,       False]
        # # visual_scen.plot_ind_line_meteo_radiation_TF                = [True,      True,       False]

        # # # -- def plot_ALL_mcalgorithm(self,): --------- [run plot,  show plot,  show all scen] ---------
        # # visual_scen.plot_ind_line_installedCap_TF                 = [True,      True,       False]    
        # # visual_scen.plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False]         
        # visual_scen.plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False]         
        # visual_scen.plot_ind_line_productionHOY_per_node_TF         = [True,      False,       False]         
        # visual_scen.plot_ind_line_PVproduction_TF                   = [True,      False,       False]         
        # visual_scen.plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False]         
        # visual_scen.plot_ind_line_gridPremiumHOY_per_node_TF        = [True,      True,       False]         
        # # visual_scen.plot_ind_line_gridPremiumHOY_per_EGID_TF        = [True,      True,       False]         
        # # visual_scen.plot_ind_line_gridPremium_structure_TF          = [True,      True,       False]         
        # # visual_scen.plot_ind_hist_NPV_freepartitions_TF             = [True,      True,       False]         
        # visual_scen.plot_ind_map_topo_egid_TF                       = [True,      True,       False]         
        # visual_scen.plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       True]         
        # visual_scen.plot_ind_hist_contcharact_newinst_TF            = [True,      True,       True]  
        # visual_scen.plot_ind_bar_catgcharact_newinst_TF             = [True,      True,       True]  

        # visual_scen.plot_ind_lineband_contcharact_newinst_TF        = [True,      True,       False]    

        

        visual_class = Visualization(visual_scen)
        visual_class.plot_ALL()
    
    print('end main_visualization.py')


