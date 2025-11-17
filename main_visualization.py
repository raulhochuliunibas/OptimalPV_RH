
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                ], 
            pvalloc_include_pattern_list = [
                # 'pvalloc_2nbfs_2y_testMC', 

                # pvalloc_16nbfs_RUR_max_30y_histgr0-05    
                # pvalloc_16nbfs_RUR_max_30y_eb2050
                # pvalloc_16nbfs_RUR_max_30y_eb2050_1hll
                'pvalloc_16nbfs_RUR_max_*', 

                # pvalloc_10nbfs_SUB_max_30y_histgr0
                # pvalloc_10nbfs_SUB_max_30y_eb2050
                # pvalloc_10nbfs_SUB_max_30y_eb2050_1hll
                'pvalloc_10nbfs_SUB_max_*',

                # pvalloc_29nbfs_30y_histgr0-05
                # pvalloc_29nbfs_30y_eb2050
                # pvalloc_29nbfs_max_30y_eb2050_1hll
                'pvalloc_29nbfs*',

            ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization  = False,  
            remove_old_csvs_in_visualization   = False,
            ),   

    ]    


if __name__ == '__main__':

    # visualization ---------------------
    for visual_scen in visualization_list:

        # # -- def plot_ALL_init_sanitycheck(self, ): --- [run plot,  show plot,  show all scen] ---------
        visual_scen.plot_ind_var_summary_stats_TF                   = [True,      True,       False]
        # visual_scen.plot_ind_hist_pvcapaprod_sanitycheck_TF         = [True,      True,       False]
        # visual_scen.plot_ind_hist_pvprod_deviation_TF               = [True,      True,       False]
        visual_scen.plot_ind_charac_omitted_gwr_TF                  = [True,      True,       False]
        # visual_scen.plot_ind_line_meteo_radiation_TF                = [True,      True,       False]

        # # -- def plot_ALL_mcalgorithm(self,): --------- [run plot,  show plot,  show all scen] ---------
        # visual_scen.plot_ind_line_installedCap_TF                 = [True,      True,       False]    
        # visual_scen.plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False]         
        visual_scen.plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False]         
        visual_scen.plot_ind_line_productionHOY_per_node_TF         = [True,      True,       False]         
        visual_scen.plot_ind_line_PVproduction_TF                   = [True,      True,       False]         
        visual_scen.plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False]         
        # visual_scen.plot_ind_line_gridPremiumHOY_per_node_TF        = [True,      True,       False]         
        # visual_scen.plot_ind_line_gridPremiumHOY_per_EGID_TF        = [True,      True,       False]         
        # visual_scen.plot_ind_line_gridPremium_structure_TF          = [True,      True,       False]         
        # visual_scen.plot_ind_hist_NPV_freepartitions_TF             = [True,      True,       False]         
        visual_scen.plot_ind_map_topo_egid_TF                       = [True,      True,       False]         
        # visual_scen.plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False]         

        visual_scen.plot_ind_lineband_contcharact_newinst_TF        = [True,      True,       False]    
        

        visual_class = Visualization(visual_scen)
        visual_class.plot_ALL_init_sanitycheck()
        visual_class.plot_ALL_mcalgorithm()

