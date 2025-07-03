
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                '*testX',
                '*pandas_no_outtopo*',
                '*test2_*'
                ], 
            pvalloc_include_pattern_list = [
                '*test2a*_selfcons1',
                # 'pvalloc_BLsml_test2_default_w_pvdf_adjust_max_FORPLOT',  
            ],
            # save_plot_by_scen_directory        = True, 
            # remove_old_plot_scen_directories   = True,  
            # remove_old_plots_in_visualization  = True,  
            # remove_old_csvs_in_visualization   = True,
            ),   
        Visual_Settings(
            pvalloc_include_pattern_list = ['*test2b*_selfcons1',],),  
        Visual_Settings(
            pvalloc_include_pattern_list = ['*test2c*_selfcons1',],),   
            
            
        Visual_Settings(
            pvalloc_include_pattern_list = ['pvalloc*_default_rnd_selfcons1',],),         
        Visual_Settings(
            pvalloc_include_pattern_list = ['pvalloc*_southfacing_rnd_selfcons1',],),
        Visual_Settings(
            pvalloc_include_pattern_list = ['pvalloc*_eastwestfacing_rnd_selfcons1',],),
        Visual_Settings(
            pvalloc_include_pattern_list = ['pvalloc*_default_max_selfcons1',],),

    ]    


if __name__ == '__main__':

    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        plot_method_names = [

            # # # -- def plot_ALL_init_sanitycheck(self, ): -------------
            # "plot_ind_var_summary_stats",                      # runs as intended
            # "plot_ind_hist_pvcapaprod_sanitycheck",            # runs as intended
            # # # "plot_ind_boxp_radiation_rng_sanitycheck", 
            # "plot_ind_charac_omitted_gwr",                     # runs as intended
            # # "plot_ind_line_meteo_radiation",                   # runs as intended

            # # -- def plot_ALL_mcalgorithm(self,): -------------
            # # "plot_ind_line_installedCap",                     # runs as intended
            # "plot_ind_line_productionHOY_per_node",           # runs as intended
            # "plot_ind_line_productionHOY_per_EGID",           # runs as intended
            "plot_ind_hist_cols_HOYagg_per_EGID",                    # runs as intended
            # "plot_ind_line_PVproduction",                   # runs — optional, uncomment if needed
            # "plot_ind_hist_NPV_freepartitions",               # runs as intended
            # # "plot_ind_line_gridPremiumHOY_per_node",          # runs
            # # "plot_ind_line_gridPremium_structure",            # runs
            # # "plot_ind_lineband_contcharact_newinst",          # status not noted
            # "plot_ind_map_topo_egid",                         # runs as intended
            # "plot_ind_map_topo_egid_incl_gridarea",         # runs as intended — optional
            # # "plot_ind_map_node_connections"                   # status not noted        
        
            # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
            # plot_ind_map_node_connections()
            # plot_ind_map_omitted_egids()
            # "plot_ind_line_installedCap",                     # runs as intended

        ]

        for plot_method in plot_method_names:
            method = getattr(visual_class, plot_method)
            method()

