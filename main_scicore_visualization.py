
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                '*testX',
                '*pandas_no_outtopo*',
                ], 
            pvalloc_include_pattern_list = [
                '*test2*', 
            ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = True,  
            remove_old_plots_in_visualization = True,  
            ),    
    ]    


if __name__ == '__main__':

    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        plot_method_names = [
            
            # # -- def plot_ALL_init_sanitycheck(self, ): -------------
            # visual_class.plot_ind_var_summary_stats()                     # runs as intended
            # visual_class.plot_ind_hist_pvcapaprod_sanitycheck()           # runs as intended
            # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
            # visual_class.plot_ind_charac_omitted_gwr()                    # runs as intended
            # visual_class.plot_ind_line_meteo_radiation()                  # runs as intended

            # # -- def plot_ALL_mcalgorithm(self,): -------------
            "plot_ind_line_installedCap",                     # runs as intended
            "plot_ind_line_productionHOY_per_node",           # runs as intended
            "plot_ind_line_productionHOY_per_EGID",           # runs as intended
            # "plot_ind_line_PVproduction",                   # runs — optional, uncomment if needed
            "plot_ind_hist_NPV_freepartitions",               # runs as intended
            "plot_ind_line_gridPremiumHOY_per_node",          # runs
            "plot_ind_line_gridPremium_structure",            # runs
            "plot_ind_lineband_contcharact_newinst",          # status not noted
            "plot_ind_map_topo_egid",                         # runs as intended
            "plot_ind_map_topo_egid_incl_gridarea",         # runs as intended — optional
            # "plot_ind_map_node_connections"                   # status not noted
        ]

        for plot_method in plot_method_names:
            try:
                method = getattr(visual_class, plot_method)
                method()
                print(f"Ran successfully: {plot_method}")
            except Exception as e:
                print(f"Error in {plot_method}: {e}")


print('done')

