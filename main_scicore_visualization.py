
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                '*testX',
                '*pandas_no_outtopo*',
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

        # # -- def plot_ALL_init_sanitycheck(self, ): -------------
        visual_class.plot_ind_var_summary_stats()                     # runs as intended
        visual_class.plot_ind_hist_pvcapaprod_sanitycheck()           # runs as intended
        # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
        visual_class.plot_ind_charac_omitted_gwr()                    # runs as intended
        visual_class.plot_ind_line_meteo_radiation()                  # runs as intended

        # # -- def plot_ALL_mcalgorithm(self,): -------------
        visual_class.plot_ind_line_installedCap()                     # runs as intended
        visual_class.plot_ind_line_productionHOY_per_node()           # runs as intended
        visual_class.plot_ind_line_productionHOY_per_EGID()           # runs as intended
        # visual_class.plot_ind_line_PVproduction()                     # runs
        visual_class.plot_ind_hist_NPV_freepartitions()               # runs as intended
        visual_class.plot_ind_line_gridPremiumHOY_per_node()          # runs 
        visual_class.plot_ind_line_gridPremium_structure()            # runs 



print('done')

