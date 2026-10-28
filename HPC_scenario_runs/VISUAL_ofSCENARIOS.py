import os 
import sys


from SCENARIOS import get_subscen_list

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.MAIN_pvallocation_Class import PVAllocScenario_Settings, PVAllocScenario, make_scenario
from src.MAIN_visualization_Class import Visual_Settings, Visualization

# SELECT SUB-SCENARIO TO PLOT --------------------------------------------------------------------------

sub_scen_str = 'DEV'

pvalloc_scen_list = get_subscen_list(sub_scen_str = sub_scen_str, )
scen_to_plot = [scen.name_dir_export for scen in pvalloc_scen_list ]

# ------------------------------------------------------------------------------------------------------


visual_default = Visual_Settings(
        # pvalloc_scen_list              =  scen_to_plot,
        pvalloc_include_pattern_list       = [
            # 'pvalloc_26nbfs_LRG_*', 
            # 'pvalloc_LRG4*', 
            'pvalloc_DEV_*', 
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


if __name__ == "__main__":
    
    visual_class = Visualization(visual_default)
    visual_class.plot_ind_line_PVproduction(plot_show = False, plot_export = True)



