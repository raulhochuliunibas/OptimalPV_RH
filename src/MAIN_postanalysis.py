import sys
import os as os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr, get_bfsnr_name_tuple_list
from src.postanalysis.post_utils import calendar_week_to_date
from src.postanalysis.post_worstnode_worstweek import WorstNodes, BatteryBuffer_WorstWeek
from src.postanalysis.post_econ_driven_constrcapa import EconDrivenConstrcapa


@dataclass
class PostAnalysis_Settings:
    pvalloc_scen_list : List[str]                = field(default_factory=lambda: [])
    pvalloc_exclude_pattern_list : List[str]     = field(default_factory=lambda: [
                                                    '*.txt',
                                                    '*old_vers*',
                                                    '*old*',
                                                    'x_*',
                                                    ])
    pvalloc_include_pattern_list : List[str]      = field(default_factory=lambda: [])
    plot_show: bool                              = True
    save_plot_by_scen_directory: bool            = True


class PostAnalysis(
    WorstNodes,
    BatteryBuffer_WorstWeek,
    EconDrivenConstrcapa
    ):
    def __init__(self, settings: PostAnalysis_Settings,):
        self.settings = settings

        self.data_path               = os.path.join('C:', os.sep, 'Models', 'OptimalPV_RH', 'data')
        self.dir_path_postanalysis   = os.path.join('C:', os.sep, 'Models', 'OptimalPV_RH', 'data', 'postanalysis')
        self.dir_path_constrcapa_adj = os.path.join('C:', os.sep, 'Models', 'OptimalPV_RH', 'data', 'postanalysis', 'constracpa_adjustments')
        for dir in [self.dir_path_postanalysis, self.dir_path_constrcapa_adj]:
            os.makedirs(dir, exist_ok=True)

        self.plot_width    = 5
        self.plot_height   = 5
        self.plot_dpi      = 600
        self.line_opacity  = 0.8


if __name__ == "__main__":
    # Example usage
    settings = PostAnalysis_Settings(
        pvalloc_scen_list=['pvalloc_LRG3_max'],
        plot_show=True,
    )
    post_analysis = PostAnalysis(settings)

    # -- DERIVE NPV based CONSTRUCTION CAPACITY ------------------------------
    # working combination 
        # elecpri_Rp_kWh_base = 31.98,
        # pvtarif_Rp_kWh_base = 9.03,
            # ('adj_3',   40.0,    6.0,    0),
        # elecpri_Rp_kWh_base = 31.98,
        # pvtarif_Rp_kWh_base = 6.2
            # ('adj_24',   30.0,    5.0,    0),
            
    post_analysis.econ_driven_constrcapa_analysis(
        scen = 'pvalloc_LRG3_max',
        show_export_plots = (False, True),
        # elecpri_Rp_kWh_base = 31.98,
        # pvtarif_Rp_kWh_base = 6.2,
        adjustment_tupls = [
            # suffix,   elecpri, pvtarif, ndisregard_tail
            ('adj_10',   31.98,   9.03,   1),
            ('adj_11',   31.98,   6.2,    0),
            ('adj_12',   40.0,    9.0,    0),
            ('adj_13',   40.0,    6.0,    0),
            ],
    )
    post_analysis.econ_driven_constrcapa_analysis(
        scen = 'pvalloc_LRG3_max',
        show_export_plots = (False, True),
        elecpri_Rp_kWh_base = 31.98,
        pvtarif_Rp_kWh_base = 6.2,
        adjustment_tupls = [
            # suffix,   elecpri, pvtarif, ndisregard_tail
            ('adj_20',   31.98,   9.03,   0),
            ('adj_21',   31.98,   6.2,    1),
            ('adj_22',   40.0,    9.0,    0),
            ('adj_23',   40.0,    6.0,    0),
            ('adj_24',   30.0,    5.0,    0),
            ('adj_25',   30.0,    4.0,    0),
            ],
    )
        
 
    # -- Excess Feedin - LAST iterations ------------------------------
    # post_analysis.excfeedin_freq_TS(
    #                     scen = 'pvalloc_LRG3_max',
    #                     freq = 'weekly',
    #                     title = 'Excess Feed-in - Frequency Time Series',
    #                     export_name = 'excfeedin_freq_TS',
    #                     excess_rgb_line = (214, 39, 40),
    #                     nodes_share_rgb_line = (31, 119, 180),
    #                     houses_share_rgb_line = (44, 160, 44),
    #                     linewidth = 1.5,
    #                     legend_loc = 'upper left',
    #                     )
    # post_analysis.excessfeedin_worstnodes_bars(
    #         n_bars = 100, 
    #         export_name = 'excessfeedin_worstnodes_n100_bars',
    #         plot_width_func  = 8, plot_height_func = 12
    #         )
    # post_analysis.excessfeedin_worstnodes_bars(
    #         n_bars = 50, 
    #         export_name = 'excessfeedin_worstnodes_n50_bars',
    #         plot_width_func  = 8, plot_height_func = 12
    #         )
    # top5nodes = post_analysis.excessfeedin_worstnodes_bars(
    #         n_bars = 20, 
    #         export_name = 'excessfeedin_worstnodes_n20_bars',
    #         plot_width_func  = 6, plot_height_func = 10
    #         )

    # # Excess Feedin - 2031+ Year
    # for freq in ['weekly', 'daily']:
    #     post_analysis.excfeedin_freq_TS(
    #                     scen = 'pvalloc_LRG3_max',
    #                     # freq = 'weekly',
    #                     freq = freq,
    #                     title = 'Excess Feed-in - Frequency Time Series',
    #                     export_name = 'excfeedin_freq_TS',
    #                     iter_list = [7, 8, 30],
    #                     excess_rgb_line = (214, 39, 40),
    #                     nodes_share_rgb_line = (31, 119, 180),
    #                     houses_share_rgb_line = (44, 160, 44),
    #                     linewidth = 1.5,
    #                     legend_loc = 'upper left',
    #                     )
    # top5nodes = post_analysis.excessfeedin_worstnodes_bars(
    #         n_bars = 20, 
    #         iter_list = [7, 8],
    #         export_name = 'excessfeedin_worstnodes_n20_bars',
    #         plot_width_func  = 6, plot_height_func = 10
    #         )

    # Battery Buffer - 10 Years
    # post_analysis.worstnode_worstweek()
    date = calendar_week_to_date(2025, 18, 1)

    # post_analysis.maxbatterycap_to_buffer(
    #     # start_date = "2025-04-28",
    #     start_date = "2025-07-17",
    #     # nodes = ["524", "10", "734", "751"],
    #     nodes = ["751"],
    #     n_days_buffer = 60,
    # )





    print(f"\n----- End of file: {os.path.basename(__file__)} -----")

