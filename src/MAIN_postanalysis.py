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
    BatteryBuffer_WorstWeek
    ):
    def __init__(self, settings: PostAnalysis_Settings, export_dir_list= ['data', 'postanalysis']):
        self.settings = settings

        self.data_path        = os.path.join('C:', os.sep, 'Models', 'OptimalPV_RH', 'data')
        self.dir_path_export  = os.path.join('C:', os.sep, 'Models', 'OptimalPV_RH', *export_dir_list)
        os.makedirs(self.dir_path_export, exist_ok=True)

        self.plot_width    = 8
        self.plot_height   = 4
        self.plot_dpi      = 500
        self.line_opacity  = 0.8


if __name__ == "__main__":
    # Example usage
    settings = PostAnalysis_Settings(
        pvalloc_scen_list=['pvalloc_LRG3_max'],
        plot_show=True,
    )
    post_analysis = PostAnalysis(settings)

    # # Excess Feedin - LAST iterations
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

    # Excess Feedin - 2031+ Year
    for freq in ['weekly', 'daily']:
        post_analysis.excfeedin_freq_TS(
                        scen = 'pvalloc_LRG3_max',
                        # freq = 'weekly',
                        freq = freq,
                        title = 'Excess Feed-in - Frequency Time Series',
                        export_name = 'excfeedin_freq_TS',
                        iter_list = [7, 8, 30],
                        excess_rgb_line = (214, 39, 40),
                        nodes_share_rgb_line = (31, 119, 180),
                        houses_share_rgb_line = (44, 160, 44),
                        linewidth = 1.5,
                        legend_loc = 'upper left',
                        )
    # top5nodes = post_analysis.excessfeedin_worstnodes_bars(
    #         n_bars = 20, 
    #         iter_list = [7, 8],
    #         export_name = 'excessfeedin_worstnodes_n20_bars',
    #         plot_width_func  = 6, plot_height_func = 10
    #         )

    # Battery Buffer - 10 Years
    # post_analysis.worstnode_worstweek()
    calendar_week_to_date(2025, 18, 1)
    post_analysis.maxbatterycap_to_buffer(
        # start_date = "2025-04-28",
        start_date = "2025-07-17",
        # nodes = ["524", "10", "734", "751"],
        nodes = ["751"],
        n_days_buffer = 60,
    )



    



    print(f"\n----- End of file: {os.path.basename(__file__)} -----")

