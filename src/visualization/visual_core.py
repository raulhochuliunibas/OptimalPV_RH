# python integrated packages
import sys
import os as os
import fnmatch
import glob
import json

# external packages
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import print_to_logfile
from src.MAIN_pvallocation_Class import PVAllocScenario_Settings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.MAIN_visualization_Class import Visual_Settings


class VisualizationCore:
    """
    Base class of all visualization plot classes (src/visualization/plot_*.py); the Visualization class in
    MAIN_visualization_Class.py inherits from all of them. Holds the scenario settings, the list of pvalloc
    scenarios to visualize and the plot-auxiliary helpers shared by every plot.

    Every plot method takes the same two kwargs:
        plot_show:   open the figure(s) in the browser
        plot_export: write the figure(s) (html) and plot data (csv) to disk; False writes nothing
    Whether a plot is run at all is decided by calling its method (or not).
    """

    def __init__(self, settings: "Visual_Settings"):
        self.visual_sett = settings

        # SETUP --------------------
        self.visual_sett.wd_path = os.getcwd()
        self.visual_sett.data_path = os.path.join(self.visual_sett.wd_path, 'data')
        self.visual_sett.visual_path = os.path.join(self.visual_sett.data_path, 'visualization')
        self.visual_sett.visual_static_export_path = os.path.join(self.visual_sett.data_path, 'visualization_static_wpaper')
        self.visual_sett.log_name = f'{self.visual_sett.visual_path}/visual_log.txt'

        os.makedirs(self.visual_sett.visual_path, exist_ok=True)


        if self.visual_sett.pvalloc_scen_list == []:
            # create a str list of scenarios in pvalloc to visualize (include/exclude by pattern recognition)
            filtered_scen_list = os.listdir(f'{self.visual_sett.data_path}/pvalloc')

            # include scens by pattern
            if not self.visual_sett.pvalloc_include_pattern_list == []:
                filtered_scen_list: list[str] = [
                    scen for scen in filtered_scen_list
                    if any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_include_pattern_list)
                ]

            # exclude scens by pattern
            if not self.visual_sett.pvalloc_exclude_pattern_list == []:
                filtered_scen_list: list[str] = [
                    scen for scen in filtered_scen_list 
                    if not any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_exclude_pattern_list)
                    # if not any(fnmatch.fnmatch(scen, pattern) for pattern in self.visual_sett.pvalloc_exclude_pattern_list)
                ]

            self.visual_sett.pvalloc_scen_list = sorted(filtered_scen_list)



    def get_pvalloc_sett_output(self,pvalloc_scen_name):
        # pickle_path = glob.glob(f'{self.data_path}/pvalloc/{pvalloc_scen_name}/*.pkl')[0]
        # with open(pickle_path, 'rb') as f:
        #     pvalloc_scen = pickle.load(f)
        sett_json_path = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{pvalloc_scen_name}/*pvalloc_sett*.json')[0]
        with open(sett_json_path, 'r') as f:
            sett_json = json.load(f)

        self.pvalloc_scen = PVAllocScenario_Settings()
        for key, value in sett_json.items():
            if hasattr(self.pvalloc_scen, key):
                setattr(self.pvalloc_scen, key, value)


    def add_scen_name_to_plot(self, fig_func, scen, pvalloc_scen):
        # add scenario name
        fig_func.add_annotation(
            text=f'Scen: {scen}, (start T0: {self.pvalloc_scen.T0_prediction.split(" ")[0]}, {self.pvalloc_scen.months_prediction} prediction months)',
            xref="paper", yref="paper",
            x=0.5, y=1.05, showarrow=False,
            font=dict(size=12)
        )
        return fig_func


    def add_day_night_bands_HOY_plot(self, fig, t_int_array, start_hour=6, end_hour=20, color_day='rgba(255,255,200,0.7)', color_night='rgba(180,200,255,0.7)'):
        # Assume t_int_array is sorted and covers all hours in the year
        t_int_array = np.array(t_int_array)
        min_t = t_int_array.min()
        max_t = t_int_array.max()

        # For each day, add a vrect for the night period (22:00 to 07:00 next day)
        hours_per_day = 24
        n_days = int(np.ceil((max_t - min_t + 1) / hours_per_day))
        for day in range(n_days):
            # Calculate hour indices for this day
            day_start = min_t + day * hours_per_day
            night1_start = day_start
            night1_end = day_start + start_hour  # 00:00 to 07:00
            night2_start = day_start + end_hour
            night2_end = day_start + hours_per_day  # 22:00 to 24:00

            # Night: 00:00 to XY:00
            fig.add_vrect(
                x0=night1_start, x1=night1_end,
                fillcolor=color_night, opacity=0.2, layer="below", line_width=0
            )
            # Night: XY:00 to 24:00
            fig.add_vrect(
                x0=night2_start, x1=night2_end,
                fillcolor=color_night, opacity=0.2, layer="below", line_width=0
            )
            # Optional: Day band (07:00 to 22:00)
            fig.add_vrect(
                x0=night1_end, x1=night2_start,
                fillcolor=color_day, opacity=0.1, layer="below", line_width=0
            )
        return fig


    def set_default_fig_zoom_year(self, fig, zoom_window, df, datecol):
        start_zoom = pd.to_datetime(f'{zoom_window[0]}-01-01')
        max_date = df[datecol].max() + pd.DateOffset(years=1)
        if pd.to_datetime(f'{zoom_window[1]}-01-01') > max_date:
            end_zoom = max_date
        else:
            end_zoom = pd.to_datetime(f'{zoom_window[1]}-01-01')
        fig.update_layout(
            xaxis = dict(range=[start_zoom, end_zoom])
        )
        return fig 


    def set_default_fig_zoom_hour(self, fig, zoom_window):
        start_zoom, end_zoom = zoom_window[0], zoom_window[1]
        fig.update_layout(
            xaxis_range=[start_zoom, end_zoom])

        return fig


    def flatten_geometry(self, geom):
        if geom.has_z:
            if geom.geom_type == 'Polygon':
                exterior = [(x, y) for x, y, z in geom.exterior.coords]
                interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                return Polygon(exterior, interiors)
            elif geom.geom_type == 'MultiPolygon':
                return MultiPolygon([self.flatten_geometry(poly) for poly in geom.geoms])
        return geom


    def _set_mc_data_path_or_skip(self, scen):
        mc_pattern = f'{self.visual_sett.data_path}/pvalloc/{scen}/{self.visual_sett.MC_subdir_for_plot}'
        mc_paths = sorted(glob.glob(mc_pattern))

        if len(mc_paths) == 0:
            print_to_logfile(
                f'\tSKIP scen (missing MC folder): {scen} | pattern: {self.visual_sett.MC_subdir_for_plot}',
                self.visual_sett.log_name
            )
            return False

        if len(mc_paths) > 1:
            print_to_logfile(
                f'\tINFO multiple MC folders for {scen}; using first sorted match: {os.path.basename(mc_paths[0])}',
                self.visual_sett.log_name
            )

        self.visual_sett.mc_data_path = mc_paths[0]
        return True
