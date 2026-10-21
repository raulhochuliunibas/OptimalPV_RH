# python integrated packages
import sys
import os as os

# external packages
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLineMeteoRadiation(VisualizationCore):
    """Plot of the pvalloc_initalization + sanitycheck stage."""

    def plot_ind_line_meteo_radiation(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_meteo_radiation', self.visual_sett.log_name)

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            meteo_col_dir_radiation = self.pvalloc_scen.WEAspec_meteo_col_dir_radiation
            meteo_col_diff_radiation = self.pvalloc_scen.WEAspec_meteo_col_diff_radiation
            meteo_col_temperature = self.pvalloc_scen.WEAspec_meteo_col_temperature

            # import meteo data -----
            meteo = pd.read_parquet(f'{self.visual_sett.data_path}/pvalloc/{scen}/meteo_ts.parquet')

            # transform to zoom to smaller time series
            meteo['t_int'] = meteo['t'].str.extract(r't_(\d+)').astype(int)
            if self.visual_sett.cut_timeseries_to_zoom_hour:
                meteo = meteo[
                    (meteo['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                    (meteo['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                ].copy()

            # try to also get raw data to show how radidation is derived
            try: 
                meteo_raw = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/meteo.parquet')
                meteo_raw = meteo_raw.loc[meteo_raw['timestamp'].isin(meteo['timestamp'])]
                meteo_raw[meteo_col_temperature] = meteo_raw[meteo_col_temperature].astype(float)
            except Exception:
                print('... no raw meteo data available')

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            try:  # necessary to accomodate older code versions where radiation is not strictly split into direct and diffuse
                fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo[['rad_direct', 'rad_diffuse']].sum(axis = 1), name='Radiation [W/m^2]'))
            except Exception:
                fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo['radiation'], name='Radiation [W/m^2]'))
            fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo['temperature'], name='Temperature [°C]'), secondary_y=True)

            radiation_cols = [meteo_col_dir_radiation, meteo_col_diff_radiation]
            try: 
                for col in radiation_cols:
                    fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[col], name=f'Rad. raw data: {col}'))

                fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[radiation_cols].sum(axis=1), name='Rad. raw data: sum of rad types'))
                fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[meteo_col_temperature], name=f'Temp. raw data: {meteo_col_temperature}'), secondary_y=True)
            except Exception:
                pass

            fig.update_layout(title_text = f'Meteo Data: Temperature and Radiation (if Direct & Diffuse. flat_diffuse_rad_factor: {self.pvalloc_scen.WEAspec_flat_diffuse_rad_factor})')
            fig.update_xaxes(title_text='Time')
            fig.update_yaxes(title_text='Radiation [W/m^2]', secondary_y=False)
            fig.update_yaxes(title_text='Temperature [°C]', secondary_y=True)

            fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)

            # recalc date and time to zoom to the correct HOY
            start_date = pd.Timestamp(f'{self.pvalloc_scen.WEAspec_weather_year}-01-01 00:00:00')
            timestamp_zoom_start = start_date + pd.to_timedelta(self.visual_sett.default_zoom_hour[0]-1, unit='h')
            timestamp_zoom_end = start_date + pd.to_timedelta(self.visual_sett.default_zoom_hour[1]-1, unit='h')
            fig.update_layout(xaxis_range=[timestamp_zoom_start, timestamp_zoom_end])  


            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_line_meteo_radiation.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_line_meteo_radiation.html')
                print_to_logfile(f'\texport: plot_ind_line_meteo_radiation.html (for: {scen})', self.visual_sett.log_name)
