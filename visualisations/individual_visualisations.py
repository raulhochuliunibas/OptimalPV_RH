# PACKAGES --------------------------------------------------------------------
if True:
    import os as os
    import sys
    sys.path.append(os.getcwd())

    # external packages
    import os as os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import json 
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.colors as pc
    import copy
    import glob
    import matplotlib.pyplot as plt
    import winsound
    import itertools

    from datetime import datetime
    from pprint import pformat
    from shapely.geometry import Polygon, MultiPolygon
    from plotly.subplots import make_subplots
    

    # own packages and functions
    import pv_allocation.default_settings as pvalloc_default_sett
    import visualisations.defaults_settings as visual_default_sett

    from auxiliary_functions import chapter_to_logfile, print_to_logfile, checkpoint_to_logfile
    from pv_allocation.default_settings import *
    from visualisations.defaults_settings import *


# all weather years  --------------------------------------------------------------------
# > plot all possible weather years in a single plot for comparision
if True:
    # settings
    plot_by_time_interval = [
        # 'hour', 
        # 'day', 
        # 'week', 
        # 'month', 
        'year'
        ]   # ['hour', 'day', 'week', 'month', 'year']
    incl_std = True


    wd_path = 'C:/Models/OptimalPV_RH'
    data_path     = f'{wd_path}_data'
    data_path_def = f'{wd_path}_data'
    scen = "pvalloc_smallBL_10y_npv_weighted"
    name_dir_import = 'preprep_BL_22to23_1and2homes'

    # import 
    meteo = pd.read_csv(f'{data_path}/input/Meteoblue_BSBL/Meteodaten_Basel_2018_2024_reduziert_bereinigt.csv')
    meteo.dtypes

    # transformations
    meteo['timestamp'] = pd.to_datetime(meteo['timestamp'], format='%d.%m.%Y %H:%M:%S')
    meteo['t']= meteo['timestamp'].apply(lambda x: f't_{(x.dayofyear -1) * 24 + x.hour +1}')
    meteo['t_int'] = meteo['t'].apply(lambda x: int(x.split('_')[1]))
    meteo['t'].value_counts()
    t_only_once = meteo['t'].value_counts()[meteo['t'].value_counts() == 1].index

    schaltjahr_meteo = meteo.loc[meteo['t'].isin(t_only_once)]
    meteo.loc[(meteo['timestamp'] >= '2020-02-28 20:00:00') & (meteo['timestamp'] <= '2020-03-01 00:00:00'), 'timestamp'] 

    # remove incomplete years and hours of schaltjahr
    meteo = meteo.loc[~meteo['t'].isin(t_only_once)]
    meteo = meteo.loc[(meteo['timestamp'] >= '2018-01-01 00:00:00') & (meteo['timestamp'] <= '2023-12-31 23:00:00')] 

    # plot 
    fig = go.Figure()

    if 'hour' in plot_by_time_interval:
        for y in meteo['timestamp'].dt.year.unique():
            fig.add_trace(go.Scatter
                (x = meteo.loc[meteo['timestamp'].dt.year == y, 't_int'],
                y = meteo.loc[meteo['timestamp'].dt.year == y, 'Basel Direct Shortwave Radiation'],
                mode = 'lines',
                name = f'ByHour: Direct_Rad {y}',
                )
            )
        for y in meteo['timestamp'].dt.year.unique():
            fig.add_trace(go.Scatter
                (x = meteo.loc[meteo['timestamp'].dt.year == y, 't_int'],
                y = meteo.loc[meteo['timestamp'].dt.year == y, 'Basel Diffuse Shortwave Radiation'],
                mode = 'lines',
                name = f'ByHour: Diffuse_Rad {y}',)
            )

    if 'day' in plot_by_time_interval:
        meteo_by_day = copy.deepcopy(meteo)
        meteo_by_day['Basel Direct Shortwave Radiation'].value_counts()

        agg_results = meteo_by_day.groupby(meteo_by_day['timestamp'].dt.date).agg( 
            timestamp = ('timestamp', 'first'),
            t = ('t_int', 'first'),
            Direct_Rad_mean = ('Basel Direct Shortwave Radiation', 'mean'),
            Direct_Rad_std = ('Basel Direct Shortwave Radiation', 'std'),
            Diffuse_Rad_mean = ('Basel Diffuse Shortwave Radiation', 'mean'),
            Diffuse_Rad_std = ('Basel Diffuse Shortwave Radiation', 'std'),
            )
        agg_results['timestamp'] = pd.to_datetime(agg_results['timestamp'])

        agg_results.dtypes
        agg_results['Direct_Rad_mean']
        plot_cols = ['Direct_Rad_mean', 'Diffuse_Rad_mean'] if not incl_std else ['Direct_Rad_mean', 'Direct_Rad_std', 'Diffuse_Rad_mean', 'Diffuse_Rad_std']
        for col in plot_cols:
            for y in agg_results['timestamp'].dt.year.unique():
                fig.add_trace(go.Scatter
                    (x = agg_results.loc[agg_results['timestamp'].dt.year == y, 't'],
                    y = agg_results.loc[agg_results['timestamp'].dt.year == y, col],
                    mode = 'lines',
                    name = f'ByDay: {col} {y}',
                    ))
                
    if 'week' in plot_by_time_interval:
        meteo_by_week = copy.deepcopy(meteo)
        meteo_by_week['Basel Direct Shortwave Radiation'].value_counts()

        meteo_by_week['year_week'] = meteo_by_week['timestamp'].dt.to_period('W')
        agg_results = meteo_by_week.groupby(meteo_by_week['year_week']).agg(
            timestamp = ('timestamp', 'first'),
            t = ('t_int', 'first'),
            Direct_Rad_mean = ('Basel Direct Shortwave Radiation', 'mean'),
            Direct_Rad_std = ('Basel Direct Shortwave Radiation', 'std'),
            Diffuse_Rad_mean = ('Basel Diffuse Shortwave Radiation', 'mean'),
            Diffuse_Rad_std = ('Basel Diffuse Shortwave Radiation', 'std'),
            )

        agg_results.dtypes
        plot_cols = ['Direct_Rad_mean', 'Diffuse_Rad_mean'] if not incl_std else ['Direct_Rad_mean', 'Direct_Rad_std', 'Diffuse_Rad_mean', 'Diffuse_Rad_std']
        for col in plot_cols:
            for y in agg_results['timestamp'].dt.year.unique():
                fig.add_trace(go.Scatter
                    (x = agg_results.loc[agg_results['timestamp'].dt.year == y, 't'],
                    y = agg_results.loc[agg_results['timestamp'].dt.year == y, col],
                    mode = 'lines',
                    name = f'ByWeek: {col} {y}',
                    ))
                

    if 'month' in plot_by_time_interval:
        meteo_by_month = copy.deepcopy(meteo)
        meteo_by_month['Basel Direct Shortwave Radiation'].value_counts()

        meteo_by_month['year_month'] = meteo_by_month['timestamp'].dt.to_period('M')
        agg_results = meteo_by_month.groupby(meteo_by_month['year_month']).agg( 
            timestamp = ('timestamp', 'first'),
            t = ('t_int', 'first'),
            Direct_Rad_mean = ('Basel Direct Shortwave Radiation', 'mean'),
            Direct_Rad_std = ('Basel Direct Shortwave Radiation', 'std'),
            Diffuse_Rad_mean = ('Basel Diffuse Shortwave Radiation', 'mean'),
            Diffuse_Rad_std = ('Basel Diffuse Shortwave Radiation', 'std'),
            )

        agg_results.dtypes
        plot_cols = ['Direct_Rad_mean', 'Diffuse_Rad_mean'] if not incl_std else ['Direct_Rad_mean', 'Direct_Rad_std', 'Diffuse_Rad_mean', 'Diffuse_Rad_std']
        for col in plot_cols:
            for y in agg_results['timestamp'].dt.year.unique():
                fig.add_trace(go.Scatter
                    (x = agg_results.loc[agg_results['timestamp'].dt.year == y, 't'],
                    y = agg_results.loc[agg_results['timestamp'].dt.year == y, col],
                    mode = 'lines',
                    name = f'ByMonth: {col} {y}',
                    ))
                
    if 'year' in plot_by_time_interval:
        meteo_by_year = copy.deepcopy(meteo)
        meteo_by_year['Basel Direct Shortwave Radiation'].value_counts()

        agg_results = meteo_by_year.groupby(meteo_by_year['timestamp'].dt.year).agg( 
            timestamp = ('timestamp', 'first'),
            t = ('t', 'first'),
            Direct_Rad_mean = ('Basel Direct Shortwave Radiation', 'mean'),
            Direct_Rad_std = ('Basel Direct Shortwave Radiation', 'std'),
            Diffuse_Rad_mean = ('Basel Diffuse Shortwave Radiation', 'mean'),
            Diffuse_Rad_std = ('Basel Diffuse Shortwave Radiation', 'std'),
            )
        agg_results = agg_results.reset_index(drop=True)
        t_values = [f't_{i}' for i in [1, 8761]]
        # Repeat the rows for each `t` value
        repeated_results = agg_results.loc[agg_results.index.repeat(len(t_values))].reset_index(drop=True)

        # Now assign the `t` values to the expanded rows
        repeated_results['t'] = t_values * len(agg_results)


        plot_cols = ['Direct_Rad_mean', 'Diffuse_Rad_mean'] if not incl_std else ['Direct_Rad_mean', 'Direct_Rad_std', 'Diffuse_Rad_mean', 'Diffuse_Rad_std']
        for col in plot_cols:
            for y in agg_results['timestamp'].dt.year.unique():
                fig.add_trace(go.Scatter
                    (x = repeated_results.loc[repeated_results['timestamp'].dt.year == y, 't'],
                    y = repeated_results.loc[repeated_results['timestamp'].dt.year == y, col],
                    mode = 'lines',
                    name = f'ByYear: {col} {y}',
                    ))
                
    fig.update_layout(title = 'Direct and Diffuse Radiation in Basel 2018-2023',
                    xaxis_title = 'Hour of the Year',
                    yaxis_title = 'Radiation [W/m²]',
                    )
    fig.show()
    if not os.path.exists(f'{data_path}/output/visualizations/individual_visualizations'):
        os.makedirs(f'{data_path}/output/visualizations/individual_visualizations')
    plot_by_time_interval_str = '_'.join(plot_by_time_interval)
    file_name = f'all_wheather_years_diff_by_{plot_by_time_interval_str}.html'
    fig.write_html(f'{data_path}/output/visualizations/individual_visualizations/{file_name}.html')
    print(" END: all weather years  --------------------------------------------------------------------\n\n\n")