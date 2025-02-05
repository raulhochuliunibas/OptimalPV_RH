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

# pv production deviation between method 2 + 3  --------------------------------------------------------------------
# > plot the deviation between the pv production of method 2 and 3 for every single house to check if the difference is a 
# > constant factor or if there are some houses with a higher deviation
# if True:

# mth_pttrn = ['meth2', 'meth3']
# scenarios_to_plot_pattern= [
#     ['pvalloc_BLsml_1roof_extSolkatEGID_12m_',  '.2_rad_dfuid_ind'], 
#     ['pvalloc_BLsml_07roof_extSolkatEGID_12m_',  '2_rad_dfuid_ind'],
# ]
scenario_comparison_groups = {
    'pvalloc_BLsml_24m_methods2vs3': (
        ('pvalloc_BLsml_24m_meth2.2_random', 'meth2.2'), 
        ('pvalloc_BLsml_24m_meth3.2_random', 'meth3.2'),
    ), 
}

wd_path = 'C:/Models/OptimalPV_RH'
data_path     = f'{wd_path}_data'
data_path_def = f'{wd_path}_data'

scen_group = list(scenario_comparison_groups.keys())[0]
for scen_group in scenario_comparison_groups:
    scenarios_in_group = scenario_comparison_groups.get(scen_group)

    # check if all scenarios are available in the output folder
    check_all_scens_of_group_available = []
    scen_tuple = scenarios_in_group[0]
    for scen_tuple in scenarios_in_group:
        check_all_scens_of_group_available.append(scen_tuple[0] in os.listdir(f'{data_path}/output/'))

    if not all(check_all_scens_of_group_available):
        print('*** ERROR *** not all scenarios are available in the output folder')

    elif all(check_all_scens_of_group_available):
        scen_comparison_list = []

        # loop through scens in group
        scen_tuple = scenarios_in_group[0]
        for scen_tuple in scenarios_in_group:
            scen = scen_tuple[0]
            meth = scen_tuple[1]

            # import data
            sanity_scen_data_path = f'{data_path}/output/{scen}/sanity_check_byEGID'
            topo = json.load(open(f'{sanity_scen_data_path}/topo_egid.json', 'r'))
            topo_subdf_paths = glob.glob(f'{sanity_scen_data_path}/topo_subdf_*.parquet')

            # scen settings
            pvalloc_scen = json.load(open(f'{data_path}/output/{scen}/pvalloc_settings.json', 'r'))
            kWpeak_per_m2, share_roof_area_available = pvalloc_scen['tech_economic_specs']['kWpeak_per_m2'],pvalloc_scen['tech_economic_specs']['share_roof_area_available']
            xbins = 0

            # aggregation
            aggdf_combo_list = []
            path = topo_subdf_paths[0]
            for i_path, path in enumerate(topo_subdf_paths):
                subdf = pd.read_parquet(path)

                agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG']).agg({'pvprod_kW': 'sum',}).reset_index()
                aggsub_npry = np.array(agg_subdf)

                egid_list, dfuid_list, flaeche_list, pvprod_list, pvprod_ByTotalPower_list, stromertrag_list = [], [], [], [], [], []

                for egid in subdf['EGID'].unique():
                    mask_egid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('EGID')], egid)
                    df_uids  = list(aggsub_npry[mask_egid_subdf, agg_subdf.columns.get_loc('df_uid')])

                    for r in range(1,len(df_uids)+1):
                        for combo in itertools.combinations(df_uids,r):
                            combo_key_str = '_'.join([str(c) for c in combo])
                            mask_dfuid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))

                            egid_list.append(egid)
                            dfuid_list.append(combo_key_str)
                            flaeche_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECHE')].sum())
                            pvprod_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_kW')].sum())
                            # pvprod_ByTotalPower_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_TotalPower_kW')].sum())
                            stromertrag_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('STROMERTRAG')].sum())
                            
                aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 
                                                'FLAECHE': flaeche_list, 'pvprod_kW': pvprod_list, 
                                                # 'pvprod_ByTotalPower_kW': pvprod_ByTotalPower_list,
                                                'STROMERTRAG': stromertrag_list})
                
                aggdf_combo_list.append(aggsubdf_combo)
                print(f'aggregated subdf: {i_path+1}/{len(topo_subdf_paths)}')
            
            aggdf_combo = pd.concat(aggdf_combo_list, axis=0)
            aggdf_combo['inst_capa_kW'] = aggdf_combo['FLAECHE'] * kWpeak_per_m2 * share_roof_area_available

            scen_comparison_list.append(aggdf_combo)
            print('finished aggregation for method:', meth)
    
    
        # merge scenarios to 1 df
        meth_suffixes = [f'_{scen_tuple[1]}' for scen_tuple in scenarios_in_group]
        merged_scen_df = pd.merge(
            scen_comparison_list[0],
            scen_comparison_list[1],
            on=['EGID', 'df_uid', ],
            suffixes=(meth_suffixes[0], meth_suffixes[1])
        )

        # add difference columns
        merged_scen_df['diff_FLAECHE_abs'] =       merged_scen_df[f'FLAECHE{meth_suffixes[0]}'] -     merged_scen_df[f'FLAECHE{meth_suffixes[1]}']
        merged_scen_df['diff_pvprod_kW_abs'] =     merged_scen_df[f'pvprod_kW{meth_suffixes[0]}'] -   merged_scen_df[f'pvprod_kW{meth_suffixes[1]}']
        merged_scen_df['diff_STROMERTRAG_abs'] =   merged_scen_df[f'STROMERTRAG{meth_suffixes[0]}'] - merged_scen_df[f'STROMERTRAG{meth_suffixes[1]}']
        merged_scen_df['diff_inst_capa_kW_abs'] =  merged_scen_df[f'inst_capa_kW{meth_suffixes[0]}']- merged_scen_df[f'inst_capa_kW{meth_suffixes[1]}']

        merged_scen_df['diff_FLAECHE_rel'] =       merged_scen_df[f'FLAECHE{meth_suffixes[0]}'] /     merged_scen_df[f'FLAECHE{meth_suffixes[1]}']
        merged_scen_df['diff_pvprod_kW_rel'] =     merged_scen_df[f'pvprod_kW{meth_suffixes[0]}'] /   merged_scen_df[f'pvprod_kW{meth_suffixes[1]}']
        merged_scen_df['diff_STROMERTRAG_rel'] =   merged_scen_df[f'STROMERTRAG{meth_suffixes[0]}'] / merged_scen_df[f'STROMERTRAG{meth_suffixes[1]}']
        merged_scen_df['diff_inst_capa_kW_rel'] =  merged_scen_df[f'inst_capa_kW{meth_suffixes[0]}']/ merged_scen_df[f'inst_capa_kW{meth_suffixes[1]}']
        

        # plot ----------------------
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Histogram(
                x=merged_scen_df.loc[merged_scen_df['diff_pvprod_kW_abs'] != 0, 'diff_pvprod_kW_abs'],
                name=f'diff_pvprod_kW abs ({meth_suffixes[0]} - {meth_suffixes[1]})',
                xbins=dict(start=-1, size=500),
                opacity=0.75  # Adjust opacity here
            ),
            secondary_y=False  # Primary y-axis
        )

        # Add the second histogram to the secondary y-axis
        fig.add_trace(
            go.Histogram(
                x=merged_scen_df.loc[merged_scen_df['diff_pvprod_kW_rel'] != 0, 'diff_pvprod_kW_rel'],
                name=f'diff_pvprod_kW rel ({meth_suffixes[0]} / {meth_suffixes[1]})',
                xbins=dict(start=-1, size=0.05),
                opacity=0.75  # Adjust opacity here
            ),
            secondary_y=True  # Secondary y-axis
        )

        # add other histograms for different data points
        fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_FLAECHE_abs'], name = f'diff_FLAECHE abs ({meth_suffixes[0]} - {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))
        fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_FLAECHE_rel'], name = f'diff_FLAECHE rel ({meth_suffixes[0]} / {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))
        fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_STROMERTRAG_abs'], name = f'diff_STROMERTRAG abs ({meth_suffixes[0]} - {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))
        fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_STROMERTRAG_rel'], name = f'diff_STROMERTRAG rel ({meth_suffixes[0]} / {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))
        fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_inst_capa_kW_abs'], name = f'diff_inst_capa_kW abs ({meth_suffixes[0]} - {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))
        fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_inst_capa_kW_rel'], name = f'diff_inst_capa_kW rel ({meth_suffixes[0]} / {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))

        fig.update_layout(title = f'Difference between Method {meth_suffixes[0]} and {meth_suffixes[1]}',
                        xaxis_title = 'Difference [abs]',
                        yaxis_title = 'Frequency',
                        barmode='overlay',
                        )
        fig.update_yaxes(title_text="Frequency", secondary_y=False)
        fig.update_yaxes(title_text="Relative Frequency", secondary_y=True)

        fig.update_traces(opacity=0.75)
        fig.show()
        if not os.path.exists(f'{data_path}/visualizations_individual_plots'):
            os.makedirs(f'{data_path}/visualizations_individual_plots')
        file_name = f'hist_pvCapaProd_comparison_{scen_group}.html'
        fig.write_html(f'{data_path}/visualizations_individual_plots/{file_name}.html')

                  
# all weather years  --------------------------------------------------------------------
# > plot all possible weather years in a single plot for comparision
if False:
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
    if not os.path.exists(f'{data_path}/output/visualizations_individual_plots'):
        os.makedirs(f'{data_path}/output/visualizations_individual_plots')
    plot_by_time_interval_str = '_'.join(plot_by_time_interval)
    file_name = f'all_wheather_years_diff_by_{plot_by_time_interval_str}.html'
    fig.write_html(f'{data_path}/output/visualizations_individual_plots/{file_name}.html')
    print(" END: all weather years  --------------------------------------------------------------------\n\n\n")