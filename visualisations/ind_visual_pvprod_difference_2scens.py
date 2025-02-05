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
run_on_server = True
scenario_comparison_groups = {
    # 'pvalloc_BLsml_24m_random_methods2vs3': (
    #     ('pvalloc_BLsml_24m_meth2.2_random', 'meth2.2'), 
    #     ('pvalloc_BLsml_24m_meth3.2_random', 'meth3.2'),
    # ), 
    # 'pvalloc_BLsml_24m_npvweight_methods2vs3': (
    #     ('pvalloc_BLsml_24m_meth2.2_npvweight', 'meth2.2'), 
    #     ('pvalloc_BLsml_24m_meth3.2_npvweight', 'meth3.2'),
    # ), 

    'pvalloc_BLSOmed_48m_random_methods2vs3': (
        ('pvalloc_BLSOmed_48m_meth2.2_random', 'meth2.2'), 
        ('pvalloc_BLSOmed_48m_meth3.2_random', 'meth3.2'),
    ), 
    'pvalloc_BLSOmed_48m_npvweight_methods2vs3': (
        ('pvalloc_BLSOmed_48m_meth2.2_npvweight', 'meth2.2'), 
        ('pvalloc_BLSOmed_48m_meth3.2_npvweight', 'meth3.2'),
    ),  

}

plot_additional_histograms_list = ['FLAECHE', ]# 'STROMERTRAG', 'inst_capa_kW']

wd_path = os.getcwd()
data_path     = f'{wd_path}_data'
data_path_def = f'{wd_path}_data'


# PLOT SCENARIO COMPARISON --------------------------------------------------------------------
print('\n\nSTART: PLOT SCENARIO COMPARISON: \n> pv production deviation between methods')

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
        print('scenarios missing:', [scen_tuple[0] for scen_tuple in scenarios_in_group if scen_tuple[0] not in os.listdir(f'{data_path}/output/')])

    elif all(check_all_scens_of_group_available):
        scen_comparison_list = []

        # loop through scens in group
        scen_tuple = scenarios_in_group[0]
        for scen_tuple in scenarios_in_group:
            scen = scen_tuple[0]
            meth = scen_tuple[1]
            print('start aggregation for method:', meth)

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
        meth_suffixes = [f'{scen_tuple[1]}' for scen_tuple in scenarios_in_group]
        merged_scen_df = pd.merge(
            scen_comparison_list[0],
            scen_comparison_list[1],
            on=['EGID', 'df_uid', ],
            suffixes=(f'_{meth_suffixes[0]}', f'_{meth_suffixes[1]}')
        )

        # add difference columns
        merged_scen_df['diff_pvprod_kW_abs'] =     merged_scen_df[f'pvprod_kW_{meth_suffixes[0]}'] -   merged_scen_df[f'pvprod_kW_{meth_suffixes[1]}']
        merged_scen_df['diff_FLAECHE_abs'] =       merged_scen_df[f'FLAECHE_{meth_suffixes[0]}'] -     merged_scen_df[f'FLAECHE_{meth_suffixes[1]}']
        merged_scen_df['diff_STROMERTRAG_abs'] =   merged_scen_df[f'STROMERTRAG_{meth_suffixes[0]}'] - merged_scen_df[f'STROMERTRAG_{meth_suffixes[1]}']
        merged_scen_df['diff_inst_capa_kW_abs'] =  merged_scen_df[f'inst_capa_kW_{meth_suffixes[0]}']- merged_scen_df[f'inst_capa_kW_{meth_suffixes[1]}']

        merged_scen_df['diff_pvprod_kW_rel'] =     merged_scen_df[f'pvprod_kW_{meth_suffixes[0]}'] /   merged_scen_df[f'pvprod_kW_{meth_suffixes[1]}']
        merged_scen_df['diff_FLAECHE_rel'] =       merged_scen_df[f'FLAECHE_{meth_suffixes[0]}'] /     merged_scen_df[f'FLAECHE_{meth_suffixes[1]}']
        merged_scen_df['diff_STROMERTRAG_rel'] =   merged_scen_df[f'STROMERTRAG_{meth_suffixes[0]}'] / merged_scen_df[f'STROMERTRAG_{meth_suffixes[1]}']
        merged_scen_df['diff_inst_capa_kW_rel'] =  merged_scen_df[f'inst_capa_kW_{meth_suffixes[0]}']/ merged_scen_df[f'inst_capa_kW_{meth_suffixes[1]}']
        

        # plot histograms ----------------------
        if True: 
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Absolute prod difference, first yaxis 
            fig.add_trace(
                go.Histogram(
                    x=merged_scen_df.loc[merged_scen_df['diff_pvprod_kW_abs'] != 0, 'diff_pvprod_kW_abs'],
                    name=f'diff_pvprod_kW abs ({meth_suffixes[0]} - {meth_suffixes[1]})',
                    xbins=dict(start=-1, size=500),
                    opacity=0.75  # Adjust opacity here
                ),
                secondary_y=False 
            )
            
            # Relative prod difference, second yaxis
            fig.add_trace(
                go.Histogram(
                    x=merged_scen_df.loc[merged_scen_df['diff_pvprod_kW_rel'] != 0, 'diff_pvprod_kW_rel'],
                    name=f'diff_pvprod_kW rel ({meth_suffixes[0]} / {meth_suffixes[1]})',
                    xbins=dict(start=-1, size=0.05),
                    opacity=0.75  # Adjust opacity here
                ),
                secondary_y=True 
            )

            # add other histograms for different data points
            for add_hist_name in plot_additional_histograms_list:

                # add traces only if there is a difference between the scenarios
                if sum(merged_scen_df[f'diff_{add_hist_name}_abs']) != 0:
                    fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_{add_hist_name}_abs'], name = f'diff_{add_hist_name} abs ({meth_suffixes[0]} - {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))
                if sum(merged_scen_df[f'diff_{add_hist_name}_rel']) != 1:
                    fig.add_trace(go.Histogram(x = merged_scen_df[f'diff_{add_hist_name}_rel'], name = f'diff_{add_hist_name} rel ({meth_suffixes[0]} / {meth_suffixes[1]})', xbins=dict(start=-1, size=0.1)))

                # otherwise, add the distribution of the data points themselves
                if sum(merged_scen_df[f'diff_{add_hist_name}_abs']) == 0:
                    fig.add_trace(go.Histogram(x = merged_scen_df[f'{add_hist_name}_{meth_suffixes[0]}'], name = f'{add_hist_name} ({meth_suffixes[0]})', xbins=dict(start=-1, size=0.1)))

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
            file_name = f'hist_pvprod_diff_hist_{scen_group}.html'
            fig.write_html(f'{data_path}/visualizations_individual_plots/{file_name}.html')


        # plot JOINT histograms ----------------------
        fig_joint_list = []
        meth_suffix = meth_suffixes[0]
        
        additonal_hist_col = plot_additional_histograms_list[0]
        for additonal_hist_col in plot_additional_histograms_list:
            fig_abs = px.scatter(
                merged_scen_df, 
                x=f'diff_pvprod_kW_abs',
                y=f'{additonal_hist_col}_{meth_suffix}',
                marginal_x="histogram",
                marginal_y="histogram",
                color_discrete_sequence=['blue'],   
                title=f'Diff ABSOLUTE in pvprod vs {additonal_hist_col}',
            )
            fig_rel = px.scatter(
                merged_scen_df, 
                x=f'diff_pvprod_kW_rel',
                y=f'{additonal_hist_col}_{meth_suffix}',
                marginal_x="histogram",
                marginal_y="histogram", 
                color_discrete_sequence=['red'],   
                title=f'Diff RELATIVE ({meth_suffixes[0]}/{meth_suffixes[1]}) in pvprod vs {additonal_hist_col}',
            )
            
            # add log transformations
            merged_scen_df[f'diff_pvprod_kW_abs_log'] = np.log(merged_scen_df[f'diff_pvprod_kW_abs'])
            merged_scen_df[f'diff_pvprod_kW_rel_log'] = np.log(merged_scen_df[f'diff_pvprod_kW_rel'])
            merged_scen_df[f'{additonal_hist_col}_{meth_suffix}_log'] = np.log(merged_scen_df[f'{additonal_hist_col}_{meth_suffix}'])

            fig_loglog_abs = px.scatter(
                merged_scen_df, 
                x=f'diff_pvprod_kW_abs_log',
                y=f'{additonal_hist_col}_{meth_suffix}_log', 
                marginal_x="histogram",
                marginal_y="histogram",
                color_discrete_sequence=['green'],   
                title=f'LOG Diff Absolute in pvprod vs LOG {additonal_hist_col}',
                )
            fig_loglog_rel = px.scatter(
                merged_scen_df, 
                x=f'diff_pvprod_kW_rel_log',
                y=f'{additonal_hist_col}_{meth_suffix}_log', 
                marginal_x="histogram",
                marginal_y="histogram",
                color_discrete_sequence=['purple'],   
                title=f'LOG Diff Relative in pvprod vs LOG {additonal_hist_col}',
            )
            
            fig_abs.show()
            fig_rel.show()
            fig_loglog_abs.show()
            fig_loglog_rel.show()

            # export
            if not os.path.exists(f'{data_path}/visualizations_individual_plots'):
                os.makedirs(f'{data_path}/visualizations_individual_plots')
            
            ind_visual_path = f'{data_path}/visualizations_individual_plots'
            fig_abs.write_html(f'{ind_visual_path}/joint_hist_pvprod_abs_{additonal_hist_col}_{scen_group}.html')
            fig_rel.write_html(f'{ind_visual_path}/joint_hist_pvprod_rel_{additonal_hist_col}_{scen_group}.html')
            fig_loglog_abs.write_html(f'{ind_visual_path}/joint_hist_pvprod_loglog_{additonal_hist_col}_{scen_group}.html')
            fig_loglog_rel.write_html(f'{ind_visual_path}/joint_hist_pvprod_loglog_rel_{additonal_hist_col}_{scen_group}.html')