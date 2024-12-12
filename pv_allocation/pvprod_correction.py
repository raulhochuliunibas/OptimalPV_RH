import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import winsound
import json
import plotly.graph_objects as go
import plotly.express as px
import copy

from datetime import datetime
from shapely.geometry import Point
from shapely.ops import unary_union
from plotly.subplots import make_subplots


# ** SOURCES: **
# https://www.home-energy.ch/de/preise-berechnen/solarrechner-einfamilienhaus; settings for max consumption

# FUNCTION INPUT ==========================
pvalloc_scenarios = {
    'pvallco_BL_small_12m_meth2.2_rad_dfuid_ind ':{
        'MC_loop_specs': {'fresh_initial_files': 'topo_egid.json, '
                                          'months_prediction.parquet, '
                                          'gridprem_ts.parquet, '
                                          'constrcapa.parquet, '
                                          'dsonodes_df.parquet',
                   'keep_files_month_iter_TF': True,
                   'keep_files_month_iter_list': 'topo_egid.json, '
                                                 'npv_df.parquet, '
                                                 'pred_inst_df.parquet, '
                                                 'gridprem_ts.parquet',
                   'keep_files_month_iter_max': 8,
                   'montecarlo_iterations': 1},
 'T0_prediction': '2023-01-01 00:00:00',
 'algorithm_specs': {'constr_capa_overshoot_fact': 1,
                     'inst_selection_method': 'random',
                     'npv_update_agg_cols_topo_aggdf': {'demand_kW': 'sum',
                                                        'econ_inc_chf': 'sum',
                                                        'econ_spend_chf': 'sum',
                                                        'netdemand_kW': 'sum',
                                                        'netfeedin_kW': 'sum',
                                                        'pvprod_kW': 'sum',
                                                        'selfconsum_kW': 'sum'},
                     'npv_update_grouby_cols_topo_aggdf': 'EGID, df_uid, '
                                                          'grid_node, bfs, '
                                                          'gklas, demandtype, '
                                                          'inst_TF, '
                                                          'info_source, pvid, '
                                                          'pv_tarif_Rp_kWh, '
                                                          'elecpri_Rp_kWh, '
                                                          'FLAECHE, '
                                                          'FLAECH_angletilt, '
                                                          'AUSRICHTUNG, '
                                                          'NEIGUNG, '
                                                          'STROMERTRAG',
                     'rand_seed': 42,
                     'topo_subdf_partitioner': 400,
                     'tweak_constr_capacity_fact': 1,
                     'tweak_gridnode_df_prod_demand_fact': 1,
                     'tweak_npv_calc': 1,
                     'tweak_npv_excl_elec_demand': True,
                     'while_inst_counter_max': 5000},
 'bfs_numbers': '2768, 2761, 2772, 2473, 2475, 2785, 2480',
 'constr_capacity_specs': {'ann_capacity_growth': 0.05,
                           'constr_capa_overshoot_fact': 1,
                           'share_to_summer': 0.6,
                           'share_to_winter': 0.4,
                           'summer_months': '4, 5, 6, 7, 8, 9',
                           'winter_months': '10, 11, 12, 1, 2, 3'},
 'create_gdf_export_of_topology': True,
 'data_path': 'C:/Models/OptimalPV_RH_data',
 'export_csvs': False,
 'fast_debug_run': False,
 'gridprem_adjustment_specs': {'colnames': 'tier_level, used_node_capa_rate, '
                                           'gridprem_Rp_kWh',
                               'perf_factor_1kVA_to_XkW': 0.8,
                               'power_factor': 1,
                               'tier_description': 'tier_level: '
                                                   '(voltage_threshold, '
                                                   'gridprem_Rp_kWh)',
                               'tiers': {1: '0.7, 1',
                                         2: '0.8, 3',
                                         4: '0.9, 7',
                                         5: '0.95, 15',
                                         6: '10, 100'}},
 'gwr_selection_specs': {'DEMAND_proxy': 'GAREA',
                         'GBAUJ_minmax': '1950, 2022',
                         'GKLAS': '1110, 1121',
                         'GSTAT': '1004',
                         'building_cols': 'EGID, GDEKT, GGDENR, GKODE, GKODN, '
                                          'GKSCE, GSTAT, GKAT, GKLAS, GBAUJ, '
                                          'GBAUM, GBAUP, GABBJ, GANZWHG, '
                                          'GWAERZH1, GENH1, GWAERSCEH1, '
                                          'GWAERDATH1, GEBF, GAREA',
                         'dwelling_cols': None,
                         'solkat_max_n_partitions': 10},
 'interim_path': 'C:/Models/OptimalPV_RH_data/output/pvalloc_run',
 'kt_numbers': '',
 'log_file_name': 'C:/Models/OptimalPV_RH_data/output/pvalloc_init_log.txt',
 'months_lookback': 12,
 'months_prediction': 12,
 'n_egid_in_topo': 200,
 'name_dir_export': 'pvallco_BL_small_12m_meth2.2_rad_dfuid_ind',
 'name_dir_import': 'preprep_BL_22to23_1and2homes',
 'panel_efficiency_specs': {'hot_hours_discount': 0.1,
                            'hotsummer_hours': '11, 12, 13, 14, 15, 16, 17',
                            'summer_months': '6, 7, 8, 9',
                            'variable_panel_efficiency_TF': True},
 'pvalloc_path': 'C:/Models/OptimalPV_RH_data/output/pvalloc_run',
 'recalc_economics_topo_df': True,
 'recreate_topology': True,
 'sanitycheck_byEGID': True,
 'sanitycheck_summary_byEGID_specs': {'egid_list': '391292, 390601, 2347595, '
                                                   '401781391263, 245057295, '
                                                   '401753',
                                      'n_EGIDs_of_alloc_algorithm': 20,
                                      'n_iterations_before_sanitycheck': 12},
 'script_run_on_server': False,
 'show_debug_prints': True,
 'summary_file_name': 'C:/Models/OptimalPV_RH_data/output/summary_data_selection_log.txt',
 'tech_economic_specs': {'elecpri_category': 'H4',
                         'elecpri_year': 2022,
                         'interest_rate': 0.01,
                         'inverter_efficiency': 0.8,
                         'invst_maturity': 25,
                         'kWpeak_per_m2': 0.2,
                         'max_distance_m_for_EGID_node_matching': 0,
                         'panel_efficiency': 0.15,
                         'pvprod_calc_method': 'method2.2',
                         'pvtarif_col': 'energy1, eco1',
                         'pvtarif_year': 2022,
                         'self_consumption_ifapplicable': 0,
                         'share_roof_area_available': 0.7},
 'wd_path': 'C:/Models/OptimalPV_RH',
 'wd_path_laptop': 'C:/Models/OptimalPV_RH',
 'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',
 'weather_specs': {'flat_diffuse_rad_factor': 1,
                   'flat_direct_rad_factor': 1,
                   'meteo_col_diff_radiation': 'Basel Diffuse Shortwave '
                                               'Radiation',
                   'meteo_col_dir_radiation': 'Basel Direct Shortwave '
                                              'Radiation',
                   'meteo_col_temperature': 'Basel Temperature [2 m elevation '
                                            'corrected]',
                   'rad_rel_loc_max_by': 'dfuid_specific',
                   'radiation_to_pvprod_method': 'dfuid_ind',
                   'weather_year': 2022}}
                   
}

pvalloc_settings = pvalloc_scenarios[list(pvalloc_scenarios.keys())[0]]
# =========================================

# setup --------------------
wd_path = pvalloc_settings['wd_path_laptop']
data_path = f'{wd_path}_data'
# mc_path = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/zMC_1'

# import --------------------
solkat = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat.parquet')
pv = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv.parquet')

topo = json.load(open(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/topo_egid.json', 'r'))
npv_df = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/sanity_check_byEGID/pred_npv_inst_by_M/npv_df_2023-01.parquet')

# solkat

# manually select data --------------------
col_names = ['EGID',    'Adress',       'DF_UID_solkat_selection',  'flaech_bkw',   'instcapa_kWp_bkw',     'pvprod_kWh_pyear_bkw', 'estim_demand_pyear_kWh_bkw', 'estim_investcost_chf_bkw']
rows = [
# ['EGID',    'Adress',                 'DF_UID_solkat_selection',  'flaech_bkw',   'instcapa_kWp_bkw', 'pvprod_kWh_pyear_bkw', 'estim_demand_pyear_kWh_bkw', 'estim_investcost_chf_bkw']
# ['391292', 'Lerchenstrasse. 35, Aesch', ['10213764',],              174,            18.66,              19824,                  18930,                         28354], 
['410320', 'Byfangweg 3, Pfeffingen',   ['10208685',],              95,             16.84,              16429,                  18930,                         28042],             
['410187', 'Alemannenweg 8, Pfeffingen',['10206773',],              100,            17.29,              14662,                  18930,                         28399],
['410227', 'Moosackerweg 9, Pfeffingen',['10206727',],              113,            18.66,              16824,                  18930,                         28354],
['391291', 'Lerchenstrasse 33, Aesch',  ['10213735','10213736', 
                                         '10213753', '10213754'],   112,            18.66,              18084,                  18930,                         28354],
# ['391290', 'Lerchenstrasse 31, Aesch',  ['10213733','10213734'],    82,             14.56,              14147,                  18930,                         26141],
# ['245060521', 'Drosselweg 12, Aesch',   ['10213776','10213777'],    119,            13.65,              13186 ,                 18930,                         25397],
# ['245054175', 'Drosselweg 10, Aesch',   ['10213805','10213806'],    148,            18.2,               15438,                  18930,                         27814],
['391392',  'Klusstrasse 27a, Aesch',   ['10212856', '10212857'],   108,            18.2,               14970,                  18930,                         27814],
['391393',  'Klusstrasse 27b, Aesch',   ['10212854', '10212855'],   109,            18.2,               14970,                  18930,                         27814],
['3032639', 'Klusstrasse 29, Aesch',    ['10212957'],               63,             10.01,              9667,                   18930,                         22311],
['391404', 'Trottmattweg 2, Aesch',     ['10212880'],               93,             15.47,              14917,                  18930,                         26847],   
] 

comparison_df = pd.DataFrame(rows, columns=col_names)


# attach solkat data to comparison_df --------------------
i, row = 7, comparison_df.loc[7]
for i, row in comparison_df.iterrows():
    comparison_df.loc[i, 'n_roofs'] = len(row['DF_UID_solkat_selection'])
    comparison_df.loc[i, 'FLAECHE'] = solkat.loc[solkat['DF_UID'].isin(row['DF_UID_solkat_selection']), 'FLAECHE'].sum()
    comparison_df.loc[i, 'instcap_kWp'] = comparison_df.loc[i, 'FLAECHE'] * pvalloc_settings['tech_economic_specs']['kWpeak_per_m2']
    comparison_df.loc[i, 'STROMERTRAG'] = solkat.loc[solkat['DF_UID'].isin(row['DF_UID_solkat_selection']), 'STROMERTRAG'].sum()

    df_uid_asc, df_uid_desc = sorted(row['DF_UID_solkat_selection']), sorted(row['DF_UID_solkat_selection'], reverse=True)
    df_uid_joined_asc, df_uid_joined_desc = '_'.join(df_uid_asc), '_'.join(df_uid_desc)
    # comparison_df.loc[i, 'pvprod_kW'] =   

    df_uid_combo_insample_TF = (npv_df['df_uid_combo'] == df_uid_joined_asc) | (npv_df['df_uid_combo'] == df_uid_joined_desc)
    comparison_df.loc[i, 'pvprod_kW'] = npv_df.loc[df_uid_combo_insample_TF, 'pvprod_kW'].sum()
    npv_df.loc[npv_df['df_uid_combo'].isin(row['DF_UID_solkat_selection']), 'pvprod_kW'].sum()  
       
    # comparison_df.loc[i, 'AUSRICHTUNG'] = solkat.loc[solkat['DF_UID'].isin(row['DF_UID_solkat_selection']), 'AUSRICHTUNG'].values[0]
    # comparison_df.loc[i, 'NEIGUNG'] = solkat.loc[solkat['DF_UID'].isin(row['DF_UID_solkat_selection']), 'NEIGUNG'].values[0]

 
# plot --------------------
fig = go.Figure()
plot_cols = ['FLAECHE',                 'flaech_bkw', 
             'instcap_kWp',             'instcapa_kWp_bkw', 
             'pvprod_kWh_pyear_bkw',    'STROMERTRAG',          'pvprod_kW',
             ]
cols_in_second_axis = ['pvprod_kWh_pyear_bkw','STROMERTRAG','pvprod_kW',]

comparison_df['x_label'] = comparison_df['Adress'] + ' (' + comparison_df['EGID'] + ')'

for col in plot_cols:
    if col in cols_in_second_axis:
        comparison_df[col] = comparison_df[col] / 1000
    
    name_col = col + ' (1/1000)' if col in cols_in_second_axis else col
    fig.add_trace(go.Bar(
        x=comparison_df['x_label'],
        y = comparison_df[col],
        name=name_col,
        text=comparison_df[col],
    ))

fig.update_layout(
    barmode='group',  # Automatically groups bars without overlap
    yaxis=dict(title='Primary Axis'),  # Configure primary y-axis
    yaxis2=dict(title='Secondary Axis', overlaying='y', side='right'),  # Configure secondary y-axis
)
fig.show()

print('..just for breakpoint..')

