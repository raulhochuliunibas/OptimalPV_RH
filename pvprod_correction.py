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
import glob

from datetime import datetime
from shapely.geometry import Point
from shapely.ops import unary_union
from plotly.subplots import make_subplots


# ** SOURCES: **
# https://www.home-energy.ch/de/preise-berechnen/solarrechner-einfamilienhaus; settings for max consumption

# FUNCTION INPUT ==========================
pvalloc_scenarios = [    
    'pvalloc_BLsml_24m_meth2.2_random',
    # 'pvalloc_BLsml_24m_meth2.2_npvweight',
    # 'pvalloc_BLsml_24m_meth3.2_random',

]
excl_buildings_of_mean_calc = ['410320', '391290', '391291',  # Byfangweg,  Lerchenstrasse 31 + 33
                               '291263', 
]
# estim_cost_chf_pkWp_correctionfactor = 1.243
estim_cost_chf_pkWp_correctionfactor = 1



# --------------------------------------------------------------------------------------------------------------------------------------------

# pvalloc_settings = pvalloc_scenarios[list(pvalloc_scenarios.keys())[0]]
fig_agg = go.Figure()
for scen in pvalloc_scenarios:
    # =========================================
    # glob.glob(f'{os.getcwd()}_data/output/{scen}/pvalloc_settings.json')
    wd_path = os.getcwd()
    pvalloc_settings = json.load(open(f'{wd_path}_data/output/{scen}/pvalloc_settings.json', 'r'))


    # setup --------------------
    data_path = f'{wd_path}_data'
    # mc_path = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/zMC_1'

    # import --------------------
    solkat = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat.parquet')
    pv = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv.parquet')

    topo = json.load(open(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/topo_egid.json', 'r'))
    npv_df = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/sanity_check_byEGID/pred_npv_inst_by_M/npv_df_2023-01.parquet')

    # solkat

    # manually select data --------------------
    col_names = ['EGID',    'Adress',       'DF_UID_solkat_selection',  'flaech_bkw',   'instcapa_kWp_bkw',     'pvprod_kWh_pyear_bkw', 'estim_demand_pyear_kWh_bkw', 'estim_investcost_inclsubsidy_chf_bkw', 
                                                                        'flaech_ewz',   'instcapa_kWp_ewz',     'pvprod_kWh_pyear_ewz', 'estim_demand_pyear_kWh_ewz', 'estim_investcost_inclsubsidy_chf_ewz']
    rows = [
    # ['EGID',    'Adress',                 'DF_UID_solkat_selection',  'flaech_bkw',   'instcapa_kWp_bkw', 'pvprod_kWh_pyear_bkw', 'estim_demand_pyear_kWh_bkw', 'estimated_investcost_inclsubsidy_chf_bkw'],
    #                                                                   'flaech_ewz',   'instcapa_kWp_ewz', 'pvprod_kWh_pyear_ewz', 'estim_demand_pyear_kWh_ewz', 'estimated_investcost_inclsubsidy_chf_ewz'],
    ['391292', 'Lerchenstrasse 35, Aesch', ['10213764',],               174,            18.66,              19824,                  18930,                         28354, 
                                                                        174,            31.5,               33537,                  18930,                         48010],
    ['391291', 'Lerchenstrasse 33, Aesch',  ['10213735','10213736', 
                                            '10213753', '10213754'],    112,            18.66,              18084,                  18930,                         28354,
                                                                        112,            20.25,              21382,                  18930,                         33401],
    ['391290', 'Lerchenstrasse 31, Aesch',  ['10213733','10213734'],    82,             14.56,              14147,                  18930,                         26141,
                                                                        82,             14.85,              15696,                  18930,                         25879],
    
    
    ['410320', 'Byfangweg 3, Pfeffingen',   ['10208685',],              95,             16.84,              16429,                  18930,                         28042,
                                                                        95,             17.1,               16744,                  18930,                         29464],
    ['410187', 'Alemannenweg 8, Pfeffingen',['10206773',],              100,            17.29,              14662,                  18930,                         28399,
                                                                        100,            18,                 15274,                  18930,                         30465],
    ['410227', 'Moosackerweg 9, Pfeffingen',['10206727',],              113,            18.66,              16824,                  18930,                         28354,
                                                                        113,            20.25,              18268,                  18930,                         33401],
    ['245060521', 'Drosselweg 12, Aesch',   ['10213776','10213777'],    119,            18.66,              18020,                  18930,                         28354,
                                                                        119,            19.8,               19125,                  18930,                         32469],
    ['245054175', 'Drosselweg 10, Aesch',   ['10213805','10213806'],    148,            18.66,              15438,                  18930,                         28354 ,
                                                                        148,            26.55,              30183,                  18930,                         39840],
    ['391392',  'Klusstrasse 27a, Aesch',   ['10212856', '10212857'],   108,            18.2,               14970,                  18930,                         27814,
                                                                        108,            19.35,              21712,                  18930,                         31968],
    ['391393',  'Klusstrasse 27b, Aesch',   ['10212854', '10212855'],   109,            18.66,              15746,                  18930,                         28354,
                                                                        109,            19.8,               22181,                  18930,                         32469],
    ['3032639', 'Klusstrasse 29, Aesch',    ['10212957'],               63,             10.01,              9667,                   18930,                         22311,
                                                                        63,             10.35,              9996,                   18930,                         23357],
    ['391404', 'Trottmattweg 2, Aesch',     ['10212880'],               93,             15.47,              14917,                  18930,                         26847,
                                                                        93,             15.75,              15186,                  18930,                         27961],   
    ['391289' , 'Lerchenstrasse 29, Aesch', ['10213770', '10213771',
                                             '10213772', '10213773',
                                             '10213774', '10213775',],  164,            18.66,              17757,                  18930,                         28354,        
                                                                        164,            29.7,               32576,                  18930,                         42921],
    ['391186', 'Amselweg 3a, Aesch',        ['10213751', '10213752'],   58,             10.01,              9885,                   18930,                         22311,
                                                                        58,             10.35,              10180,                  18930,                         23357],
    ['391187', 'Amselweg 3b, Aesch',        ['10213751', '10213752'],   58,             10.01,              9885,                   18930,                         22311,
                                                                        58,             10.35,              10180,                  18930,                         23357],
    ['391263', 'Drosselweg 24, Aesch',      ['10213721', '10213722', 
                                             '10213814', '10213815',],  201,            18.66,              18094,                  18930,                         28354,
                                                                        201,            36.45,              39392,                  18930,                         55793],
    ['391262', 'Drosselweg 22, Aesch',      ['10213778', '10213779', 
                                             '10213818', '10213819',],  175,            18.66,              17688,                  18930,                         28354,
                                                                        175,            31.5,               31265,                  18930,                         48010],
    ['391379', 'Drosselweg 72, Aesch',      ['10212718', '10212719'],   228,            18.66,              18780,                  18930,                         28354,
                                                                        228,            41.4,               40392,                  18930,                         61412], 
    ['391977', 'Klusstrasse 66, Aesch',     ['10212671', '10212672'],   167,            18.66,              16747,                  18930,                         28354, 
                                                                        167,            30.15,              28775,                  18930,                         47201],
    ]
    comparison_df = pd.DataFrame(rows, columns=col_names)
    comparison_df['DF_UID_solkat_selection'] = comparison_df['DF_UID_solkat_selection'].apply(lambda x: sorted(x))


    # attach solkat data to comparison_df --------------------
    i, row = 0, comparison_df.loc[0]
    for i, row in comparison_df.iterrows():
        comparison_df.loc[i, 'n_roofs'] = len(row['DF_UID_solkat_selection'])
        comparison_df.loc[i, 'FLAECHE'] = solkat.loc[(solkat['EGID'] == row['EGID']) & 
                                                     (solkat['DF_UID'].isin(row['DF_UID_solkat_selection'])), 'FLAECHE'].sum()
        comparison_df.loc[i, 'instcap_kWp'] = comparison_df.loc[i, 'FLAECHE'] * pvalloc_settings['tech_economic_specs']['kWpeak_per_m2']
        comparison_df.loc[i, 'STROMERTRAG'] = solkat.loc[(solkat['EGID'] == row['EGID']) & 
                                                        (solkat['DF_UID'].isin(row['DF_UID_solkat_selection'])), 'STROMERTRAG'].sum()
        

        # row_npv = npv_df.loc[npv_df['EGID'] == row['EGID']].iloc[2]
        npv_df['df_uid_combo_list']  = npv_df['df_uid_combo'].apply(lambda x: x.split('_') if '_' in x else [x])
        # npv_df['df_uid_combo_list'] = npv_df['df_uid_combo_list'].apply(lambda x: sorted(x)) 


        # Find matching rows in npv_df
        matching_rows = npv_df.loc[(npv_df['EGID'] == row['EGID']) &
                                   (npv_df['df_uid_combo_list'].apply(lambda x: set(x) == set(row['DF_UID_solkat_selection'])))]
        
        # matching_rows = npv_df[npv_df['df_uid_combo_list'].apply(lambda x: set(x) == set(row['DF_UID_solkat_selection']))]
        # Sum the values for the matching rows
        comparison_df.loc[i, 'pvprod_kW'] = matching_rows['pvprod_kW'].sum()
        comparison_df.loc[i, 'estim_pvinstcost_chf'] = matching_rows['estim_pvinstcost_chf'].sum()

        # cost transformations to see deviations
        comparison_df.loc[i, 'estim_pvinstcost_chf'] = comparison_df.loc[i, 'estim_pvinstcost_chf'] /estim_cost_chf_pkWp_correctionfactor

        # calc cost/kWp and cost/kWh
        comparison_df.loc[i, 'estim_cost_chf_pkWp'] =     comparison_df.loc[i, 'estim_pvinstcost_chf'] / comparison_df.loc[i, 'instcap_kWp']
        comparison_df.loc[i, 'estim_cost_chf_pkWp_bkw'] = comparison_df.loc[i, 'estim_investcost_inclsubsidy_chf_bkw'] / comparison_df.loc[i, 'instcapa_kWp_bkw']
        comparison_df.loc[i, 'estim_cost_chf_pkWp_ewz'] = comparison_df.loc[i, 'estim_investcost_inclsubsidy_chf_ewz'] / comparison_df.loc[i, 'instcapa_kWp_ewz']

        comparison_df.loc[i, 'estim_cost_chf_pkWh'] =     comparison_df.loc[i, 'estim_pvinstcost_chf'] / comparison_df.loc[i, 'pvprod_kW']
        comparison_df.loc[i, 'estim_cost_chf_pkWh_bkw'] = comparison_df.loc[i, 'estim_investcost_inclsubsidy_chf_bkw'] / comparison_df.loc[i, 'pvprod_kWh_pyear_bkw']
        comparison_df.loc[i, 'estim_cost_chf_pkWh_ewz'] = comparison_df.loc[i, 'estim_investcost_inclsubsidy_chf_ewz'] / comparison_df.loc[i, 'pvprod_kWh_pyear_ewz']

        comparison_df.loc[i, 'delta_cost_pkWp_mod_bkw'] = (comparison_df.loc[i, 'estim_cost_chf_pkWp' ] - comparison_df.loc[i, 'estim_cost_chf_pkWp_bkw']) / comparison_df.loc[i, 'estim_cost_chf_pkWp_bkw']
        comparison_df.loc[i, 'delta_cost_pkWp_mod_ewz'] = (comparison_df.loc[i, 'estim_cost_chf_pkWp' ] - comparison_df.loc[i, 'estim_cost_chf_pkWp_ewz']) / comparison_df.loc[i, 'estim_cost_chf_pkWp_ewz']



    
    # plot --------------------
    fig = go.Figure()
    plot_cols = [
        'FLAECHE',     
        # 'flaech_bkw',           'flaech_ewz',
        'instcap_kWp',  'instcapa_kWp_bkw',     #'instcapa_kWp_ewz',
        # 'pvprod_kW',    'pvprod_kWh_pyear_bkw', 'pvprod_kWh_pyear_ewz',
        # 'STROMERTRAG',
        'estim_pvinstcost_chf', 'estim_investcost_inclsubsidy_chf_bkw', 'estim_investcost_inclsubsidy_chf_ewz',
        'estim_cost_chf_pkWp', 'estim_cost_chf_pkWp_bkw', 'estim_cost_chf_pkWp_ewz', 
        # 'estim_cost_chf_pkWh', 'estim_cost_chf_pkWh_bkw', 'estim_cost_chf_pkWh_ewz',   
        'delta_cost_pkWp_mod_bkw',     
        'delta_cost_pkWp_mod_ewz',
                ]
    cols_in_second_axis_tuples = [
        ('pvprod_kW', 1000), ('pvprod_kWh_pyear_bkw', 1000), ('pvprod_kWh_pyear_ewz', 1000),
        ('STROMERTRAG', 1000),  #('FLAECHE', 10),
        ('estim_pvinstcost_chf', 1000), ('estim_investcost_inclsubsidy_chf_bkw', 1000), ('estim_investcost_inclsubsidy_chf_ewz', 1000),
        ('estim_cost_chf_pkWp', 100), ('estim_cost_chf_pkWp_bkw', 100), ('estim_cost_chf_pkWp_ewz', 100),
        ('estim_cost_chf_pkWh', 0.1), ('estim_cost_chf_pkWh_bkw', 0.1), ('estim_cost_chf_pkWh_ewz', 0.1),
        ]
    comparison_df['x_label'] = comparison_df['Adress'] + ' (' + comparison_df['EGID'] + ')'


    for col in plot_cols:
        cols_in_second_axis = [col_tuple[0] for col_tuple in cols_in_second_axis_tuples]
        if col in cols_in_second_axis:
            col_secondaxis_tuple = [col_tuple for col_tuple in cols_in_second_axis_tuples if col_tuple[0] == col][0]
            col_secondaxis_name = col_secondaxis_tuple[0]
            col_secondaxis_denom = col_secondaxis_tuple[1]

            comparison_df[col] = comparison_df[col] / col_secondaxis_denom
        
        name_col = col + f' (1/{col_secondaxis_denom})' if col in cols_in_second_axis else col
        fig.add_trace(go.Bar(
            x=comparison_df['x_label'],
            y = comparison_df[col],
            name=name_col,
            text=comparison_df[col],
        ))
        fig_agg.add_trace(go.Bar(
            x=comparison_df['x_label'],
            y = comparison_df[col],
            name=name_col,
            text=comparison_df[col],
        ))
    # title trace for aggregation plot
    fig_agg.add_trace(go.Scatter(x=[0,],y=[0,],
        name=scen,opacity=0,))
    fig_agg.add_trace(go.Bar(
        x=[0,],y=[0,],
        name='---', opacity=0,))
    fig_agg.add_trace(go.Scatter(x=[0,],y=[0,],
        name='',opacity=0,))

    fig.update_layout(
        barmode='group',  # Automatically groups bars without overlap
        title='Comparions OptPV-Model to BKW / EWZ Solarrechner',
        yaxis=dict(title='Primary Axis'),  # Configure primary y-axis
        yaxis2=dict(title='Secondary Axis', overlaying='y', side='right'),  # Configure secondary y-axis
    )
    # fig.show()
    
    if not os.path.exists(f'{data_path}/output/visualizations_pvprod_correction'):
        os.makedirs(f'{data_path}/output/visualizations_pvprod_correction')

    fig.write_html(f'{data_path}/output/visualizations_pvprod_correction/pvprod_correction_{scen}.html')

fig_agg.update_layout(
    barmode='group',  # Automatically groups bars without overlap
    title='AGGREGATION: Comparions OptPV-Model to BKW / EWZ Solarrechner',
    yaxis=dict(title='Primary Axis'),  # Configure primary y-axis
    yaxis2=dict(title='Secondary Axis', overlaying='y', side='right'),  # Configure secondary y-axis
)
fig_agg.show()
fig_agg.write_html(f'{data_path}/output/visualizations_pvprod_correction/pvprod_correction_agg.html')

print_mean_deviation_tobkw = comparison_df.loc[~comparison_df['EGID'].isin(excl_buildings_of_mean_calc), 'delta_cost_pkWp_mod_bkw'].mean()
print_mean_deviation_toewz = comparison_df.loc[~comparison_df['EGID'].isin(excl_buildings_of_mean_calc), 'delta_cost_pkWp_mod_ewz'].mean()
print(f'Mean deviation of cost/kWp between OptPV-Model and BKW-Solarrechner: {print_mean_deviation_tobkw:.5%}')
print(f'Mean deviation of cost/kWp between OptPV-Model and EWZ-Solarrechner: {print_mean_deviation_toewz:.5%}')

comparison_df.to_excel(f'{data_path}/output/visualizations_pvprod_correction/pvprod_correction_comparison.xlsx')
print('..just for breakpoint..')

