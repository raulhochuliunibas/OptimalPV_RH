import sys
import os
import pandas as pd
import numpy as np

from collections import OrderedDict

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# INVESTMENT COSTS
# ------------------------------------------------------------------------------------------------------

# COST DATA FRAME --------------------------------------------
def attach_pv_cost(
        script_run_on_server_def = None,
        recreate_parquet_files_def = None,
        smaller_import_def = None,
        log_file_name_def = None,
        wd_path_def = None,
        data_path_def = None,
        show_debug_prints_def = None,
        ):        
    """
    Function to create assumed cost df
    Source: https://www.energieschweiz.ch/tools/solarrechner/
    """ 

    conversion_m2_to_kw = 0.1  # A 1m2 area can fit 0.1 kWp of PV Panels

    # setup -------------------
    wd_path = "D:\RaulHochuli_inuse\OptimalPV_RH" if script_run_on_server_def else "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'

    # create directory + log file
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')
    checkpoint_to_logfile('run function: create_cost_df_and_func.py', log_file_name_def=log_file_name_def, n_tabs_def = 5) 
    print_to_logfile(f' > assuming cost of energieschweiz.ch/tools/solarrechner/; January 2024', log_file_name_def= log_file_name_def)


    # create cost df -------------------
    installation_cost_dict = {
        "on_roof_installation_cost_pkW": {
            2:   4636,
            3:   3984,
            5:   3373,
            10:  2735,
            15:  2420,
            20:  2219,
            30:  1967,
            50:  1710,
            75:  1552,
            100: 1463,
            125: 1406,
            150: 1365
        },
        "on_roof_installation_cost_total": {
            2:   9272,
            3:   11952,
            5:   16863,
            10:  27353,
            15:  36304,
            20:  44370,
            30:  59009,
            50:  85478,
            75:  116420,
            100: 146349,
            125: 175748,
            150: 204816
        },}
    
    installation_cost_df = pd.DataFrame({
        'kw': list(installation_cost_dict['on_roof_installation_cost_pkW'].keys()),
        'chf_pkW': list(installation_cost_dict['on_roof_installation_cost_pkW'].values()),
        'chf_total': list(installation_cost_dict['on_roof_installation_cost_total'].values())
    })
    installation_cost_df.reset_index(inplace=True)



    # export cost df -------------------
    installation_cost_df.to_parquet(f'{data_path}/output/preprep_data/pvinstcost_table.parquet')
    checkpoint_to_logfile(f'exported pvinstcost_table', log_file_name_def=log_file_name_def, n_tabs_def = 5)



    # import and transform for COST dfs -------------------
    solkat = pd.read_parquet(f'{data_path}/output/preprep_data/solkat_by_gm.parquet')
    def convert_srs_to_str(df, colname):
        df[colname] = df[colname].fillna(-1).astype(int).astype(str)
        df[colname] = df[colname].replace('-1', np.nan)
        return df

    solkat = convert_srs_to_str(solkat, 'GWR_EGID')
    solkat = convert_srs_to_str(solkat, 'BFS_NUMMER')
    
    solkat['n_partition'] = 1
    checkpoint_to_logfile(f'imported + transformed solkat_by_gm.parquet', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)



    # extend COST for 3+ partition -------------------

    solkatcost_3up = solkat.groupby(['GWR_EGID', ]).agg(
        {'FLAECHE': 'sum', 'MSTRAHLUNG': 'mean', 'GSTRAHLUNG': 'sum', 'STROMERTRAG': 'sum', 'n_partition':'sum'}).reset_index()
    solkatcost_3up['pvpot_bysurface_kw'] = solkatcost_3up['FLAECHE'] * conversion_m2_to_kw
    checkpoint_to_logfile(f'created solkatcost_3up', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)
    
    # use COST intrapolation
    # solkatcost_3up['partition_pv_cost_chf'] = np.interp(solkatcost_3up['pvpot_bysurface_kw'], installation_cost_df['kw'], installation_cost_df['chf_pkW'])
    checkpoint_to_logfile(f'attached intrapolated cost to solkatcost_3up', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def=show_debug_prints_def)

    # export cost_3up
    # solkatcost_3up.to_parquet(f'{data_path}/output/preprep_data/solkatcost_3up.parquet')
    # solkatcost_3up.to_csv(f'{data_path}/output/preprep_data/solkatcost_3up.csv')
    checkpoint_to_logfile(f'exported solkatcost_3up', log_file_name_def=log_file_name_def, n_tabs_def = 2, show_debug_prints_def= show_debug_prints_def)



    # extend COST per ADDITIONAL partition -------------------
    
    # use groupby and .cumsum()
    # prepare df to calculate cumulative cost
    solkatcost_cumm = solkat.loc[solkat['GWR_EGID'].notna(), ['GWR_EGID', 'DF_UID', 'DF_NUMMER', 'SB_UUID', 'KLASSE', 
                                                        'FLAECHE', 'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG']].copy()

    solkatcost_cumm = solkatcost_cumm.sort_values(by=['GWR_EGID', 'GSTRAHLUNG', 'KLASSE'], ascending=[True, False, False])
    solkatcost_cumm['counter_partition'] = solkatcost_cumm.groupby('GWR_EGID').cumcount() + 1

    # apply cummulative sum in groupby on copy of variable of interest
    cumulative_cols = ['FLAECH_cumm', 'GSTRAH_cumm', 'STROME_cumm']
    solkatcost_cumm[cumulative_cols] = solkatcost_cumm[['FLAECHE', 'GSTRAHLUNG', 'STROMERTRAG']]
    for col in cumulative_cols:
        checkpoint_to_logfile(f'start cumulative sum for: {col}', log_file_name_def=log_file_name_def, n_tabs_def=2, show_debug_prints_def=show_debug_prints_def)
        solkatcost_cumm[col] = solkatcost_cumm.groupby('GWR_EGID')[col].transform(pd.Series.cumsum)

    # convert m2 to kw and intrapolate costs
    solkatcost_cumm['pvpot_bysurface_kw_cumm'] = solkatcost_cumm['FLAECH_cumm'] * conversion_m2_to_kw
    solkatcost_cumm['partition_pv_cost_chf'] = np.interp(solkatcost_cumm['pvpot_bysurface_kw_cumm'], installation_cost_df['kw'], installation_cost_df['chf_pkW'])
    checkpoint_to_logfile(f'attached intrapolated cost to solkatcost_cumm', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def=show_debug_prints_def)

    # export cost_cumm
    solkatcost_cumm.to_parquet(f'{data_path}/output/preprep_data/solkatcost_cumm.parquet')
    solkatcost_cumm.to_csv(f'{data_path}/output/preprep_data/solkatcost_cumm.csv')
    checkpoint_to_logfile(f'exported solkatcost_3up + solkatcost_cumm', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)

    # use numpy.cumsum.group => doesn't work!
    if False: 
        # prepare df to calculate cumulative cost
        solkatcost_cumm = solkat.loc[solkat['GWR_EGID'].notna(), ['GWR_EGID', 'DF_UID', 'DF_NUMMER', 'SB_UUID', 'KLASSE', 
                                                                'FLAECHE', 'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG']].copy()
        solkatcost_cumm = solkatcost_cumm.sort_values(by=['GWR_EGID', 'GSTRAHLUNG', 'KLASSE'], ascending=[True, False, False])
        solkatcost_cumm['counter_partition'] = solkatcost_cumm.groupby('GWR_EGID').cumcount() + 1
        solkatcost_cumm[['FLAECH_cumm', 'GSTRAH_cumm', 'STROME_cumm']]= np.nan

        #  get GWR_EGID counts

        # a faster version of this >> egid_n = [(x, list(solkatcost_cumm['GWR_EGID']).count(x)) for x in solkatcost_cumm['GWR_EGID'].unique()]
        checkpoint_to_logfile(f'start GWR_egid counts: GPT version', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)
        egid_list = solkatcost_cumm['GWR_EGID']

        egid_counts_dict = OrderedDict() # Initialize an ordered dictionary to preserve the order
        for egid in solkatcost_cumm['GWR_EGID']:    # Iterate through the 'GWR_EGID' column and count occurrences
            egid_counts_dict[egid] = egid_counts_dict.get(egid, 0) + 1 
            
        egid_counts_list = list(egid_counts_dict.items())  # Convert the ordered dictionary to a list of tuples (value, count)
        egid_counts = list(zip(*egid_counts_list))[1]

        checkpoint_to_logfile(f'end GWR_egid counts: GPT version', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)

        # checkpoint_to_logfile(f'start GWR_egid counts: lambdalist version', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)
        # egid_n = [(x, list(solkatcost_cumm['GWR_EGID']).count(x)) for x in solkatcost_cumm['GWR_EGID'].unique()]
        # checkpoint_to_logfile(f'end GWR_egid counts: lambdalist version', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)










####################
# V V V to be deleted in March 2024
# extend COST per ADDITONAL partition IN LOOPS -------------------

"""
# TAKING MULTIPLE DAYS TO SOLVE!
# prepare df to calculate cumulative cost
solkatcost_cumm = solkat.loc[solkat['GWR_EGID'].notna(), ['GWR_EGID', 'DF_UID', 'DF_NUMMER', 'SB_UUID', 'KLASSE', 
                                                        'FLAECHE', 'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG',]].copy()

solkatcost_cumm = solkatcost_cumm.sort_values(by=['GWR_EGID', 'GSTRAHLUNG', 'KLASSE'], ascending=[True, False, False])

solkatcost_cumm['counter_partition'] = solkatcost_cumm.groupby('GWR_EGID').cumcount() + 1 # add counter variable and shift column to front
cols = solkatcost_cumm.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
solkatcost_cumm = solkatcost_cumm[cols]

# do 2 loop to assign cummulative values
solkatcost_cumm['FLAECH_cumm'] = np.nan
solkatcost_cumm['GSTRAH_cumm'] = np.nan
solkatcost_cumm['MSTRAH_cumm'] = np.nan
solkatcost_cumm['STROME_cumm'] = np.nan

egid_list = solkatcost_cumm['GWR_EGID'].unique().tolist()
egid_list = egid_list[0:3] if smaller_import_def else egid_list
checkpoint_to_logfile(f'start loop for accumulation over solkatcost', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)

counter_10perc_loop = len(egid_list) // 2000000
for n, e in enumerate(egid_list,start = 1):
    df_sub = solkatcost_cumm.loc[solkatcost_cumm['GWR_EGID']== e]
    df_sub = df_sub.sort_values(by = ['GSTRAHLUNG', 'KLASSE'], ascending = [False, False])
    counter_list = df_sub['counter_partition'].to_list()

    for c in counter_list:
        print(f'GWR_EGID: {e} partition: {c}') if smaller_import_def else None

        solkatcost_cumm.loc[(solkatcost_cumm['GWR_EGID'] == e) & (solkatcost_cumm['counter_partition'] == c), 'FLAECH_cumm'] = df_sub.loc[df_sub['counter_partition'] <= c, 'FLAECHE'].sum()
        solkatcost_cumm.loc[(solkatcost_cumm['GWR_EGID'] == e) & (solkatcost_cumm['counter_partition'] == c), 'GSTRAH_cumm'] = df_sub.loc[df_sub['counter_partition'] <= c, 'GSTRAHLUNG'].sum()
        solkatcost_cumm.loc[(solkatcost_cumm['GWR_EGID'] == e) & (solkatcost_cumm['counter_partition'] == c), 'MSTRAH_cumm'] = df_sub.loc[df_sub['counter_partition'] <= c, 'MSTRAHLUNG'].sum()
        solkatcost_cumm.loc[(solkatcost_cumm['GWR_EGID'] == e) & (solkatcost_cumm['counter_partition'] == c), 'STROME_cumm'] = df_sub.loc[df_sub['counter_partition'] <= c, 'STROMERTRAG'].sum()
    
    if n % counter_10perc_loop==0:
        checkpoint_to_logfile(f'cumulated {n} of {len(egid_list)} GWR_EGIDs', log_file_name_def=log_file_name_def, n_tabs_def = 2, show_debug_prints_def= show_debug_prints_def)

checkpoint_to_logfile(f'end loop for accumulation over solkatcost', log_file_name_def=log_file_name_def, n_tabs_def = 5)
solkatcost_cumm['pvpot_bysurface_kw_cumm'] = solkatcost_cumm['FLAECH_cumm'] * conversion_m2_to_kw
"""

####################


