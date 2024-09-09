import sys
import os as os
import numpy as np
import pandas as pd
import json


def select_AND_adjust_topology(pvalloc_settings, 
           npv_df_func, 
           inst_selection_method_func,
           dfuid_installed_list_func, 
           pred_inst_df_func,
           m):
    # select a random building out of npv_df_func to attribute a PV system to

    rand_seed = pvalloc_settings['algorithm_specs']['rand_seed']
    conv_m2toKWP = pvalloc_settings['tech_economic_specs']['conversion_m2tokW']
    data_path_def = pvalloc_settings['data_path']

    npv_df = npv_df_func
    inst_selection_method = inst_selection_method_func
    dfuid_installed_list = dfuid_installed_list_func
    pred_inst_df = pred_inst_df_func


    # import ----------
    topo = json.load(open(f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'r'))
    pred_inst_df = pd.read_parquet(f'{data_path_def}/output/pvalloc_run/pred_inst_df.parquet') if os.path.exists(f'{data_path_def}/output/pvalloc_run/pred_inst_df.parquet') else pd.DataFrame()

    # set random seed
    if rand_seed is not None:
        np.random.seed(rand_seed)

    
    # select a building ----------
    if inst_selection_method == 'random':
        npv_pick = npv_df.sample(n=1).copy()

    elif inst_selection_method == 'prob_weighted_npv':
        rand_num = np.random.uniform(0, 1)

        npv_df['NPV_stand'] = npv_df['NPV_uid'] / max(npv_df['NPV_uid'])
        npv_df['diff_NPV_rand'] = abs(npv_df['NPV_stand'] - rand_num)
        npv_pick = npv_df[npv_df['diff_NPV_rand'] == min(npv_df['diff_NPV_rand'])].copy()

        if npv_pick.shape[0] > 1:
            rand_row = np.random.randint(0, npv_pick.shape[0])
            npv_pick = npv_pick.iloc[rand_row]

        npv_df.drop(columns=['NPV_stand', 'diff_NPV_rand'], inplace=True)

    if isinstance(npv_pick['EGID'], pd.Series):
        picked_egid = npv_pick['EGID'].values[0]
        picked_uid = npv_pick['df_uid_combo'].values[0]
        picked_flaech = npv_pick['FLAECHE'].values[0]

    else:
        picked_egid = npv_pick['EGID']
        picked_uid = npv_pick['df_uid_combo']
        picked_flaech = sum(npv_pick['FLAECHE'])

    # if isinstance(npv_pick['df_uid_combo'], pd.Series):
        # picked_uid = npv_pick['df_uid_combo'].values[0]
    # else:
        # picked_uid = npv_pick['df_uid_combo']

    inst_power = picked_flaech * conv_m2toKWP * pvalloc_settings['algorithm_specs']['capacity_tweak_fact']
    npv_pick['inst_power'] = inst_power
    

    # Adjust export lists / df
    dfuid_installed_list.append(picked_uid)

    if '_' in picked_uid:
        picked_combo_uid = list(picked_uid.split('_'))
    else:
        picked_combo_uid = [picked_uid]

    if isinstance(npv_pick, pd.Series):
        pred_inst_df = pd.concat([pred_inst_df, npv_pick.to_frame().T])
    elif isinstance(npv_pick, pd.DataFrame):
        pred_inst_df = pd.concat([pred_inst_df, npv_pick])

    
    # Adjust topo
    topo[picked_egid]['pv_inst'] = {'inst_TF': True, 'info_source': 'alloc_algorithm', 'xtf_id': picked_uid, 'BeginOp': f'{m}', 'TotalPower': inst_power}


    # export to main pvalloc folder ----------
    npv_df.to_parquet(f'{data_path_def}/output/pvalloc_run/npv_df.parquet')
    pred_inst_df.to_parquet(f'{data_path_def}/output/pvalloc_run/pred_inst_df.parquet')
    pred_inst_df.to_csv(f'{data_path_def}/output/pvalloc_run/pred_inst_df.csv')
    with open (f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'w') as f:
        json.dump(topo, f)


    # saves to interim folder ----------
    npv_df.to_parquet(f'{data_path_def}/output/pvalloc_run/interim_predictions/npv_df_{m}.parquet')
    pred_inst_df.to_parquet(f'{data_path_def}/output/pvalloc_run/interim_predictions/pred_inst_df_{m}.parquet')
    with open(f'{data_path_def}/output/pvalloc_run/interim_predictions/topo_{m}.json', 'w') as f:
        json.dump(topo, f)

    npv_df.to_csv(f'{data_path_def}/output/pvalloc_run/interim_predictions/npv_df_{m}.csv')
    pred_inst_df.to_csv(f'{data_path_def}/output/pvalloc_run/interim_predictions/pred_inst_df_{m}.csv')


    return  inst_power # , picked_uid, picked_combo_uid, pred_inst_df, dfuid_installed_list, topo
