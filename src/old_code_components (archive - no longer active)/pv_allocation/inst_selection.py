import sys
import os as os
import numpy as np
import pandas as pd
import json
import copy

# own functions 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import  print_to_logfile

def select_AND_adjust_topology(
        scen,
        subdir_path, 
        dfuid_installed_list_func, 
        pred_inst_df_func,
        m, i_m):
    
    # select a random building out of npv_df_func to attribute a PV system to
    # wd_path = pvalloc_settings['wd_path']
    # data_path = f'{wd_path}_data'
    # rand_seed = pvalloc_settings['algorithm_specs']['rand_seed']
    # kWpeak_per_m2 = pvalloc_settings['tech_economic_specs']['kWpeak_per_m2']
    # share_roof_area_available = pvalloc_settings['tech_economic_specs']['share_roof_area_available']
    # inst_selection_method = pvalloc_settings['algorithm_specs']['inst_selection_method']
    
    
    print_to_logfile('run function: select_AND_adjust_topology', scen.log_name)
    i_alloc_loop = i_m


    # import ----------
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
    npv_df = pd.read_parquet(f'{subdir_path}/npv_df.parquet') 
    pred_inst_df = pd.read_parquet(f'{subdir_path}/pred_inst_df.parquet') if os.path.exists(f'{subdir_path}/pred_inst_df.parquet') else pd.DataFrame()


    # drop installed partitions from npv_df 
    #   -> otherwise multiple selection possible
    #   -> easier to drop inst before each selection than to create a list / df and carry it through the entire code)
    npv_df_start_inst_selection = copy.deepcopy(npv_df)
    egid_wo_inst = [egid for egid in topo if topo.get(egid, {}).get('pv_inst', {}).get('inst_TF') == False]
    npv_df = copy.deepcopy(npv_df.loc[npv_df['EGID'].isin(egid_wo_inst)])


    # SELECTION BY METHOD ---------------
    # set random seed
    if scen.ALGOspec_rand_seed is not None:
        np.random.seed(scen.ALGOspec_rand_seed)

    # have a list of egids to install on for sanity check. If all build, start building on the rest of EGIDs
    install_EGIDs_summary_sanitycheck = scen.CHECKspec_egid_list

    if isinstance(install_EGIDs_summary_sanitycheck, list):
        # remove duplicates from install_EGIDs_summary_sanitycheck
        unique_EGID = []
        for e in install_EGIDs_summary_sanitycheck:
                if e not in unique_EGID:
                    unique_EGID.append(e)
        install_EGIDs_summary_sanitycheck = unique_EGID
        # get remaining EGIDs of summary_sanitycheck_list that are not yet installed
        # > not even necessary if installed EGIDs get dropped from npv_df?
        remaining_egids = [
            egid for egid in install_EGIDs_summary_sanitycheck 
            if not topo.get(egid, {}).get('pv_inst', {}).get('inst_TF', False) == False ]
        
        if any([True if egid in npv_df['EGID'] else False for egid in remaining_egids]):
            npv_df = npv_df.loc[npv_df['EGID'].isin(remaining_egids)].copy()
        else:
            npv_df = npv_df.copy()
            

    if scen.ALGOspec_inst_selection_method == 'random':
        npv_pick = npv_df.sample(n=1).copy()
    
    elif scen.ALGOspec_inst_selection_method == 'max_npv':
        npv_pick = npv_df[npv_df['NPV_uid'] == max(npv_df['NPV_uid'])].copy()

    elif scen.ALGOspec_inst_selection_method == 'prob_weighted_npv':
        rand_num = np.random.uniform(0, 1)
        
        npv_df['NPV_stand'] = npv_df['NPV_uid'] / max(npv_df['NPV_uid'])
        npv_df['diff_NPV_rand'] = abs(npv_df['NPV_stand'] - rand_num)
        npv_pick = npv_df[npv_df['diff_NPV_rand'] == min(npv_df['diff_NPV_rand'])].copy()
        
        # if multiple rows at min to rand num 
        if npv_pick.shape[0] > 1:
            rand_row = np.random.randint(0, npv_pick.shape[0])
            npv_pick = npv_pick.iloc[rand_row]


    # remove cols for uniform format between selection methods
    for col in ['NPV_stand', 'diff_NPV_rand']:
        if col in npv_df.columns:
            npv_df.drop(columns=['NPV_stand', 'diff_NPV_rand'], inplace=True)
    # ---------------


    if isinstance(npv_pick, pd.DataFrame):
        picked_egid = npv_pick['EGID'].values[0]
        picked_uid = npv_pick['df_uid_combo'].values[0]
        picked_flaech = npv_pick['FLAECHE'].values[0]
        for col in ['NPV_stand', 'diff_NPV_rand']:
            if col in npv_pick.columns:
                npv_pick.drop(columns=['NPV_stand', 'diff_NPV_rand'], inplace=True)

    elif isinstance(npv_pick, pd.Series):
        picked_egid = npv_pick['EGID']
        picked_uid = npv_pick['df_uid_combo']
        picked_flaech = npv_pick['FLAECHE']
        for col in ['NPV_stand', 'diff_NPV_rand']:
            if col in npv_pick.index:
                npv_pick.drop(index=['NPV_stand', 'diff_NPV_rand'], inplace=True)
                
    inst_power = picked_flaech * scen.TECspec_kWpeak_per_m2 * scen.TECspec_share_roof_area_available
    npv_pick['inst_TF'], npv_pick['info_source'], npv_pick['xtf_id'], npv_pick['BeginOp'], npv_pick['TotalPower'], npv_pick['iter_round'] = [True, 'alloc_algorithm', picked_uid, f'{m}', inst_power, i_alloc_loop]
    

    # Adjust export lists / df
    if '_' in picked_uid:
        picked_combo_uid = list(picked_uid.split('_'))
    else:
        picked_combo_uid = [picked_uid]

    if isinstance(npv_pick, pd.DataFrame):
        pred_inst_df = pd.concat([pred_inst_df, npv_pick])
    elif isinstance(npv_pick, pd.Series):
        pred_inst_df = pd.concat([pred_inst_df, npv_pick.to_frame().T])
    

    # Adjust topo
    topo[picked_egid]['pv_inst'] = {'inst_TF': True, 'info_source': 'alloc_algorithm', 'xtf_id': picked_uid, 'BeginOp': f'{m}', 'TotalPower': inst_power}


    # export main dfs ------------------------------------------
    # do not overwrite the original npv_df, this way can reimport it every month and filter for sanitycheck
    pred_inst_df.to_parquet(f'{subdir_path}/pred_inst_df.parquet')
    pred_inst_df.to_csv(f'{subdir_path}/pred_inst_df.csv') if scen.export_csvs else None
    with open (f'{subdir_path}/topo_egid.json', 'w') as f:
        json.dump(topo, f)


    # export by Month ------------------------------------------
    pred_inst_df.to_parquet(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{m}.parquet')
    pred_inst_df.to_csv(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{m}.csv') if scen.export_csvs else None
    with open(f'{subdir_path}/pred_npv_inst_by_M/topo_{m}.json', 'w') as f:
        json.dump(topo, f)
                
    return  inst_power, npv_df# , picked_uid, picked_combo_uid, pred_inst_df, dfuid_installed_list, topo
