import sys
import os as os
import numpy as np
import pandas as pd
import json
import itertools
import math

from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# IMPORT PREPREPED DATA & CREATE TOPOLOGY
# ------------------------------------------------------------------------------------------------------

def import_prepre_AND_create_topology(
        pvalloc_settings, ):
    """ 
    Reimport all the data that is intermediatley stored in the prepreped folder. Transform it such that it can be used throughout the
    PV allocation algorithm. Return all the data objects back to the main code file. 
    - Input: PV allocation settings
    - Output: all the data objects that are needed for later calculations
    """

    # import settings + setup -------------------
    script_run_on_server_def = pvalloc_settings['script_run_on_server']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    fast_debug_def = pvalloc_settings['fast_debug_run']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    wd_path_def = pvalloc_settings['wd_path']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    bfs_number_def = pvalloc_settings['bfs_numbers']
    gwr_selection_specs_def = pvalloc_settings['gwr_selection_specs']
    print_to_logfile('run function: import_prepreped_data', log_file_name_def)



    # IMPORT & TRANSFORM ============================================================================
    # Import all necessary data objects from prepreped folder and transform them for later calculations
    print_to_logfile('import & transform data', log_file_name_def)

    # transformation functions --------------------------------
    def import_large_or_small_pq_file(path, batchsize, fast_run):
        # import a parquet file, either in full or only small batch for faster developping
        if not fast_run: 
            df = pd.read_parquet(path)
        elif fast_run:
            pf = ParquetFile(path)
            first_n_rows = next(pf.iter_batches(batch_size = batchsize))
            df = first_n_rows.to_pandas()
        return df
        

    # data for topo --------------------------------
    # GWR -------
    gwr = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/gwr.parquet')
    gwr['EGID'] = gwr['EGID'].astype(str)
    
    gwr['GBAUJ'] = gwr['GBAUJ'].replace('', 0).astype(int)
    gwr = gwr.loc[(gwr['GSTAT'].isin(gwr_selection_specs_def['GSTAT'])) & 
                  (gwr['GKLAS'].isin(gwr_selection_specs_def['GKLAS'])) &
                  (gwr['GBAUJ'] >= gwr_selection_specs_def['GBAUJ_minmax'][0]) &
                  (gwr['GBAUJ'] <= gwr_selection_specs_def['GBAUJ_minmax'][1])]
    gwr['GBAUJ'] = gwr['GBAUJ'].replace(0, '').astype(str)

    gwr = gwr.loc[gwr['GGDENR'].isin(bfs_number_def)]
    gwr = gwr.copy()

    # SOLKAT -------
    solkat = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/solkat.parquet')
    solkat['EGID'] = solkat['EGID'].fillna('').astype(str)
    solkat['DF_UID'] = solkat['DF_UID'].fillna('').astype(str)
    solkat['DF_NUMMER'] = solkat['DF_NUMMER'].fillna('').astype(str)
    solkat['SB_UUID'] = solkat['SB_UUID'].fillna('').astype(str)
    solkat['FLAECHE'] = solkat['FLAECHE'].fillna(0).astype(float)
    solkat['STROMERTRAG'] = solkat['STROMERTRAG'].fillna(0).astype(float)

    solkat = solkat.loc[solkat['BFS_NUMMER'].isin(bfs_number_def)]
    solkat = solkat.copy()

    # PV -------
    pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/pv.parquet')
    pv['xtf_id'] = pv['xtf_id'].fillna(0).astype(int).replace(0, '').astype(str)    
    pv['TotalPower'] = pv['TotalPower'].fillna(0).astype(float)

    pv['BeginningOfOperation'] = pd.to_datetime(pv['BeginningOfOperation'], format='%Y-%m-%d', errors='coerce')
    gbauj_range = [pd.to_datetime(f'{gwr_selection_specs_def["GBAUJ_minmax"][0]}-01-01'), pd.to_datetime(f'{gwr_selection_specs_def["GBAUJ_minmax"][1]}-12-31')]
    pv = pv.loc[(pv['BeginningOfOperation'] >= gbauj_range[0]) & (pv['BeginningOfOperation'] <= gbauj_range[1])]
    pv['BeginningOfOperation'] = pv['BeginningOfOperation'].dt.strftime('%Y-%m-%d')

    pv = pv.loc[pv["BFS_NUMMER"].isin(bfs_number_def)]
    pv = pv.copy()

    # Map solkat_dfuid > egid -------
    Map_solkatdfuid_egid = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_solkatdfuid_egid.parquet')
    Map_solkatdfuid_egid['EGID'] = Map_solkatdfuid_egid['EGID'].fillna(0).astype(int).astype(str)
    Map_solkatdfuid_egid['EGID'].replace('0', '', inplace=True) 
    Map_solkatdfuid_egid['DF_UID'] = Map_solkatdfuid_egid['DF_UID'].astype(int).astype(str)
    
    # Map solkat_egid > pv -------
    Map_egid_pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_pv.parquet')
    Map_egid_pv = Map_egid_pv.dropna()
    Map_egid_pv['EGID'] = Map_egid_pv['EGID'].astype(int).astype(str)
    Map_egid_pv['xtf_id'] = Map_egid_pv['xtf_id'].fillna('').astype(int).astype(str)


    # EGID specific data (demand type, cost, etc.) --------
    # NOTE: CLEAN UP when aggregation is adjusted
    if os.path.exists(f'{data_path_def}/output/{name_dir_import_def}/Map_demandtype_EGID.json'):
        with open(f'{data_path_def}/output/{name_dir_import_def}/Map_demandtype_EGID.json', 'r') as file:
            Map_demandtypes_eigd = json.load(file)
    elif os.path.exists(f'{data_path_def}/output/{name_dir_import_def}/Map_demand_type_gwrEGID.json'):
        with open(f'{data_path_def}/output/{name_dir_import_def}/Map_demand_type_gwrEGID.json', 'r') as file:
            Map_demandtypes_eigd = json.load(file)

    with open(f'{data_path_def}/output/{name_dir_import_def}/Map_EGID_demandtypes.json', 'r') as file:
        Map_egid_demandtypes = json.load(file)

    with open(f'{data_path_def}/output/{name_dir_import_def}/pvinstcost_coefficients.json', 'r') as file:
        pvinstcost_coefficients = json.load(file)
    params_pkW = pvinstcost_coefficients['params_pkW']
    coefs_total = pvinstcost_coefficients['coefs_total']


    # PV Cost functions --------
    # Define the interpolation functions using the imported coefficients
    def func_chf_pkW(x, a, b):
        return a + b / x

    estim_instcost_chfpkW = lambda x: func_chf_pkW(x, *params_pkW)

    def func_chf_total_poly(x, coefs_total):
        return sum(c * x**i for i, c in enumerate(coefs_total))

    estim_instcost_chftotal = lambda x: func_chf_total_poly(x, coefs_total)



    # CREATE TOPOLOGY ============================================================================
    print_to_logfile(f'start creating topology - Taking EGIDs from GWR', log_file_name_def)
    log_str1 = f'Of {gwr["EGID"].nunique()} gwrEGIDs, {len(np.intersect1d(gwr["EGID"].unique(), solkat["EGID"].unique()))} covered by solkatEGIDs ({round(len(np.intersect1d(gwr["EGID"].unique(), solkat["EGID"].unique()))/gwr["EGID"].nunique()*100,2)} % covered)'
    log_str2 = f'Solkat specs (WTIH assigned EGID): {solkat.loc[solkat["EGID"] !="", "SB_UUID"].nunique()} of {solkat.loc[:, "SB_UUID"].nunique()} ({round((solkat.loc[solkat["EGID"] !="", "SB_UUID"].nunique() / solkat.loc[:, "SB_UUID"].nunique())*100,2)} %); {solkat.loc[solkat["EGID"] !="", "DF_UID"].nunique()} of {solkat.loc[:, "DF_UID"].nunique()} ({round((solkat.loc[solkat["EGID"] !="", "DF_UID"].nunique() / solkat.loc[:, "DF_UID"].nunique())*100,2)} %)'
    checkpoint_to_logfile(log_str1, log_file_name_def)
    checkpoint_to_logfile(log_str2, log_file_name_def)
    
    # egid = '425447' 
    # len(gwr['EGID'])
    # gwr = gwr.head(3)
    # gwr['EGID'].isin(solkat['EGID'])
    # egid = gwr['EGID'].iloc[0]

    if pvalloc_settings['fast_debug_run']:
        gwr_before_copy = gwr.copy()
        gwr = gwr.iloc[0:pvalloc_settings['n_egid_in_topo']]


    # start loop ------------------------------------------------
    topo = {'EGID': {}}
    modulus_print = int(len(gwr['EGID'])//5)
    checkpoint_to_logfile(f'start attach to topo', log_file_name_def, 1 , True)
    print_to_logfile(f'\n', log_file_name_def)

    for i, egid in enumerate(gwr['EGID']):

         
        # # add pv data --------
        pv_invst = {}
        egid_without_pv = []
        pv_npry = np.array(pv)
        Map_xtf = Map_egid_pv.loc[Map_egid_pv['EGID'] == egid, 'xtf_id']

        if Map_xtf.empty:
            # checkpoint_to_logfile(f'egid {egid} not in PV Mapping (Map_egid_pv?)', log_file_name_def, 3, show_debug_prints_def)
            egid_without_pv.append(egid)
        elif not Map_xtf.empty:
            xtfid = Map_xtf.iloc[0]
            if xtfid not in pv['xtf_id']:
                checkpoint_to_logfile(f'---- pv xtf_id {xtfid} in Mapping_egid_pv, but NOT in pv data', log_file_name_def, 3, False)
                
            if (Map_xtf.shape[0] == 1) and (xtfid in pv['xtf_id']):
                mask_xtfid = np.isin(pv_npry[:, pv.columns.get_loc('xtf_id')], [xtfid,])

                pv_invst['pv_inst_TF'] = True,
                pv_invst['xtf_id'] = str(xtfid),
                
                pv_invst['BeginOp'] = pv_npry[mask_xtfid, pv.columns.get_loc('BeginningOfOperation')][0],
                pv_invst['InitialPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('InitialPower')][0]
                pv_invst['TotalPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('TotalPower')][0]
            
                # pv_invst['BeginOp'] = pv.loc[pv['xtf_id'] == xtfid, 'BeginningOfOperation'].iloc[0]
                # pv_invst['InitialPower'] = pv.loc[pv['xtf_id'] == xtfid, 'InitialPower'].iloc[0]
                # pv_invst['TotalPower'] = pv.loc[pv['xtf_id'] == xtfid, 'TotalPower'].iloc[0]
                
            elif Map_xtf.shape[0] > 1:
                checkpoint_to_logfile(f'ERROR: multiple xtf_ids for EGID: {egid}', log_file_name_def, 3, show_debug_prints_def)


        # if pvalloc_settings['topo_type'] in [5,]:
        if egid in solkat['EGID'].unique():
            solkat_sub = solkat.loc[solkat['EGID'] == egid]
            if solkat.duplicated(subset=['DF_UID', 'EGID']).any():
                solkat_sub = solkat_sub.drop_duplicates(subset=['DF_UID', 'EGID'])
            solkat_partitions = solkat_sub.set_index('DF_UID')[['FLAECHE', 'STROMERTRAG']].to_dict(orient='index') 


            # add solkat combos
            solkat_combos = {}
            combo_partitions = []
            solkat_npry = np.array(solkat_sub)
            partitions = solkat_npry[:, solkat_sub.columns.get_loc('DF_UID')]
            keys = list(partitions)
            if len(keys) < 15:
                for r in range(1, len(keys) + 1):
                    combo = list(itertools.combinations(keys, r))
                    combo_partitions.extend(combo)
                
                for c in combo_partitions:
                    mask_dfuid = np.isin(solkat_npry[:, solkat_sub.columns.get_loc('DF_UID')], c)
                    flaeche = solkat_npry[mask_dfuid, solkat_sub.columns.get_loc('FLAECHE')].sum()
                    conv_m2toKWP = pvalloc_settings['assumed_parameters']['conversion_m2_to_kw']

                    solkat_combos['_'.join(c)] = {
                        'DF_UID': list(c),
                        'DF_NUMMER': solkat_npry[mask_dfuid, solkat_sub.columns.get_loc('DF_NUMMER')].tolist(),
                        'FLAECHE': flaeche,
                        'STROMERTRAG': solkat_npry[mask_dfuid, solkat_sub.columns.get_loc('STROMERTRAG')].sum(),
                        'estim_pvinstcost': estim_instcost_chfpkW(flaeche * conv_m2toKWP),
                    }
            elif len(keys) >= 15:
                checkpoint_to_logfile(f'ATTENTION* : EGID {egid} has more than 15 partitions > not computing combos', log_file_name_def, 3, show_debug_prints_def)
                solkat_combos = {}
                        
        
        elif egid not in solkat['EGID'].unique():
            solkat_partitions = {}
            solkat_combos = {}
            checkpoint_to_logfile(f'egid {egid} not in solkat', log_file_name_def, 3, show_debug_prints_def)

        # add demand type --------
        if egid in Map_egid_demandtypes.keys():
            demand_type = Map_egid_demandtypes[egid]
        elif egid not in Map_egid_demandtypes.keys():
            print_to_logfile(f'\n ** ERROR ** EGID {egid} not in Map_egid_demandtypes, but must be because Map file is based on GWR', log_file_name_def)
            demand_type = 'NA'


        # attach to topo --------
        topo['EGID'][egid] = {
            'pv_inst': pv_invst,
            'solkat_partitions': solkat_partitions, 
            'solkat_combos': solkat_combos,
            'demand_type': demand_type,
            }  
        checkpoint_to_logfile(f'Attachment EGID:{egid} completed', log_file_name_def, 3, show_debug_prints_def) 

        # Checkpoint prints
        if i % modulus_print == 0:
            spacer = f'\t' if i < 100 else f''
            checkpoint_to_logfile(f'EGID {i} of {len(gwr["EGID"])}{spacer}', log_file_name_def, 1, True)
        # checkpoint_to_logfile(f'Attach {egid} to topo {15*"-"}', log_file_name_def, 3, show_debug_prints_def)
        
    # end loop ------------------------------------------------
    # gwr['EGID'].apply(populate_topo_byEGID)
    checkpoint_to_logfile('end attach to topo', log_file_name_def, 1 , True)



    # EXPORT TOPO + Mappings ============================================================================
    topo
    mapping_list = [Map_solkatdfuid_egid, Map_egid_pv, Map_demandtypes_eigd, Map_egid_demandtypes]
    mapping_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_eigd', 'Map_egid_demandtypes']


    with open(f'{data_path_def}/output/pvalloc_run/topo.txt', 'w') as f:
        f.write(str(topo))
    
    with open(f'{data_path_def}/output/pvalloc_run/topo.json', 'w') as f:
        json.dump(topo, f)

    for i, m in enumerate(mapping_list): 
        if isinstance(m, pd.DataFrame):
            m.to_parquet(f'{data_path_def}/output/pvalloc_run/{mapping_names[i]}.parquet')
        elif isinstance(m, dict):
            with open(f'{data_path_def}/output/pvalloc_run/{mapping_names[i]}.json', 'w') as f:
                json.dump(m, f)
        


    # RETURN OBJECTS ============================================================================
    return topo, mapping_list


    # ARCHIV ============================================================================

            # add solkat partitions --------
    if False: # pvalloc_settings['topo_type'] in [1,2]:

            if egid in solkat['EGID'].unique():
                solkat_sub = solkat.loc[solkat['EGID'] == egid]
            
                # drop EGID DF_UID duplicates
                if solkat.duplicated(subset=['DF_UID', 'EGID']).any():
                    # checkpoint_to_logfile(f'\n ATTENTION: EGID-DF_UID duplicates present for EGID {egid} > now dropped', log_file_name_def)
                    solkat_sub = solkat_sub.drop_duplicates(subset=['DF_UID', 'EGID'])
                solkat_partitions = solkat_sub.set_index('DF_UID')[['FLAECHE', 'STROMERTRAG']].to_dict(orient='index')  


                # add solkat combos 
                solkat_combos = {}
                df_uids = solkat_sub['DF_NUMMER'].unique()
                df_uids = solkat_sub['DF_UID'].unique()
                r=2
                total_combinations = sum(math.comb(len(df_uids), r) for r in range(1, len(df_uids) + 1))

                checkpoint_to_logfile(f'---- compute combos for {egid} > {solkat_sub["DF_UID"].nunique()} partitions, {total_combinations} combinations', log_file_name_def, 3, show_debug_prints_def)
                for r in range(1, len(df_uids) + 1):
                    for combo in itertools.combinations(df_uids, r):
                        combo_key_str = '_'.join(map(str, combo))
                        flaeche  = solkat_sub.loc[solkat_sub['DF_UID'].isin(combo), 'FLAECHE'].sum()
                        conv_m2toKWP = pvalloc_settings['assumed_parameters']['conversion_m2_to_kw']

                        solkat_combos[combo_key_str] = {
                            'DF_UID': list(combo),
                            'DF_NUMMER': list(solkat_sub.loc[solkat_sub['DF_UID'].isin(combo), 'DF_NUMMER']),
                            'FLAECHE': flaeche,
                            'STROMERTRAG': solkat_sub.loc[solkat_sub['DF_UID'].isin(combo), 'STROMERTRAG'].sum(),
                            'estim_pvinstcost': estim_instcost_chfpkW(flaeche * conv_m2toKWP), 
                        }
     






        

    





# ------------------------------------------------------------------------------------------------------
# IMPORT TS DATA
# ------------------------------------------------------------------------------------------------------

def import_ts_data(
        pvalloc_settings, ):
    """
    Import the time series data that is needed for the PV allocation algorithm.
    - Input: PV allocation settings
    - Output: all the time series data objects that are needed for later calculations
    """

    # import settings + setup -------------------
    script_run_on_server_def = pvalloc_settings['script_run_on_server']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    fast_debug_def = pvalloc_settings['fast_debug_run']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    wd_path_def = pvalloc_settings['wd_path']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    print_to_logfile('run function: import_ts_data', log_file_name_def)



    # IMPORT ----------------------------------------------------------------------------
    # demand types
    demand_types_tformat = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/demand_types.parquet')
    
    # meteo
    meteo = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/meteo.parquet')

    # create time structure for TS
    # T0 = pd.to_datetime(f'{pvalloc_settings["T0_prediction"]}-01-01 00:00:00')
    # start_year= pvalloc_settings['T0_prediction'] - 
    # date_range = 
    # (8784 -24*365)/24
    
    
    




    
        
        



