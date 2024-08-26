import sys
import os as os
import numpy as np
import pandas as pd
import json
import itertools
import math
import glob
import plotly.graph_objs as go
import plotly.offline as pyo
import geopandas as gpd

from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# GET INTERIM DATA PATH
# ------------------------------------------------------------------------------------------------------
def get_interim_path(pvalloc_settings):
    """
    Return the path to the latest interim folder that ran pvalloc file to the end (and renamed pvalloc_run accordingly)
    """
    data_path_def = pvalloc_settings['data_path']
    name_dir_export_def = pvalloc_settings['name_dir_export']
    log_file_name_def = pvalloc_settings['log_file_name']

    interim_pvalloc_folder = glob.glob(f'{data_path_def}/output/{name_dir_export_def}*')
    if len(interim_pvalloc_folder) == 0:
        checkpoint_to_logfile(f'No existing interim pvalloc folder found, use "pvalloc_run" instead', log_file_name_def)
        iterim_path = f'{data_path_def}/output/pvalloc_run'

    if len(interim_pvalloc_folder) > 0:
        iterim_path = interim_pvalloc_folder[-1]

    return iterim_path


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
    gm_shp = gpd.read_file(f'{data_path_def}\input\swissboundaries3d_2023-01_2056_5728.shp\swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')

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


    # PV TARIF -------
    pvtarif_year = pvalloc_settings['tech_economic_specs']['pvtarif_year']
    pvtarif_col = pvalloc_settings['tech_economic_specs']['pvtarif_col']
    
    Map_gm_ewr = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_gm_ewr.parquet')
    pvtarif = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/pvtarif.parquet')
    pvtarif = pvtarif.merge(Map_gm_ewr, how='left', on='nrElcom')

    # find bfs that have now value in energy1, eco1, etc.


    (pvtarif['bfs'] == "").sum()
    (pvtarif['nrElcom'] == "").sum()
    pvtarif.dtypes

    # ELECTRICITY PRICE -------
    elecpri = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/elecpri.parquet')
    elecpri['bfs_number'] = elecpri['bfs_number'].astype(str)


    # transformation
    pvtarif['bfs'] = pvtarif['bfs'].astype(str)
    pvtarif[pvtarif_col] = pvtarif[pvtarif_col].fillna(0).astype(float)

    pvtarif = pvtarif.loc[(pvtarif['year'] == str(pvtarif_year)[2:4]) & 
                          (pvtarif['bfs'].isin(pvalloc_settings['bfs_numbers']))]

    empty_cols = [col for col in pvtarif.columns if pvtarif[col].isna().all()]
    pvtarif = pvtarif.drop(columns=empty_cols)

    select_cols = ['nrElcom', 'nomEw', 'year', 'bfs', 'idofs'] + pvtarif_col
    pvtarif = pvtarif[select_cols].copy()


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
            Map_demandtypes_egid = json.load(file)
    elif os.path.exists(f'{data_path_def}/output/{name_dir_import_def}/Map_demand_type_gwrEGID.json'):
        with open(f'{data_path_def}/output/{name_dir_import_def}/Map_demand_type_gwrEGID.json', 'r') as file:
            Map_demandtypes_egid = json.load(file)

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

        num_strata = 10
        n_egid_in_topo = pvalloc_settings['n_egid_in_topo']
        gwr['GGDENR'] = pd.to_numeric(gwr['GGDENR'], errors='coerce')
        gwr['strata'] = pd.qcut(gwr['GGDENR'], q=num_strata, labels=False)
        samples_per_stratum = pvalloc_settings['n_egid_in_topo'] // num_strata
        sampled_gwr = gwr.groupby('strata').apply(lambda x: x.sample(n=samples_per_stratum, random_state=1)).reset_index(drop=True)
        if len(sampled_gwr) < n_egid_in_topo:
            additional_samples = gwr.sample(n=n_egid_in_topo - len(sampled_gwr), random_state=1)
            sampled_gwr = pd.concat([sampled_gwr, additional_samples])

        sampled_gwr = sampled_gwr.drop(columns='strata')
        sampled_gwr['GGDENR'] = sampled_gwr['GGDENR'].astype(str)
        gwr = sampled_gwr.copy()
        # gwr = gwr.iloc[0:pvalloc_settings['n_egid_in_topo']]


    # start loop ------------------------------------------------
    # topo = {'EGID': {}}
    topo_egid = {}
    modulus_print = int(len(gwr['EGID'])//5)
    print_to_logfile(f'\n', log_file_name_def)
    checkpoint_to_logfile(f'start attach to topo', log_file_name_def, 1 , True)

    CHECK_egid_with_problems = []

    # transform to np.array for faster lookups
    pv_npry = np.array(pv)
    gwr_npry = np.array(gwr)
    elecpri_npry = np.array(elecpri)

    for i, egid in enumerate(gwr['EGID']):

         
        # add pv data --------
        pv_inst = {}
        egid_without_pv = []
        Map_xtf = Map_egid_pv.loc[Map_egid_pv['EGID'] == egid, 'xtf_id']

        if Map_xtf.empty:
            # checkpoint_to_logfile(f'egid {egid} not in PV Mapping (Map_egid_pv?)', log_file_name_def, 3, show_debug_prints_def)
            egid_without_pv.append(egid)
            pv_inst['inst_TF'] = False
            pv_inst['info_source'] = ''
            pv_inst['xtf_id'] = ''
            pv_inst['BeginOp'] = ''
            pv_inst['InitialPower'] = ''
            pv_inst['TotalPower'] = ''

        elif not Map_xtf.empty:
            xtfid = Map_xtf.iloc[0]
            if xtfid not in pv['xtf_id']:
                checkpoint_to_logfile(f'---- pv xtf_id {xtfid} in Mapping_egid_pv, but NOT in pv data', log_file_name_def, 3, False)
                
            if (Map_xtf.shape[0] == 1) and (xtfid in pv['xtf_id']):
                mask_xtfid = np.isin(pv_npry[:, pv.columns.get_loc('xtf_id')], [xtfid,])

                pv_inst['inst_TF'] = True,
                pv_inst['info_source'] = 'pv_df'
                pv_inst['xtf_id'] = str(xtfid),
                
                pv_inst['BeginOp'] = pv_npry[mask_xtfid, pv.columns.get_loc('BeginningOfOperation')][0],
                pv_inst['InitialPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('InitialPower')][0]
                pv_inst['TotalPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('TotalPower')][0]
            
                # pv_inst['BeginOp'] = pv.loc[pv['xtf_id'] == xtfid, 'BeginningOfOperation'].iloc[0]
                # pv_inst['InitialPower'] = pv.loc[pv['xtf_id'] == xtfid, 'InitialPower'].iloc[0]
                # pv_inst['TotalPower'] = pv.loc[pv['xtf_id'] == xtfid, 'TotalPower'].iloc[0]
                
            elif Map_xtf.shape[0] > 1:
                checkpoint_to_logfile(f'ERROR: multiple xtf_ids for EGID: {egid}', log_file_name_def, 3, show_debug_prints_def)
                CHECK_egid_with_problems.append((egid, 'multiple xtf_ids'))


        # add solkat data --------
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
                        'estim_pvinstcost_chf': estim_instcost_chfpkW(flaeche * conv_m2toKWP),
                    }
            elif len(keys) >= 15:
                checkpoint_to_logfile(f'ATTENTION* : EGID {egid} has more than 15 partitions > not computing combos', log_file_name_def, 3, show_debug_prints_def)
                solkat_combos = {}
                CHECK_egid_with_problems.append((egid, 'more than 15 partitions'))
                        
        
        elif egid not in solkat['EGID'].unique():
            solkat_partitions = {}
            solkat_combos = {}
            checkpoint_to_logfile(f'egid {egid} not in solkat', log_file_name_def, 3, show_debug_prints_def)
            CHECK_egid_with_problems.append((egid, 'not in solkat'))


        # add demand type --------
        if egid in Map_egid_demandtypes.keys():
            demand_type = Map_egid_demandtypes[egid]
        elif egid not in Map_egid_demandtypes.keys():
            print_to_logfile(f'\n ** ERROR ** EGID {egid} not in Map_egid_demandtypes, but must be because Map file is based on GWR', log_file_name_def)
            demand_type = 'NA'
            CHECK_egid_with_problems.append((egid, 'not in Map_egid_demandtypes (both based on GWR)'))

        # add pvtarif --------
        bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0]
        pvtarif_egid = sum([pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), col].sum() for col in pvtarif_col])

        pvtarif_sub = pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid)]
        if pvtarif_sub.empty:
            checkpoint_to_logfile(f'ERROR: no pvtarif data for EGID {egid}', log_file_name_def, 3, show_debug_prints_def)
            ewr_info = {}
            CHECK_egid_with_problems.append((egid, 'no pvtarif data'))
        elif pvtarif_sub.shape[0] == 1:
            ewr_info = {
                'nrElcom': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'nrElcom'].iloc[0],
                'name': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'nomEw'].iloc[0],
                'energy1': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'energy1'].sum(),
                'eco1': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'eco1'].sum(),
            }
        else:
            checkpoint_to_logfile(f'ERROR: multiple pvtarif data for EGID {egid}', log_file_name_def, 3, show_debug_prints_def)
            ewr_info = {}
            CHECK_egid_with_problems.append((egid, 'multiple pvtarif data'))


        # add elecpri --------
        elecpri_egid = {}
        elecpri_info = {}

        mask_bfs = np.isin(elecpri_npry[:, elecpri.columns.get_loc('bfs_number')], [bfs_of_egid,]) 
        mask_year = np.isin(elecpri_npry[:, elecpri.columns.get_loc('year')], [pvalloc_settings['tech_economic_specs']['elecpri_year'],])
        mask_cat = np.isin(elecpri_npry[:, elecpri.columns.get_loc('category')], [pvalloc_settings['tech_economic_specs']['elecpri_category'],])

        if sum(mask_bfs & mask_year & mask_cat) < 1:
            checkpoint_to_logfile(f'ERROR: no elecpri data for EGID {egid}', log_file_name_def, 3, show_debug_prints_def)
            CHECK_egid_with_problems.append((egid, 'no elecpri data'))
        elif sum(mask_bfs & mask_year & mask_cat) > 1:
            checkpoint_to_logfile(f'ERROR: multiple elecpri data for EGID {egid}', log_file_name_def, 3, show_debug_prints_def)
            CHECK_egid_with_problems.append((egid, 'multiple elecpri data'))
        elif sum(mask_bfs & mask_year & mask_cat) == 1:
            energy =   elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('energy')].sum()
            grid =     elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('grid')].sum()
            aidfee =   elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('aidfee')].sum()
            fixcosts = elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('fixcosts')].sum()

            elecpri_egid = energy + grid + aidfee + fixcosts
            elecpri_info = {
                'energy': energy,
                'grid': grid,
                'aidfee': aidfee,
                'fixcosts': fixcosts,
            }

            # add GWR --------
            bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0] 
            glkas_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GKLAS')][0]

            gwr_info ={
                'bfs': bfs_of_egid,
                'gklas': glkas_of_egid,
            }


        # attach to topo --------
        # topo['EGID'][egid] = {
        topo_egid[egid] = {
            'gwr_info': gwr_info,
            'pv_inst': pv_inst,
            'solkat_partitions': solkat_partitions, 
            'solkat_combos': solkat_combos,
            'demand_type': demand_type,
            'pvtarif_Rp_kWh': pvtarif_egid, 
            'EWR': ewr_info, 
            'elecpri_Rp_kWh': elecpri_egid,
            'elecpri_info': elecpri_info,
            }  

        # Checkpoint prints
        if i % modulus_print == 0:
            spacer = f'\t' if i < 100 else f''
            checkpoint_to_logfile(f'EGID {i} of {len(gwr["EGID"])}{spacer}', log_file_name_def, 1, True)
        # checkpoint_to_logfile(f'Attach {egid} to topo {15*"-"}', log_file_name_def, 3, show_debug_prints_def)
        
    # end loop ------------------------------------------------
    # gwr['EGID'].apply(populate_topo_byEGID)
    checkpoint_to_logfile('end attach to topo', log_file_name_def, 1 , True)



    # EXPORT TOPO + Mappings ============================================================================
    topo_egid

    with open(f'{data_path_def}/output/pvalloc_run/topo_egid.txt', 'w') as f:
        f.write(str(topo_egid))
    
    with open(f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'w') as f:
        json.dump(topo_egid, f)

    # Export CHECK_egid_with_problems to txt file for trouble shooting
    with open(f'{data_path_def}/output/pvalloc_run/CHECK_egid_with_problems.txt', 'w') as f:
        f.write(f'\n ** EGID with problems: {len(CHECK_egid_with_problems)} **\n\n')
        f.write(str(CHECK_egid_with_problems))



    df_list = [Map_solkatdfuid_egid, Map_egid_pv, Map_demandtypes_egid, Map_egid_demandtypes, pvtarif, elecpri]
    df_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_egid', 'Map_egid_demandtypes', 'pvtarif', 'elecpri']

    for i, m in enumerate(df_list): 
        if isinstance(m, pd.DataFrame):
            m.to_parquet(f'{data_path_def}/output/pvalloc_run/{df_names[i]}.parquet')
        elif isinstance(m, dict):
            with open(f'{data_path_def}/output/pvalloc_run/{df_names[i]}.json', 'w') as f:
                json.dump(m, f)
        


    # RETURN OBJECTS ============================================================================
    return topo_egid, df_list, df_names

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
# import existing topology
# ------------------------------------------------------------------------------------------------------
def import_exisitng_topology(
        pvalloc_settings,
        df_search_names ):
    name_dir_export_def = pvalloc_settings['name_dir_export']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    interim_path_def = pvalloc_settings['interim_path']

    print_to_logfile('run function: import_existing_topology', log_file_name_def)

    # import existing topo & Mappings ---------
    topo = json.load(open(f'{interim_path_def}/topo_egid.json', 'r'))
        
    df_names = df_search_names
    df_list = []
    for m in df_names:
        f = glob.glob(f'{interim_path_def}/{m}.*')
        if len(f) == 1:
            if f[0].split('.')[1] == 'json':
                df_list.append(json.load(open(f[0], 'r')))
            elif f[0].split('.')[1] == 'parquet':
                df_list.append(pd.read_parquet(f[0]))
    
    return topo, df_list, df_names



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


    # create time structure for TS
    # T0 = pd.to_datetime(f'{pvalloc_settings["T0_prediction"]}-01-01 00:00:00')
    T0 = pd.to_datetime(f'{pvalloc_settings["T0_prediction"]}')
    start_loockback = T0 - pd.DateOffset(months=pvalloc_settings['months_lookback']) # + pd.DateOffset(hours=1)
    end_prediction = T0 + pd.DateOffset(months=pvalloc_settings['months_prediction']) - pd.DateOffset(hours=1)
    date_range = pd.date_range(start=start_loockback, end=end_prediction, freq='H')
    checkpoint_to_logfile(f'import TS: lookback range   {start_loockback} to {T0-pd.DateOffset(hours=1)}', log_file_name_def, 2)
    checkpoint_to_logfile(f'import TS: prediction range {T0} to {end_prediction}', log_file_name_def, 2)

    Map_daterange = pd.DataFrame({'date_range': date_range, 'DoY': date_range.dayofyear, 'hour': date_range.hour})
    Map_daterange['HoY'] = (Map_daterange['DoY'] - 1) * 24 + (Map_daterange['hour']+1)
    Map_daterange['t'] = Map_daterange['HoY'].apply(lambda x: f't_{x}')

    

    # IMPORT ----------------------------------------------------------------------------

    # demand types --------
    demandtypes_tformat = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/demand_types.parquet')
    demandtypes_ts = demandtypes_tformat.copy()

    # demandtypes_ts = demandtypes_ts.merge(Map_daterange[['date_range', 't']], how='left', on='t')
    # demandtypes_ts.drop(columns=['t'], inplace=True)
    # demandtypes_ts.sort_values(by='date_range', inplace=True)
    nas =   sum([demandtypes_ts[col].isna().sum() for col in demandtypes_ts.columns])
    nulls = sum([demandtypes_ts[col].isnull().sum() for col in demandtypes_ts.columns])
    checkpoint_to_logfile(f'sanity check demand_ts: {nas} NaNs or {nulls} Nulls for any column in df', log_file_name_def)


    # meteo types --------
    rad_proxy = pvalloc_settings['meteoblue_col_radiation_proxy']
    weater_year = pvalloc_settings['weather_year']

    meteo = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/meteo.parquet')
    meteo = meteo.loc[:,['timestamp', rad_proxy]]
    meteo.rename(columns={rad_proxy: 'radiation'}, inplace=True)
    
    start_wy, end_wy = pd.to_datetime(f'{weater_year}-01-01 00:00:00'), pd.to_datetime(f'{weater_year}-12-31 23:00:00')
    meteo = meteo.loc[(meteo['timestamp'] >= start_wy) & (meteo['timestamp'] <= end_wy)]

    meteo['t']= meteo['timestamp'].apply(lambda x: f't_{(x.dayofyear -1) * 24 + x.hour +1}')
    meteo_ts = meteo.copy()


    # grid premium --------
    t_list = [f't_{i}' for i in range(1, 8761)]
    gridprem = pd.DataFrame({'t': t_list, 'prem_Rp_kWh': 0.0})
    gridprem_ts = gridprem.copy()



    # pvtarif --------
    # is covered in import and prepare topology
    """
    pvtarif_year = pvalloc_settings['pricing_specs']['pvtarif_year']
    pvtarif_col = pvalloc_settings['pricing_specs']['pvtarif_col']
    
    Map_gm_ewr = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_gm_ewr.parquet')
    pvtarif = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/pvtarif.parquet')
    pvtarif = pvtarif.merge(Map_gm_ewr, how='left', on='nrElcom')

    # transformation
    pvtarif = pvtarif.loc[(pvtarif['year'] == str(pvtarif_year)[2:4]) & 
                          (pvtarif['bfs'].astype(str).isin(pvalloc_settings['bfs_numbers']))]

    empty_cols = [col for col in pvtarif.columns if pvtarif[col].isna().all()]
    pvtarif = pvtarif.drop(columns=empty_cols)

    select_cols = ['nrElcom', 'nomEw', 'year', 'bfs', 'idofs'] + pvtarif_col
    pvtarif.drop(columns = select_cols, inplace=True)
    """
    

    # EXPORT ----------------------------------------------------------------------------
    ts_names = ['Map_daterange', 'demandtypes_ts', 'meteo_ts', 'gridprem_ts']
    ts_list = [Map_daterange, demandtypes_ts, meteo_ts, gridprem_ts]
    for i, ts in enumerate(ts_list):
        ts.to_parquet(f'{data_path_def}/output/pvalloc_run/{ts_names[i]}.parquet')


    # RETURN ----------------------------------------------------------------------------
    return ts_list, ts_names

# ------------------------------------------------------------------------------------------------------
# import existing ts data
# ------------------------------------------------------------------------------------------------------
# > for now this part of the code runs in seconds, not needed to rely on previous runs to make it faster. 
"""
def import_exisitng_ts_data(
        pvalloc_settings, ):
    
    name_dir_export_def = pvalloc_settings['name_dir_export']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']

    print_to_logfile('run function: import_existing_topology', log_file_name_def)

    # import existing topo & Mappings ---------
    interim_pvalloc_folder = glob.glob(f'{data_path_def}/output/{name_dir_export_def}*')
    if len(interim_pvalloc_folder) == 0:
        checkpoint_to_logfile(f'ERROR: No existing interim pvalloc folder found', log_file_name_def)
        interim_path = f'{data_path_def}/output/pvalloc_run' 

    if len(interim_pvalloc_folder) >  1:
        interim_path = interim_pvalloc_folder[-1]
    else:
        interim_path = interim_pvalloc_folder[0]

    ts_names = ['Map_daterange', 'demandtypes_ts', 'meteo_ts']
    ts_list = []

    for t in ts_names:
        f = glob.glob(f'{interim_path}/{t}.*')
        if len(f) == 1:
            ts_list.append(pd.read_parquet(f[0]))

    return ts_list
"""
    


    
# ------------------------------------------------------------------------------------------------------
# CONSTRUCTION CAPACITY for pv installations
# ------------------------------------------------------------------------------------------------------
def define_construction_capacity(
        pvalloc_settings, 
        topo_func, 
        ts_list_func,):
    """
    Based on the selection of pvalloc_settings for the prediction, this function will define a time series for
    the construction capacity of new PV installations for the comming months. 
    - Input: PV allocation settings, topo, ts_list
    - Output: construction capacity time series
    """

    # import settings + setup -------------------
    script_run_on_server_def = pvalloc_settings['script_run_on_server']
    fast_debug_def = pvalloc_settings['fast_debug_run']   
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']

    topo = topo_func
    ts_list = ts_list_func
    print_to_logfile('run function: define_construction_capacity.py', log_file_name_def)


    # create monthly time structure
    T0 = pd.to_datetime(f'{pvalloc_settings["T0_prediction"]}')
    start_loockback = T0 - pd.DateOffset(months=pvalloc_settings['months_lookback']) #+ pd.DateOffset(hours=1)
    end_prediction = T0 + pd.DateOffset(months=pvalloc_settings['months_prediction']) - pd.DateOffset(hours=1)
    month_range = pd.date_range(start=start_loockback, end=end_prediction, freq='M').to_period('M')
    months_lookback = pd.date_range(start=start_loockback, end=T0, freq='M').to_period('M')
    months_prediction = pd.date_range(start=(T0 + pd.DateOffset(days=1)), end=end_prediction, freq='M').to_period('M')
    checkpoint_to_logfile(f'constr_capacity: month lookback   {months_lookback[0]} to {months_lookback[-1]}', log_file_name_def, 2)
    checkpoint_to_logfile(f'constr_capacity: month prediction {months_prediction[0]} to {months_prediction[-1]}', log_file_name_def, 2)


    # IMPORT ----------------------------------------------------------------------------
    Map_daterange = ts_list[0]
    pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/pv.parquet')
    Map_egid_pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_pv.parquet')

    topo_keys = list(topo.keys())

    # subset pv to EGIDs in TOPO, and LOOKBACK period of pvalloc settings
    pv_sub = pv.copy()
    del_cols = ['MainCategory', 'SubCategory', 'PlantCategory']
    pv_sub.drop(columns=del_cols, inplace=True)

    pv_sub = pv_sub.merge(Map_egid_pv, how='left', on='xtf_id')
    pv_sub = pv_sub.loc[pv_sub['EGID'].isin(topo_keys)]
    pv_plot = pv_sub.copy() # used for plotting later

    pv_sub['BeginningOfOperation'] = pd.to_datetime(pv_sub['BeginningOfOperation'])
    pv_sub['MonthPeriod'] = pv_sub['BeginningOfOperation'].dt.to_period('M')
    pv_sub = pv_sub.loc[pv_sub['MonthPeriod'].isin(months_lookback)]

    # plot total power over time
    if True: 
        pv_plot['BeginningOfOperation'] = pd.to_datetime(pv_plot['BeginningOfOperation'])
        pv_plot.set_index('BeginningOfOperation', inplace=True)

        # Resample by week, month, and year and calculate the sum of TotalPower
        weekly_sum = pv_plot['TotalPower'].resample('W').sum()
        monthly_sum = pv_plot['TotalPower'].resample('M').sum()
        yearly_sum = pv_plot['TotalPower'].resample('Y').sum()

        # Create traces for each time period
        trace_weekly = go.Scatter(x=weekly_sum.index, y=weekly_sum.values, mode='lines', name='Weekly')
        trace_monthly = go.Scatter(x=monthly_sum.index, y=monthly_sum.values, mode='lines', name='Monthly')
        trace_yearly = go.Scatter(x=yearly_sum.index, y=yearly_sum.values, mode='lines', name='Yearly')

        layout = go.Layout(
            title='Built PV Capacity within Sample of GWR EGIDs',
            xaxis=dict(title='Time',
                    range = ['2010-01-01', '2024-5-31']),
            yaxis=dict(title='Total Power'),
            legend=dict(x=0, y=1),
                 shapes=[
                    # Shaded region for months_lookback
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=months_lookback[0].start_time,
                        x1=months_lookback[-1].end_time,
                        y0=0,
                        y1=1,
                        fillcolor="LightSalmon",
                        opacity=0.3,
                        layer="below",
                        line_width=0,
            )
        ]
        )
        fig = go.Figure(data=[trace_weekly, trace_monthly, trace_yearly], layout=layout)
        # pyo.plot(fig)
        fig.write_html(f'{data_path_def}/output/pvalloc_run/pv_total_power_over_time.html')


    # CAPACITY ASSIGNMENT ----------------------------------------------------------------------------
    capacity_growth = pvalloc_settings['constr_capacity_specs']['ann_capacity_growth']
    summer_months = pvalloc_settings['constr_capacity_specs']['summer_months']
    winter_months = pvalloc_settings['constr_capacity_specs']['winter_months']
    share_to_summer = pvalloc_settings['constr_capacity_specs']['share_to_summer']
    share_to_winter = pvalloc_settings['constr_capacity_specs']['share_to_winter']

    sum_TP_kW_lookback = pv_sub['TotalPower'].sum()

    constrcapa = pd.DataFrame({'date': months_prediction, 'year': months_prediction.year, 'month': months_prediction.month})
    years_prediction = months_prediction.year.unique()
    for i,y in enumerate(years_prediction):

        TP_y = sum_TP_kW_lookback * (1 + capacity_growth)**(i+1)
        TP_y_summer_month = TP_y * share_to_summer / len(summer_months)
        TP_y_winter_month = TP_y * share_to_winter / len(winter_months)

        constrcapa.loc[(constrcapa['year'] == y) & 
                       (constrcapa['month'].isin(summer_months)), 'constr_capacity_kw'] = TP_y_summer_month
        constrcapa.loc[(constrcapa['year'] == y) &
                       (constrcapa['month'].isin(winter_months)), 'constr_capacity_kw'] = TP_y_winter_month
        

    # EXPORT ----------------------------------------------------------------------------
    constrcapa.to_parquet(f'{data_path_def}/output/pvalloc_run/constrcapa.parquet')
    constrcapa.to_csv(f'{data_path_def}/output/pvalloc_run/constrcapa.csv', index=False)

    return constrcapa, months_prediction, months_lookback




    
        
        



