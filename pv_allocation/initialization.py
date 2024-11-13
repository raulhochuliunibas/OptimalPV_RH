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
import copy

from pyarrow.parquet import ParquetFile
from shapely.ops import nearest_points


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
        checkpoint_to_logfile(f'ATTENTION! No existing interim pvalloc folder found, use "pvalloc_run" instead', log_file_name_def)
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
    # gm_shp = gpd.read_file(f'{data_path_def}\input\swissboundaries3d_2023-01_2056_5728.shp\swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')

    # GWR -------
    gwr = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/gwr.parquet')
    gwr['EGID'] = gwr['EGID'].astype(str)
    
    gwr['GBAUJ'] = gwr['GBAUJ'].replace('', 0).astype(int)
    gwr = gwr.loc[(gwr['GSTAT'].isin(gwr_selection_specs_def['GSTAT'])) & 
                  (gwr['GKLAS'].isin(gwr_selection_specs_def['GKLAS'])) &
                  (gwr['GBAUJ'] >= gwr_selection_specs_def['GBAUJ_minmax'][0]) &
                  (gwr['GBAUJ'] <= gwr_selection_specs_def['GBAUJ_minmax'][1])]
    gwr['GBAUJ'] = gwr['GBAUJ'].replace(0, '').astype(str)

    if pvalloc_settings['gwr_selection_specs']['dwelling_cols'] == None: 
        gwr = gwr.loc[:, pvalloc_settings['gwr_selection_specs']['building_cols']].copy()
        gwr = gwr.drop_duplicates(subset=['EGID'])


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
    solkat['AUSRICHTUNG'] = solkat['AUSRICHTUNG'].astype(int)
    solkat['NEIGUNG'] = solkat['NEIGUNG'].astype(int)


    # NOTE: remove building with maximal (outlier large) number of partitions => complicates the creation of partition combinations
    solkat['EGID'].value_counts()
    egid_counts = solkat['EGID'].value_counts()
    egids_below_max = list(egid_counts[egid_counts < pvalloc_settings['gwr_selection_specs']['solkat_max_n_partitions']].index)
    solkat = solkat.loc[solkat['EGID'].isin(egids_below_max)]
    # -

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

    pvtarif['bfs'] = pvtarif['bfs'].astype(str)
    # pvtarif[pvtarif_col] = pvtarif[pvtarif_col].fillna(0).astype(float)
    pvtarif[pvtarif_col] = pvtarif[pvtarif_col].replace('', 0).astype(float)

    # transformation
    pvtarif = pvtarif.loc[(pvtarif['year'] == str(pvtarif_year)[2:4]) & 
                          (pvtarif['bfs'].isin(pvalloc_settings['bfs_numbers']))]

    empty_cols = [col for col in pvtarif.columns if pvtarif[col].isna().all()]
    pvtarif = pvtarif.drop(columns=empty_cols)

    select_cols = ['nrElcom', 'nomEw', 'year', 'bfs', 'idofs'] + pvtarif_col
    pvtarif = pvtarif[select_cols].copy()


    # ELECTRICITY PRICE -------
    elecpri = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/elecpri.parquet')
    elecpri['bfs_number'] = elecpri['bfs_number'].astype(str)


    # Map solkat_dfuid > egid -------
    Map_solkatdfuid_egid = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_solkatdfuid_egid.parquet')
    Map_solkatdfuid_egid['EGID'] = Map_solkatdfuid_egid['EGID'].fillna(0).astype(int).astype(str)
    # Map_solkatdfuid_egid['EGID'].replace('0', '', inplace=True)  adjusted for pandas 3.0
    Map_solkatdfuid_egid.replace({'EGID': '0'}, '', inplace=True)
    Map_solkatdfuid_egid['DF_UID'] = Map_solkatdfuid_egid['DF_UID'].astype(int).astype(str)
    
    # Map solkat_egid > pv -------
    Map_egid_pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_pv.parquet')
    Map_egid_pv = Map_egid_pv.dropna()
    Map_egid_pv['EGID'] = Map_egid_pv['EGID'].astype(int).astype(str)
    Map_egid_pv['xtf_id'] = Map_egid_pv['xtf_id'].fillna('').astype(int).astype(str)


    # Map demandtypes > egid -------
    # NOTE: CLEAN UP when aggregation is adjusted, should be no longer used!
    if os.path.exists(f'{data_path_def}/output/{name_dir_import_def}/Map_demandtype_EGID.json'):
        with open(f'{data_path_def}/output/{name_dir_import_def}/Map_demandtype_EGID.json', 'r') as file:
            Map_demandtypes_egid = json.load(file)

    elif os.path.exists(f'{data_path_def}/output/{name_dir_import_def}/Map_demand_type_gwrEGID.json'):
        with open(f'{data_path_def}/output/{name_dir_import_def}/Map_demand_type_gwrEGID.json', 'r') as file:
            Map_demandtypes_egid = json.load(file)


    # Map egid > demandtypes -------
    with open(f'{data_path_def}/output/{name_dir_import_def}/Map_EGID_demandtypes.json', 'r') as file:
        Map_egid_demandtypes = json.load(file)


    # Func pv installation cost -------
    with open(f'{data_path_def}/output/{name_dir_import_def}/pvinstcost_coefficients.json', 'r') as file:
        pvinstcost_coefficients = json.load(file)
    params_pkW = pvinstcost_coefficients['params_pkW']
    coefs_total = pvinstcost_coefficients['coefs_total']


    # Map egid > node -------
    Map_egid_nodes = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_nodes.parquet')
    Map_egid_nodes.index = Map_egid_nodes['EGID']


    # angle_tilt_df -------
    angle_tilt_df = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/angle_tilt_df.parquet')


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


    if pvalloc_settings['fast_debug_run']:
        gwr_before_copy = gwr.copy()
        # a more diverse small sample of gwr to have multiple BFS gemeinde in sample. 

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

    # check how many of gwr's EGIDs are in solkat and pv
    len(np.intersect1d(gwr['EGID'].unique(), solkat['EGID'].unique()))
    len(np.intersect1d(gwr['EGID'].unique(), Map_egid_pv['EGID'].unique()))

    # throw out all EGIDs of GWR that are not in solkat
    # NOTE: this could be troublesome :/ check in QGIS if large share of buildings are missing.  
    gwr = gwr.loc[gwr['EGID'].isin(solkat['EGID'].unique())]

    for i, egid in enumerate(gwr['EGID']):

        # add pv data --------
        pv_inst = {
            'inst_TF': False,
            'info_source': '',
            'xtf_id': '',
            'BeginOp': '',
            'InitialPower': '',
            'TotalPower': '',
        }
        egid_without_pv = []
        Map_xtf = Map_egid_pv.loc[Map_egid_pv['EGID'] == egid, 'xtf_id']

        if Map_xtf.empty:
            egid_without_pv.append(egid)

        elif not Map_xtf.empty:
            xtfid = Map_xtf.iloc[0]
            if xtfid not in pv['xtf_id'].values:
                checkpoint_to_logfile(f'---- pv xtf_id {xtfid} in Mapping_egid_pv, but NOT in pv data', log_file_name_def, 3, False)
                
            if (Map_xtf.shape[0] == 1) and (xtfid in pv['xtf_id'].values):
                mask_xtfid = np.isin(pv_npry[:, pv.columns.get_loc('xtf_id')], [xtfid,])

                pv_inst['inst_TF'] = True
                pv_inst['info_source'] = 'pv_df'
                pv_inst['xtf_id'] = str(xtfid)
                
                pv_inst['BeginOp'] = pv_npry[mask_xtfid, pv.columns.get_loc('BeginningOfOperation')][0]
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
            solkat_partitions = solkat_sub.set_index('DF_UID')[['FLAECHE', 'STROMERTRAG', 'AUSRICHTUNG', 'NEIGUNG']].to_dict(orient='index')                   
        
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
        elif pvtarif_sub.shape[0] > 1:
            ewr_info = {
                'nrElcom': pvtarif_sub['nrElcom'].unique().tolist(),
                'name': pvtarif_sub['nomEw'].unique().tolist(),
                'energy1': pvtarif_sub['energy1'].mean(),
                'eco1': pvtarif_sub['eco1'].mean(),
            }
        
            # checkpoint_to_logfile(f'multiple pvtarif data for EGID {egid}', log_file_name_def, 3, show_debug_prints_def)
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
            taxes =    elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('taxes')].sum()
            fixcosts = elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('fixcosts')].sum()

            elecpri_egid = energy + grid + aidfee + taxes + fixcosts
            elecpri_info = {
                'energy': energy,
                'grid': grid,
                'aidfee': aidfee,
                'taxes': taxes,
                'fixcosts': fixcosts,
            }


            # add GWR --------
            bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0] 
            glkas_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GKLAS')][0]

            gwr_info ={
                'bfs': bfs_of_egid,
                'gklas': glkas_of_egid,
            }

            # add grid node --------
            if isinstance(Map_egid_nodes.loc[egid, 'grid_node'], str):
                grid_node = Map_egid_nodes.loc[egid, 'grid_node']
            elif isinstance(Map_egid_nodes.loc[egid, 'grid_node'], pd.Series):
                grid_node = Map_egid_nodes.loc[egid, 'grid_node'].iloc[0]
                

        # attach to topo --------
        # topo['EGID'][egid] = {
        topo_egid[egid] = {
            'gwr_info': gwr_info,
            'grid_node': grid_node,
            'pv_inst': pv_inst,
            'solkat_partitions': solkat_partitions, 
            # 'solkat_combos': solkat_combos,
            'demand_type': demand_type,
            'pvtarif_Rp_kWh': pvtarif_egid, 
            'EWR': ewr_info, 
            'elecpri_Rp_kWh': elecpri_egid,
            'elecpri_info': elecpri_info,
            }  

        # Checkpoint prints
        if i % modulus_print == 0:
            print_to_logfile(f'\t -- EGID {i} of {len(gwr["EGID"])} {15*"-"}', log_file_name_def)

        
    # end loop ------------------------------------------------
    # gwr['EGID'].apply(populate_topo_byEGID)
    checkpoint_to_logfile('end attach to topo', log_file_name_def, 1 , True)
    print_to_logfile(f'\nsanity check for installtions in topo_egid', pvalloc_settings['summary_file_name'])
    checkpoint_to_logfile(f'number of EGIDs with multiple installations: {CHECK_egid_with_problems.count("multiple xtf_ids")}', pvalloc_settings['summary_file_name'])


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

    CHECK_egid_with_problems_dict = {egid: problem for egid, problem in CHECK_egid_with_problems}
    with open(f'{data_path_def}/output/pvalloc_run/CHECK_egid_with_problems.json', 'w') as f:
        json.dump(CHECK_egid_with_problems_dict, f)


    df_list =  [Map_solkatdfuid_egid,   Map_egid_pv,    Map_demandtypes_egid,   Map_egid_demandtypes,   pv,  pvtarif,   elecpri,    angle_tilt_df,  Map_egid_nodes]
    df_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_egid', 'Map_egid_demandtypes', 'pv', 'pvtarif', 'elecpri', 'angle_tilt_df', 'Map_egid_nodes']

    for i, m in enumerate(df_list): 
        if isinstance(m, pd.DataFrame):
            m.to_parquet(f'{data_path_def}/output/pvalloc_run/{df_names[i]}.parquet')
        elif isinstance(m, dict):
            with open(f'{data_path_def}/output/pvalloc_run/{df_names[i]}.json', 'w') as f:
                json.dump(m, f)        


    # RETURN OBJECTS ============================================================================
    return topo_egid, df_list, df_names




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
    date_range = pd.date_range(start=start_loockback, end=end_prediction, freq='h')
    checkpoint_to_logfile(f'import TS: lookback range   {start_loockback} to {T0-pd.DateOffset(hours=1)}', log_file_name_def, 2)
    checkpoint_to_logfile(f'import TS: prediction range {T0} to {end_prediction}', log_file_name_def, 2)

    Map_daterange = pd.DataFrame({'date_range': date_range, 'DoY': date_range.dayofyear, 'hour': date_range.hour})
    Map_daterange['HoY'] = (Map_daterange['DoY'] - 1) * 24 + (Map_daterange['hour']+1)
    Map_daterange['t'] = Map_daterange['HoY'].apply(lambda x: f't_{x}')

    

    # IMPORT ----------------------------------------------------------------------------

    # demand types --------
    demandtypes_tformat = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/demandtypes.parquet')
    demandtypes_ts = demandtypes_tformat.copy()

    nas =   sum([demandtypes_ts[col].isna().sum() for col in demandtypes_ts.columns])
    nulls = sum([demandtypes_ts[col].isnull().sum() for col in demandtypes_ts.columns])
    checkpoint_to_logfile(f'sanity check demand_ts: {nas} NaNs or {nulls} Nulls for any column in df', log_file_name_def)


    # meteo types --------
    rad_proxy = pvalloc_settings['weather_specs']['meteoblue_col_radiation_proxy']
    weater_year = pvalloc_settings['weather_specs']['weather_year']

    meteo = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/meteo.parquet')
    meteo_cols = ['timestamp',] + rad_proxy
    meteo = meteo.loc[:,meteo_cols]
    if len(rad_proxy) == 1:    
        meteo.rename(columns={rad_proxy: 'radiation'}, inplace=True)
    elif len(rad_proxy) > 1:
        meteo['radiation'] = meteo[rad_proxy].sum(axis=1)
        meteo.drop(columns=rad_proxy, inplace=True)
    
    start_wy, end_wy = pd.to_datetime(f'{weater_year}-01-01 00:00:00'), pd.to_datetime(f'{weater_year}-12-31 23:00:00')
    meteo = meteo.loc[(meteo['timestamp'] >= start_wy) & (meteo['timestamp'] <= end_wy)]

    meteo['t']= meteo['timestamp'].apply(lambda x: f't_{(x.dayofyear -1) * 24 + x.hour +1}')
    meteo_ts = meteo.copy()


    # # grid premium --------
    # setup 
    if os.path.exists(f'{data_path_def}/output/pvalloc_run/gridprem_ts.parquet'):
        os.remove(f'{data_path_def}/output/pvalloc_run/gridprem_ts.parquet')    

    # import 
    dsonodes_df = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/dsonodes_df.parquet')
    t_range = [f't_{t}' for t in range(1,8760 + 1)]

    dsonodes_df.drop(columns=['EGID'], inplace=True)
    gridprem_ts = pd.DataFrame(np.repeat(dsonodes_df.values, len(t_range), axis=0), columns=dsonodes_df.columns)  
    gridprem_ts['t'] = np.tile(t_range, len(dsonodes_df))
    gridprem_ts['prem_Rp_kWh'] = 0

    gridprem_ts = gridprem_ts[['t', 'grid_node', 'kVA_threshold', 'prem_Rp_kWh']]
    gridprem_ts.drop(columns='kVA_threshold', inplace=True)

    # export 
    gridprem_ts.to_parquet(f'{data_path_def}/output/pvalloc_run/gridprem_ts.parquet')

    

    # EXPORT ----------------------------------------------------------------------------
    ts_names = ['Map_daterange', 'demandtypes_ts', 'meteo_ts', 'gridprem_ts' ]
    ts_list =  [ Map_daterange,   demandtypes_ts,   meteo_ts,   gridprem_ts]
    for i, ts in enumerate(ts_list):
        ts.to_parquet(f'{data_path_def}/output/pvalloc_run/{ts_names[i]}.parquet')


    # RETURN ----------------------------------------------------------------------------
    return ts_list, ts_names



# ------------------------------------------------------------------------------------------------------
# import inst cost estimation func
# ------------------------------------------------------------------------------------------------------
def get_estim_instcost_function(pvalloc_settings):
    data_path_def = pvalloc_settings['data_path']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    log_file_name_def = pvalloc_settings['log_file_name']

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

    return estim_instcost_chfpkW, estim_instcost_chftotal



# ------------------------------------------------------------------------------------------------------
# CONSTRUCTION CAPACITY for pv installations
# ------------------------------------------------------------------------------------------------------
def define_construction_capacity(
        pvalloc_settings, 
        topo_func, 
        df_list_func, df_names_func,
        ts_list_func, ts_names_func):
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
    ts_list, ts_names = ts_list_func, ts_names_func
    df_list, df_names = df_list_func, df_names_func
    print_to_logfile('run function: define_construction_capacity.py', log_file_name_def)


    # create monthly time structure
    T0 = pd.to_datetime(f'{pvalloc_settings["T0_prediction"]}')
    start_loockback = T0 - pd.DateOffset(months=pvalloc_settings['months_lookback']) #+ pd.DateOffset(hours=1)
    end_prediction = T0 + pd.DateOffset(months=pvalloc_settings['months_prediction']) - pd.DateOffset(hours=1)
    month_range = pd.date_range(start=start_loockback, end=end_prediction, freq='ME').to_period('M')
    months_lookback = pd.date_range(start=start_loockback, end=T0, freq='ME').to_period('M')
    months_prediction = pd.date_range(start=(T0 + pd.DateOffset(days=1)), end=end_prediction, freq='ME').to_period('M')
    # checkpoint_to_logfile(f'constr_capacity: month lookback   {months_lookback[0]} to {months_lookback[-1]}', log_file_name_def, 2)
    # checkpoint_to_logfile(f'constr_capacity: month prediction {months_prediction[0]} to {months_prediction[-1]}', log_file_name_def, 2)


    # IMPORT ----------------------------------------------------------------------------
    # Map_daterange = ts_list[0]
    # pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/pv.parquet')
    # Map_egid_pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_pv.parquet')

    Map_daterange = ts_list[ts_names.index('Map_daterange')]
    pv = df_list[df_names.index('pv')]
    Map_egid_pv = df_list[df_names.index('Map_egid_pv')]

    topo_keys = list(topo.keys())

    # subset pv to EGIDs in TOPO, and LOOKBACK period of pvalloc settings
    pv_sub = copy.deepcopy(pv)
    del_cols = ['MainCategory', 'SubCategory', 'PlantCategory']
    pv_sub.drop(columns=del_cols, inplace=True)

    pv_sub = pv_sub.merge(Map_egid_pv, how='left', on='xtf_id')
    pv_sub = pv_sub.loc[pv_sub['EGID'].isin(topo_keys)]
    pv_plot = copy.deepcopy(pv_sub) # used for plotting later

    pv_sub['BeginningOfOperation'] = pd.to_datetime(pv_sub['BeginningOfOperation'])
    pv_sub['MonthPeriod'] = pv_sub['BeginningOfOperation'].dt.to_period('M')
    pv_sub = pv_sub.loc[pv_sub['MonthPeriod'].isin(months_lookback)]

    # plot total power over time
    if True: 
        pv_plot['BeginningOfOperation'] = pd.to_datetime(pv_plot['BeginningOfOperation'])
        pv_plot.set_index('BeginningOfOperation', inplace=True)

        # Resample by week, month, and year and calculate the sum of TotalPower
        weekly_sum = pv_plot['TotalPower'].resample('W').sum()
        monthly_sum = pv_plot['TotalPower'].resample('ME').sum()
        yearly_sum = pv_plot['TotalPower'].resample('YE').sum()

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
        fig.show()
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
        

    # PRINTs to LOGFILE ----------------------------------------------------------------------------
    checkpoint_to_logfile(f'constr_capacity month lookback: {months_lookback[0]} to {months_lookback[-1]}', log_file_name_def, 2)
    checkpoint_to_logfile(f'constr_capacity (sum_TP_kW_lookback) in that period: {sum_TP_kW_lookback} kW', log_file_name_def, 2)

    checkpoint_to_logfile(f'constr_capacity: month prediction {months_prediction[0]} to {months_prediction[-1]}', log_file_name_def, 2)
    checkpoint_to_logfile(f'Of {sum_TP_kW_lookback} kW, {share_to_summer*100}% built in summer months ({summer_months}) and {share_to_winter*100}% in winter months ({winter_months})', log_file_name_def, 2)
    checkpoint_to_logfile(f'sum_TP_kW_lookback (T0: {sum_TP_kW_lookback} kW) increase by {capacity_growth*100}% per year', log_file_name_def, 2)


    # EXPORT ----------------------------------------------------------------------------
    constrcapa.to_parquet(f'{data_path_def}/output/pvalloc_run/constrcapa.parquet')
    constrcapa.to_csv(f'{data_path_def}/output/pvalloc_run/constrcapa.csv', index=False)



    return constrcapa, months_prediction, months_lookback




# ------------------------------------------------------------------------------------------------------
# FAKE TRAFO EGID MAPPING
# ------------------------------------------------------------------------------------------------------
def get_fake_gridnodes_v2(pvalloc_settings):
    
    # import settings + setup -------------------
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    bfs_numbers_def = pvalloc_settings['bfs_numbers']
    gwr_selection_specs_def = pvalloc_settings['gwr_selection_specs']
    print_to_logfile('run function: get_fake_gridnodes_v2', log_file_name_def)


    # create fake gridnodes ----------------------
    solkat = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/solkat.parquet')
    gwr_geo = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gwr_gdf.geojson')
    
    gwr = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/gwr.parquet')
    gwr = gwr.loc[gwr['GGDENR'].isin(bfs_numbers_def)]
    gwr = gwr.drop_duplicates(subset=['EGID'])

    gwr_nodes = gwr.merge(gwr_geo[['EGID', 'geometry']], how='left', on='EGID')
    gwr_nodes = gpd.GeoDataFrame(gwr_nodes, geometry='geometry', crs='EPSG:2056')
    gwr_nodes = gwr_nodes.drop_duplicates(subset=['EGID'])


    # sample random lists for generic node creation ---------------
    node_bldn_ration = 1700 / 42500
    n_nodes = int(gwr.shape[0] * node_bldn_ration)

    state = np.random.get_state()
    np.random.seed(42)    
    gridnode_egid_list = gwr.sample(n_nodes)['EGID'].tolist()
    gridnode_name_list = [f'node{i}' for i in range(1, n_nodes + 1)]
    gridnode_kVA_list  = np.random.choice([160, 320, 640, 800, 960], n_nodes, replace=True)
    np.random.set_state(state)

    # # small scale grid node creation ----------------
    # gridnode_egid_list = ['245020448',  '245017872',    '1368998',  '1369579',  '245059729',    '245054705',    '245014566',    '391554',]
    # gridnode_name_list = ['node1',      'node2',        'node3',    'node4',    'node5',        'node6',        'node7',        'node8',]
    # gridnode_kVA_list =  [100000,       20000,          63000,      100000,      10000,          63000,          100000,         100000, ]

    dsonodes_df = pd.DataFrame({'EGID': gridnode_egid_list, 'grid_node': gridnode_name_list, 'kVA_threshold': gridnode_kVA_list})
    dsonodes_df = dsonodes_df.loc[dsonodes_df['EGID'].isin(gwr_nodes['EGID'].unique())]

    dsonodes_df = dsonodes_df.merge(gwr_nodes[['EGID', 'geometry']], how='left', on='EGID')
    dsonodes_gdf = gpd.GeoDataFrame(dsonodes_df, crs = 'EPSG:2056', geometry='geometry')
    # dsonodes_gdf.to_crs('EPSG:4326', inplace=True) 

    # assign nearest nodes
    def nearest_grid_node(row, nodes_gdf):
        distances = nodes_gdf.geometry.distance(row.geometry)
        nearest_idx = distances.idxmin()  # Find index of minimum distance
        #

        return nodes_gdf.loc[nearest_idx, 'grid_node']

    gwr_nodes['grid_node'] = gwr_nodes.apply(nearest_grid_node, nodes_gdf=dsonodes_gdf, axis=1) 


    # export df ----------
    Map_egid_nodes = gwr_nodes[['EGID', 'grid_node']].copy()
    Map_egid_nodes.to_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_nodes.parquet')
    Map_egid_nodes.to_csv(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_nodes.csv')

    dsonodes_df = dsonodes_df.loc[:,dsonodes_df.columns != 'geometry']
    dsonodes_df.to_parquet(f'{data_path_def}/output/{name_dir_import_def}/dsonodes_df.parquet')
    with open(f'{data_path_def}/output/{name_dir_import_def}/dsonodes_gdf.geojson', 'w') as f:
        f.write(dsonodes_gdf.to_json())





# TO BE DELETED

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
# ANGLE TILT & AZIMUTH Table
# ------------------------------------------------------------------------------------------------------
def get_angle_tilt_table(pvalloc_settings):

    # import settings + setup -------------------
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    print_to_logfile('run function: get_angle_tilt_table', log_file_name_def)

    # SOURCE: table was retreived from this site: https://echtsolar.de/photovoltaik-neigungswinkel/
    # date 29.08.24
    
    # import df ---------
    index_angle = [-180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    index_tilt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    tuples_iter = list(itertools.product(index_angle, index_tilt))

    tuples = [(-180, 0), (-180, 5), (-180, 10), (-180, 15), (-180, 20), (-180, 25), (-180, 30), (-180, 35), (-180, 40), (-180, 45), (-180, 50), (-180, 55), (-180, 60), (-180, 65), (-180, 70), (-180, 75), (-180, 80), (-180, 85), (-180, 90), 
              (-170, 0), (-170, 5), (-170, 10), (-170, 15), (-170, 20), (-170, 25), (-170, 30), (-170, 35), (-170, 40), (-170, 45), (-170, 50), (-170, 55), (-170, 60), (-170, 65), (-170, 70), (-170, 75), (-170, 80), (-170, 85), (-170, 90), 
              (-160, 0), (-160, 5), (-160, 10), (-160, 15), (-160, 20), (-160, 25), (-160, 30), (-160, 35), (-160, 40), (-160, 45), (-160, 50), (-160, 55), (-160, 60), (-160, 65), (-160, 70), (-160, 75), (-160, 80), (-160, 85), (-160, 90), 
              (-150, 0), (-150, 5), (-150, 10), (-150, 15), (-150, 20), (-150, 25), (-150, 30), (-150, 35), (-150, 40), (-150, 45), (-150, 50), (-150, 55), (-150, 60), (-150, 65), (-150, 70), (-150, 75), (-150, 80), (-150, 85), (-150, 90), 
              (-140, 0), (-140, 5), (-140, 10), (-140, 15), (-140, 20), (-140, 25), (-140, 30), (-140, 35), (-140, 40), (-140, 45), (-140, 50), (-140, 55), (-140, 60), (-140, 65), (-140, 70), (-140, 75), (-140, 80), (-140, 85), (-140, 90),
              (-130, 0), (-130, 5), (-130, 10), (-130, 15), (-130, 20), (-130, 25), (-130, 30), (-130, 35), (-130, 40), (-130, 45), (-130, 50), (-130, 55), (-130, 60), (-130, 65), (-130, 70), (-130, 75), (-130, 80), (-130, 85), (-130, 90),
              (-120, 0), (-120, 5), (-120, 10), (-120, 15), (-120, 20), (-120, 25), (-120, 30), (-120, 35), (-120, 40), (-120, 45), (-120, 50), (-120, 55), (-120, 60), (-120, 65), (-120, 70), (-120, 75), (-120, 80), (-120, 85), (-120, 90),
              (-110, 0), (-110, 5), (-110, 10), (-110, 15), (-110, 20), (-110, 25), (-110, 30), (-110, 35), (-110, 40), (-110, 45), (-110, 50), (-110, 55), (-110, 60), (-110, 65), (-110, 70), (-110, 75), (-110, 80), (-110, 85), (-110, 90),
              (-100, 0), (-100, 5), (-100, 10), (-100, 15), (-100, 20), (-100, 25), (-100, 30), (-100, 35), (-100, 40), (-100, 45), (-100, 50), (-100, 55), (-100, 60), (-100, 65), (-100, 70), (-100, 75), (-100, 80), (-100, 85), (-100, 90),
              (-90, 0), (-90, 5), (-90, 10), (-90, 15), (-90, 20), (-90, 25), (-90, 30), (-90, 35), (-90, 40), (-90, 45), (-90, 50), (-90, 55), (-90, 60), (-90, 65), (-90, 70), (-90, 75), (-90, 80), (-90, 85), (-90, 90),
              (-80, 0), (-80, 5), (-80, 10), (-80, 15), (-80, 20), (-80, 25), (-80, 30), (-80, 35), (-80, 40), (-80, 45), (-80, 50), (-80, 55), (-80, 60), (-80, 65), (-80, 70), (-80, 75), (-80, 80), (-80, 85), (-80, 90),
              (-70, 0), (-70, 5), (-70, 10), (-70, 15), (-70, 20), (-70, 25), (-70, 30), (-70, 35), (-70, 40), (-70, 45), (-70, 50), (-70, 55), (-70, 60), (-70, 65), (-70, 70), (-70, 75), (-70, 80), (-70, 85), (-70, 90),
              (-60, 0), (-60, 5), (-60, 10), (-60, 15), (-60, 20), (-60, 25), (-60, 30), (-60, 35), (-60, 40), (-60, 45), (-60, 50), (-60, 55), (-60, 60), (-60, 65), (-60, 70), (-60, 75), (-60, 80), (-60, 85), (-60, 90),
              (-50, 0), (-50, 5), (-50, 10), (-50, 15), (-50, 20), (-50, 25), (-50, 30), (-50, 35), (-50, 40), (-50, 45), (-50, 50), (-50, 55), (-50, 60), (-50, 65), (-50, 70), (-50, 75), (-50, 80), (-50, 85), (-50, 90),
              (-40, 0), (-40, 5), (-40, 10), (-40, 15), (-40, 20), (-40, 25), (-40, 30), (-40, 35), (-40, 40), (-40, 45), (-40, 50), (-40, 55), (-40, 60), (-40, 65), (-40, 70), (-40, 75), (-40, 80), (-40, 85), (-40, 90),
              (-30, 0), (-30, 5), (-30, 10), (-30, 15), (-30, 20), (-30, 25), (-30, 30), (-30, 35), (-30, 40), (-30, 45), (-30, 50), (-30, 55), (-30, 60), (-30, 65), (-30, 70), (-30, 75), (-30, 80), (-30, 85), (-30, 90),
              (-20, 0), (-20, 5), (-20, 10), (-20, 15), (-20, 20), (-20, 25), (-20, 30), (-20, 35), (-20, 40), (-20, 45), (-20, 50), (-20, 55), (-20, 60), (-20, 65), (-20, 70), (-20, 75), (-20, 80), (-20, 85), (-20, 90),
              (-10, 0), (-10, 5), (-10, 10), (-10, 15), (-10, 20), (-10, 25), (-10, 30), (-10, 35), (-10, 40), (-10, 45), (-10, 50), (-10, 55), (-10, 60), (-10, 65), (-10, 70), (-10, 75), (-10, 80), (-10, 85), (-10, 90),
              (0, 0), (0, 5), (0, 10), (0, 15), (0, 20), (0, 25), (0, 30), (0, 35), (0, 40), (0, 45), (0, 50), (0, 55), (0, 60), (0, 65), (0, 70), (0, 75), (0, 80), (0, 85), (0, 90),
              (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55), (10, 60), (10, 65), (10, 70), (10, 75), (10, 80), (10, 85), (10, 90),
              (20, 0), (20, 5), (20, 10), (20, 15), (20, 20), (20, 25), (20, 30), (20, 35), (20, 40), (20, 45), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (20, 75), (20, 80), (20, 85), (20, 90),
              (30, 0), (30, 5), (30, 10), (30, 15), (30, 20), (30, 25), (30, 30), (30, 35), (30, 40), (30, 45), (30, 50), (30, 55), (30, 60), (30, 65), (30, 70), (30, 75), (30, 80), (30, 85), (30, 90),
              (40, 0), (40, 5), (40, 10), (40, 15), (40, 20), (40, 25), (40, 30), (40, 35), (40, 40), (40, 45), (40, 50), (40, 55), (40, 60), (40, 65), (40, 70), (40, 75), (40, 80), (40, 85), (40, 90),
              (50, 0), (50, 5), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30), (50, 35), (50, 40), (50, 45), (50, 50), (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90),
              (60, 0), (60, 5), (60, 10), (60, 15), (60, 20), (60, 25), (60, 30), (60, 35), (60, 40), (60, 45), (60, 50), (60, 55), (60, 60), (60, 65), (60, 70), (60, 75), (60, 80), (60, 85), (60, 90),
              (70, 0), (70, 5), (70, 10), (70, 15), (70, 20), (70, 25), (70, 30), (70, 35), (70, 40), (70, 45), (70, 50), (70, 55), (70, 60), (70, 65), (70, 70), (70, 75), (70, 80), (70, 85), (70, 90),
              (80, 0), (80, 5), (80, 10), (80, 15), (80, 20), (80, 25), (80, 30), (80, 35), (80, 40), (80, 45), (80, 50), (80, 55), (80, 60), (80, 65), (80, 70), (80, 75), (80, 80), (80, 85), (80, 90),
              (90, 0), (90, 5), (90, 10), (90, 15), (90, 20), (90, 25), (90, 30), (90, 35), (90, 40), (90, 45), (90, 50), (90, 55), (90, 60), (90, 65), (90, 70), (90, 75), (90, 80), (90, 85), (90, 90),
              (100, 0), (100, 5), (100, 10), (100, 15), (100, 20), (100, 25), (100, 30), (100, 35), (100, 40), (100, 45), (100, 50), (100, 55), (100, 60), (100, 65), (100, 70), (100, 75), (100, 80), (100, 85), (100, 90),
              (110, 0), (110, 5), (110, 10), (110, 15), (110, 20), (110, 25), (110, 30), (110, 35), (110, 40), (110, 45), (110, 50), (110, 55), (110, 60), (110, 65), (110, 70), (110, 75), (110, 80), (110, 85), (110, 90),
              (120, 0), (120, 5), (120, 10), (120, 15), (120, 20), (120, 25), (120, 30), (120, 35), (120, 40), (120, 45), (120, 50), (120, 55), (120, 60), (120, 65), (120, 70), (120, 75), (120, 80), (120, 85), (120, 90),
              (130, 0), (130, 5), (130, 10), (130, 15), (130, 20), (130, 25), (130, 30), (130, 35), (130, 40), (130, 45), (130, 50), (130, 55), (130, 60), (130, 65), (130, 70), (130, 75), (130, 80), (130, 85), (130, 90),
              (140, 0), (140, 5), (140, 10), (140, 15), (140, 20), (140, 25), (140, 30), (140, 35), (140, 40), (140, 45), (140, 50), (140, 55), (140, 60), (140, 65), (140, 70), (140, 75), (140, 80), (140, 85), (140, 90),
              (150, 0), (150, 5), (150, 10), (150, 15), (150, 20), (150, 25), (150, 30), (150, 35), (150, 40), (150, 45), (150, 50), (150, 55), (150, 60), (150, 65), (150, 70), (150, 75), (150, 80), (150, 85), (150, 90),
              (160, 0), (160, 5), (160, 10), (160, 15), (160, 20), (160, 25), (160, 30), (160, 35), (160, 40), (160, 45), (160, 50), (160, 55), (160, 60), (160, 65), (160, 70), (160, 75), (160, 80), (160, 85), (160, 90),
              (170, 0), (170, 5), (170, 10), (170, 15), (170, 20), (170, 25), (170, 30), (170, 35), (170, 40), (170, 45), (170, 50), (170, 55), (170, 60), (170, 65), (170, 70), (170, 75), (170, 80), (170, 85), (170, 90),
              (180, 0), (180, 5), (180, 10), (180, 15), (180, 20), (180, 25), (180, 30), (180, 35), (180, 40), (180, 45), (180, 50), (180, 55), (180, 60), (180, 65), (180, 70), (180, 75), (180, 80), (180, 85), (180, 90)
              ]
    index = pd.MultiIndex.from_tuples(tuples, names=['angle', 'tilt'])

    values = [89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2, 
              89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
              89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
              89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
              89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
              89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
              89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
              89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
              89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
              89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
              89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
              89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
              89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
              89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
              89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
              89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
              89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
              89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
              89.0, 92.1, 94.7, 96.8, 98.4, 99.5, 100,  100 , 99.5, 98.3, 96.8, 94.8, 92.3, 89.3, 85.8, 81.9, 77.6, 73.1, 68.1,
              89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
              89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
              89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
              89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
              89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
              89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
              89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
              89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
              89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
              89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
              89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
              89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
              89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
              89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
              89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
              89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
              89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
              89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2
              ] 
    
    angle_tilt_df = pd.DataFrame(data = values, index = index, columns = ['efficiency_factor'])
    angle_tilt_df['efficiency_factor'] = angle_tilt_df['efficiency_factor'] / 100

    # export df ----------
    angle_tilt_df.to_parquet(f'{data_path_def}/output/{name_dir_import_def}/angle_tilt_df.parquet')
    angle_tilt_df.to_csv(f'{data_path_def}/output/{name_dir_import_def}/angle_tilt_df.csv')
    return angle_tilt_df


