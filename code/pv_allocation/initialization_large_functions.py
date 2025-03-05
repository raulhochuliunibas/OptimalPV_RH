import sys
import os as os
import numpy as np
import pandas as pd
import json
import plotly.graph_objs as go
import geopandas as gpd
import copy

# own functions 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import  checkpoint_to_logfile, print_to_logfile


"""
    this file contains the functions that are used in the initialization of the PV topology and preparation of 
    model pv allocation process. All smaller "aid functions" are stored in the initialization_small_functions.py file.
"""

# ------------------------------------------------------------------------------------------------------
# IMPORT PREPREPED DATA & CREATE TOPOLOGY
# ------------------------------------------------------------------------------------------------------
def import_prepre_AND_create_topology(
        scen, ):
    """ 
    Reimport all the data that is intermediatley stored in the prepreped folder. Transform it such that it can be used throughout the
    PV allocation algorithm. Return all the data objects back to the main code file. 
    - Input: PV allocation settings
    - Output: all the data objects that are needed for later calculations
    """

    # import settings + setup -------------------
    print_to_logfile('run function: import_prepreped_data', scen.log_name)



    # IMPORT & TRANSFORM ============================================================================
    # Import all necessary data objects from prepreped folder and transform them for later calculations
    print_to_logfile('import & transform data', scen.log_name)
    if True:
            
        # GWR -------
        gwr_gdf = gpd.read_file(f'{scen.name_dir_import_path}/gwr_gdf.geojson')
        gwr = pd.read_parquet(f'{scen.name_dir_import_path}/gwr.parquet')

        gwr['EGID'] = gwr['EGID'].astype(str)
        gwr.loc[gwr['GBAUJ'] == '', 'GBAUJ'] = 0  # transform GBAUJ to apply filter and transform back
        gwr['GBAUJ'] = gwr['GBAUJ'].astype(int)
        # filtering for scen.GWR_specs
        gwr = gwr.loc[(gwr['GSTAT'].isin(scen.GWRspec_GSTAT)) &
                    (gwr['GKLAS'].isin(scen.GWRspec_GKLAS)) &
                    (gwr['GBAUJ'] >= scen.GWRspec_GBAUJ_minmax[0]) &
                    (gwr['GBAUJ'] <= scen.GWRspec_GBAUJ_minmax[1])]
        gwr['GBAUJ'] = gwr['GBAUJ'].astype(str)
        gwr.loc[gwr['GBAUJ'] == '0', 'GBAUJ'] = ''
        # because not all buldings have dwelling information, need to remove dwelling columns and rows again (remove duplicates where 1 building had multiple dwellings)
        if scen.GWRspec_dwelling_cols == []:
            gwr = copy.deepcopy(gwr.loc[:, scen.GWRspec_building_cols])
            gwr = gwr.drop_duplicates(subset=['EGID'])
        gwr = gwr.loc[gwr['GGDENR'].isin(scen.bfs_numbers)]
        gwr = copy.deepcopy(gwr)


        # SOLKAT -------
        solkat = pd.read_parquet(f'{scen.name_dir_import_path}/solkat.parquet')

        solkat['EGID'] = solkat['EGID'].fillna('').astype(str)
        solkat['DF_UID'] = solkat['DF_UID'].fillna('').astype(str)
        solkat['DF_NUMMER'] = solkat['DF_NUMMER'].fillna('').astype(str)
        solkat['SB_UUID'] = solkat['SB_UUID'].fillna('').astype(str)
        solkat['FLAECHE'] = solkat['FLAECHE'].fillna(0).astype(float)
        solkat['STROMERTRAG'] = solkat['STROMERTRAG'].fillna(0).astype(float)
        solkat['AUSRICHTUNG'] = solkat['AUSRICHTUNG'].astype(int)
        solkat['NEIGUNG'] = solkat['NEIGUNG'].astype(int)

        # remove building with maximal (outlier large) number of partitions => complicates the creation of partition combinations
        solkat['EGID'].value_counts()
        egid_counts = solkat['EGID'].value_counts()
        egids_below_max = list(egid_counts[egid_counts < scen.GWRspec_solkat_max_n_partitions].index)
        solkat = solkat.loc[solkat['EGID'].isin(egids_below_max)]

        # remove buildings with a certain roof surface because they are too large to be residential houses
        solkat_area_per_EGID_range = scen.GWRspec_solkat_area_per_EGID_range
        if solkat_area_per_EGID_range != []:
            solkat_agg_FLAECH = solkat.groupby('EGID')['FLAECHE'].sum()
            solkat = solkat.merge(solkat_agg_FLAECH, how='left', on='EGID', suffixes=('', '_sum'))
            solkat = solkat.rename(columns={'FLAECHE_sum': 'FLAECHE_total'})
            solkat = solkat.loc[(solkat['FLAECHE_total'] >= solkat_area_per_EGID_range[0]) & 
                                (solkat['FLAECHE_total'] < solkat_area_per_EGID_range[1])]
            solkat.drop(columns='FLAECHE_total', inplace=True)

        solkat = solkat.loc[solkat['BFS_NUMMER'].isin(scen.bfs_numbers)]
        solkat = copy.deepcopy(solkat)


        # SOLKAT MONTH -------
        solkat_month = pd.read_parquet(f'{scen.name_dir_import_path}/solkat_month.parquet')
        solkat_month['DF_UID'] = solkat_month['DF_UID'].fillna('').astype(str)


        # PV -------
        pv = pd.read_parquet(f'{scen.name_dir_import_path}/pv.parquet')
        pv['xtf_id'] = pv['xtf_id'].fillna(0).astype(int).replace(0, '').astype(str)    
        pv['TotalPower'] = pv['TotalPower'].fillna(0).astype(float)

        pv['BeginningOfOperation'] = pd.to_datetime(pv['BeginningOfOperation'], format='%Y-%m-%d', errors='coerce')
        gbauj_range = [pd.to_datetime(f'{scen.GWRspec_GBAUJ_minmax[0]}-01-01'), 
                    pd.to_datetime(f'{scen.GWRspec_GBAUJ_minmax[1]}-12-31')]
        pv = pv.loc[(pv['BeginningOfOperation'] >= gbauj_range[0]) & (pv['BeginningOfOperation'] <= gbauj_range[1])]
        pv['BeginningOfOperation'] = pv['BeginningOfOperation'].dt.strftime('%Y-%m-%d')

        pv = pv.loc[pv["BFS_NUMMER"].isin(scen.bfs_numbers)]
        pv = pv.copy()


        # PV TARIF -------
        pvtarif_year = scen.TECspec_pvtarif_year
        pvtarif_col =  scen.TECspec_pvtarif_col
        
        Map_gm_ewr = pd.read_parquet(f'{scen.name_dir_import_path}/Map_gm_ewr.parquet')
        pvtarif = pd.read_parquet(f'{scen.name_dir_import_path}/pvtarif.parquet')
        pvtarif = pvtarif.merge(Map_gm_ewr, how='left', on='nrElcom')

        pvtarif['bfs'] = pvtarif['bfs'].astype(str)
        # pvtarif[pvtarif_col] = pvtarif[pvtarif_col].fillna(0).astype(float)
        pvtarif[pvtarif_col] = pvtarif[pvtarif_col].replace('', 0).astype(float)

        # transformation
        pvtarif = pvtarif.loc[(pvtarif['year'] == str(pvtarif_year)[2:4]) & 
                            (pvtarif['bfs'].isin((scen.bfs_numbers)))]

        empty_cols = [col for col in pvtarif.columns if pvtarif[col].isna().all()]
        pvtarif = pvtarif.drop(columns=empty_cols)

        select_cols = ['nrElcom', 'nomEw', 'year', 'bfs', 'idofs'] + pvtarif_col
        pvtarif = pvtarif[select_cols].copy()


        # ELECTRICITY PRICE -------
        elecpri = pd.read_parquet(f'{scen.name_dir_import_path}/elecpri.parquet')
        elecpri['bfs_number'] = elecpri['bfs_number'].astype(str)


        # Map solkat_egid > pv -------
        Map_egid_pv = pd.read_parquet(f'{scen.name_dir_import_path}/Map_egid_pv.parquet')
        Map_egid_pv = Map_egid_pv.dropna()
        Map_egid_pv['EGID'] = Map_egid_pv['EGID'].astype(int).astype(str)
        Map_egid_pv['xtf_id'] = Map_egid_pv['xtf_id'].fillna('').astype(int).astype(str)


        # Map demandtypes > egid -------
        with open(f'{scen.name_dir_import_path}/Map_demandtype_EGID.json', 'r') as file:
            Map_demandtypes_egid = json.load(file)


        # Map egid > demandtypes -------
        with open(f'{scen.name_dir_import_path}/Map_EGID_demandtypes.json', 'r') as file:
            Map_egid_demandtypes = json.load(file)


        # Map egid > node -------
        Map_egid_dsonode = pd.read_parquet(f'{scen.name_dir_import_path}/Map_egid_dsonode.parquet')
        Map_egid_dsonode['EGID'] = Map_egid_dsonode['EGID'].astype(str)
        Map_egid_dsonode['grid_node'] = Map_egid_dsonode['grid_node'].astype(str)
        Map_egid_dsonode.index = Map_egid_dsonode['EGID']


        # dsonodes data -------
        dsonodes_df  = pd.read_parquet(f'{scen.name_dir_import_path}/dsonodes_df.parquet')
        dsonodes_gdf = gpd.read_file(f'{scen.name_dir_import_path}/dsonodes_gdf.geojson')


        # angle_tilt_df -------
        angle_tilt_df = pd.read_parquet(f'{scen.name_dir_import_path}/angle_tilt_df.parquet')


        # PV Cost functions --------
        """
            # Define the interpolation functions using the imported coefficients
            def func_chf_pkW(x, a, b):
                return a + b / x

            estim_instcost_chfpkW = lambda x: func_chf_pkW(x, *params_pkW)

            def func_chf_total_poly(x, coefs_total):
                return sum(c * x**i for i, c in enumerate(coefs_total))

            estim_instcost_chftotal = lambda x: func_chf_total_poly(x, coefs_total)
        """


    # EGID SELECTION / EXCLUSION ============================================================================
    # check how many of gwr's EGIDs are in solkat and pv
    len(np.intersect1d(gwr['EGID'].unique(), solkat['EGID'].unique()))
    len(np.intersect1d(gwr['EGID'].unique(), Map_egid_pv['EGID'].unique()))


    # gwr/solkat mismatch ----------
    # throw out all EGIDs of GWR that are not in solkat
    # >  NOTE: this could be troublesome :/ check in QGIS if large share of buildings are missing.  
    gwr_before_solkat_selection = copy.deepcopy(gwr)
    gwr = copy.deepcopy(gwr.loc[gwr['EGID'].isin(solkat['EGID'].unique())])


    # gwr/Map_egid_dsonodes mismatch ----------
    # > Case 1 EGID in Map but not in GWR: => drop EGID; no problem, connection to a house that is not in sample; will happen automatically when creating topology on gwr EGIDs
    # > Case 2 EGID in GWR but not in Map: more problematic; EGID "close" to next node => Match to nearest node; EGID "far away" => drop EGID
    gwr_wo_node = gwr.loc[~gwr['EGID'].isin(Map_egid_dsonode['EGID'].unique()),]
    Map_egid_dsonode_appendings =[]
        
    for egid in gwr_wo_node['EGID']:
        egid_point = gwr_gdf.loc[gwr_gdf['EGID'] == egid, 'geometry'].iloc[0]
        dsonodes_gdf['distances'] = dsonodes_gdf['geometry'].distance(egid_point)
        min_idx = dsonodes_gdf['distances'].idxmin()
        min_dist = dsonodes_gdf['distances'].min()
        
        if min_dist < scen.TECspec_max_distance_m_for_EGID_node_matching:
            Map_egid_dsonode_appendings.append([egid, dsonodes_gdf.loc[min_idx, 'grid_node'], dsonodes_gdf.loc[min_idx, 'kVA_threshold']])
    
    Map_appendings_df = pd.DataFrame(Map_egid_dsonode_appendings, columns=['EGID', 'grid_node', 'kVA_threshold'])
    Map_egid_dsonode = pd.concat([Map_egid_dsonode, Map_appendings_df], axis=0)

    gwr_before_dsonode_selection = copy.deepcopy(gwr)
    gwr = copy.deepcopy(gwr.loc[gwr['EGID'].isin(Map_egid_dsonode['EGID'].unique())])
        

    # summary prints ----------
    print_to_logfile('\nEGID selection for TOPOLOGY:', scen.summary_name)
    checkpoint_to_logfile('Loop for topology creation over GWR EGIDs', scen.summary_name, 5, True)
    checkpoint_to_logfile('In Total: {gwr["EGID"].nunique()} gwrEGIDs ({round(gwr["EGID"].nunique() / gwr_before_solkat_selection["EGID"].nunique() * 100,1)}% of {gwr_before_solkat_selection["EGID"].nunique()} total gwrEGIDs) are used for topology creation', scen.summary_name, 3, True)
    checkpoint_to_logfile('  The rest drops out because gwrEGID not present in all data sources', scen.summary_name, 3, True)
    
    subtraction1 = gwr_before_solkat_selection["EGID"].nunique() - gwr_before_dsonode_selection["EGID"].nunique()
    checkpoint_to_logfile(f'  > {subtraction1} ({round(subtraction1 / gwr_before_solkat_selection["EGID"].nunique()*100,1)} % ) gwrEGIDs missing in solkat', scen.summary_name, 5, True)
    
    subtraction2 = gwr_before_dsonode_selection["EGID"].nunique() - gwr["EGID"].nunique()
    checkpoint_to_logfile(f'  > {subtraction2} ({round(subtraction2 / gwr_before_dsonode_selection["EGID"].nunique()*100,1)} % ) gwrEGIDs missing in dsonodes', scen.summary_name, 5, True)
    if Map_appendings_df.shape[0] > 0:

        checkpoint_to_logfile(f'  > (REMARK: Even matched {Map_appendings_df.shape[0]} EGIDs matched artificially to gridnode, because EGID lies in close node range, max_distance_m_for_EGID_node_matching: {scen.TECspec_max_distance_m_for_EGID_node_matching} meters', scen.summary_name, 3, True)
    elif Map_appendings_df.shape[0] == 0:
        checkpoint_to_logfile(f'  > (REMARK: No EGIDs matched to nearest gridnode, max_distance_m_for_EGID_node_matching: {scen.TECspec_max_distance_m_for_EGID_node_matching} meters', scen.summary_name, 3, True)



    # CREATE TOPOLOGY ============================================================================
    print_to_logfile('start creating topology - Taking EGIDs from GWR', scen.log_name)
    log_str1 = f'Of {gwr["EGID"].nunique()} gwrEGIDs, {len(np.intersect1d(gwr["EGID"].unique(), solkat["EGID"].unique()))} covered by solkatEGIDs ({round(len(np.intersect1d(gwr["EGID"].unique(), solkat["EGID"].unique()))/gwr["EGID"].nunique()*100,2)} % covered)'
    log_str2 = f'Solkat specs (WTIH assigned EGID): {solkat.loc[solkat["EGID"] !="", "SB_UUID"].nunique()} of {solkat.loc[:, "SB_UUID"].nunique()} ({round((solkat.loc[solkat["EGID"] !="", "SB_UUID"].nunique() / solkat.loc[:, "SB_UUID"].nunique())*100,2)} %); {solkat.loc[solkat["EGID"] !="", "DF_UID"].nunique()} of {solkat.loc[:, "DF_UID"].nunique()} ({round((solkat.loc[solkat["EGID"] !="", "DF_UID"].nunique() / solkat.loc[:, "DF_UID"].nunique())*100,2)} %)'
    checkpoint_to_logfile(log_str1, scen.log_name)
    checkpoint_to_logfile(log_str2, scen.log_name)


    # start loop ------------------------------------------------
    topo_egid = {}
    modulus_print = int(len(gwr['EGID'])//5)
    CHECK_egid_with_problems = []
    print_to_logfile('\n', scen.log_name)
    checkpoint_to_logfile('start attach to topo', scen.log_name, 1 , True)

    # transform to np.array for faster lookups
    pv_npry, gwr_npry, elecpri_npry = np.array(pv), np.array(gwr), np.array(elecpri) 



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
                checkpoint_to_logfile(f'---- pv xtf_id {xtfid} in Mapping_egid_pv, but NOT in pv data', scen.log_name, 3, False)
                
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
                checkpoint_to_logfile(f'ERROR: multiple xtf_ids for EGID: {egid}', scen.log_name, 3, scen.show_debug_prints)
                CHECK_egid_with_problems.append((egid, 'multiple xtf_ids'))


        # add solkat data --------
        if egid in solkat['EGID'].unique():
            solkat_sub = solkat.loc[solkat['EGID'] == egid]
            if solkat.duplicated(subset=['DF_UID', 'EGID']).any():
                solkat_sub = solkat_sub.drop_duplicates(subset=['DF_UID', 'EGID'])
            solkat_partitions = solkat_sub.set_index('DF_UID')[['FLAECHE', 'STROMERTRAG', 'AUSRICHTUNG', 'NEIGUNG']].to_dict(orient='index')                   
        
        elif egid not in solkat['EGID'].unique():
            solkat_partitions = {}
            checkpoint_to_logfile(f'egid {egid} not in solkat', scen.log_name, 3, scen.show_debug_prints)


        # add demand type --------
        if egid in Map_egid_demandtypes.keys():
            demand_type = Map_egid_demandtypes[egid]
        elif egid not in Map_egid_demandtypes.keys():
            print_to_logfile(f'\n ** ERROR ** EGID {egid} not in Map_egid_demandtypes, but must be because Map file is based on GWR', scen.log_name)
            demand_type = 'NA'
            CHECK_egid_with_problems.append((egid, 'not in Map_egid_demandtypes (both based on GWR)'))

        # add pvtarif --------
        bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0]
        pvtarif_egid = sum([pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), col].sum() for col in pvtarif_col])

        pvtarif_sub = pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid)]
        if pvtarif_sub.empty:
            checkpoint_to_logfile(f'ERROR: no pvtarif data for EGID {egid}', scen.log_name, 3, scen.show_debug_prints)
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
        
            # checkpoint_to_logfile(f'multiple pvtarif data for EGID {egid}', scen.log_name, 3, scen.show_debug_prints)
            CHECK_egid_with_problems.append((egid, 'multiple pvtarif data'))


        # add elecpri --------
        elecpri_egid = {}
        elecpri_info = {}

        mask_bfs = np.isin(elecpri_npry[:, elecpri.columns.get_loc('bfs_number')], [bfs_of_egid,]) 
        mask_year = np.isin(elecpri_npry[:, elecpri.columns.get_loc('year')],    scen.TECspec_elecpri_year)
        mask_cat = np.isin(elecpri_npry[:, elecpri.columns.get_loc('category')], scen.TECspec_elecpri_category)

        if sum(mask_bfs & mask_year & mask_cat) < 1:
            checkpoint_to_logfile(f'ERROR: no elecpri data for EGID {egid}', scen.log_name, 3, scen.show_debug_prints)
            CHECK_egid_with_problems.append((egid, 'no elecpri data'))
        elif sum(mask_bfs & mask_year & mask_cat) > 1:
            checkpoint_to_logfile(f'ERROR: multiple elecpri data for EGID {egid}', scen.log_name, 3, scen.show_debug_prints)
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
            if isinstance(Map_egid_dsonode.loc[egid, 'grid_node'], str):
                grid_node = Map_egid_dsonode.loc[egid, 'grid_node']
            elif isinstance(Map_egid_dsonode.loc[egid, 'grid_node'], pd.Series):
                grid_node = Map_egid_dsonode.loc[egid, 'grid_node'].iloc[0]
                

        # attach to topo --------
        # topo['EGID'][egid] = {
        topo_egid[egid] = {
            'gwr_info': gwr_info,
            'grid_node': grid_node,
            'pv_inst': pv_inst,
            'solkat_partitions': solkat_partitions, 
            'demand_type': demand_type,
            'pvtarif_Rp_kWh': pvtarif_egid, 
            'EWR': ewr_info, 
            'elecpri_Rp_kWh': elecpri_egid,
            'elecpri_info': elecpri_info,
            }  

        # Checkpoint prints
        if i % modulus_print == 0:
            print_to_logfile(f'\t -- EGID {i} of {len(gwr["EGID"])} {15*"-"}', scen.log_name)

        
    # end loop ------------------------------------------------
    checkpoint_to_logfile('end attach to topo', scen.log_name, 1 , True)
    print_to_logfile('\nsanity check for installtions in topo_egid', scen.summary_name)
    checkpoint_to_logfile(f'number of EGIDs with multiple installations: {CHECK_egid_with_problems.count("multiple xtf_ids")}', scen.summary_name)


    # EXPORT TOPO + Mappings ============================================================================

    with open(f'{scen.name_dir_export_path}/topo_egid.txt', 'w') as f:
        f.write(str(topo_egid))
    
    with open(f'{scen.name_dir_export_path}/topo_egid.json', 'w') as f:
        json.dump(topo_egid, f)

    # Export CHECK_egid_with_problems to txt file for trouble shooting
    with open(f'{scen.name_dir_export_path}/CHECK_egid_with_problems.txt', 'w') as f:
        f.write(f'\n ** EGID with problems: {len(CHECK_egid_with_problems)} **\n\n')
        f.write(str(CHECK_egid_with_problems))

    CHECK_egid_with_problems_dict = {egid: problem for egid, problem in CHECK_egid_with_problems}
    with open(f'{scen.name_dir_export_path}/CHECK_egid_with_problems.json', 'w') as f:
        json.dump(CHECK_egid_with_problems_dict, f)


    # EXPORT ============================================================================
    # pvalloc_run folder gets crowded, > only keep the most important files
    df_names = ['Map_egid_pv', 'solkat_month', 'pv', 'pvtarif', 'elecpri', 'Map_egid_dsonode', 'dsonodes_df', 'dsonodes_gdf', 'angle_tilt_df', ]
    df_list =  [ Map_egid_pv,   solkat_month,   pv,   pvtarif,   elecpri,   Map_egid_dsonode,   dsonodes_df,   dsonodes_gdf,   angle_tilt_df, ]
    for i, m in enumerate(df_list): 
        if isinstance(m, pd.DataFrame):
            m.to_parquet(f'{scen.name_dir_export_path}/{df_names[i]}.parquet')
        elif isinstance(m, dict):
            with open(f'{scen.name_dir_export_path}/{df_names[i]}.json', 'w') as f:
                json.dump(m, f)        
        elif isinstance(m, gpd.GeoDataFrame):
            m.to_file(f'{scen.name_dir_export_path}/{df_names[i]}.geojson', driver='GeoJSON')


    # RETURN OBJECTS ============================================================================
    return topo_egid, df_list, df_names



# ------------------------------------------------------------------------------------------------------
# IMPORT TS DATA
# ------------------------------------------------------------------------------------------------------
def import_ts_data(
        scen, ):
    """
    Import the time series data that is needed for the PV allocation algorithm.
    - Input: PV allocation settings
    - Output: all the time series data objects that are needed for later calculations
    """

    # import settings + setup -------------------
    print_to_logfile('run function: import_ts_data', scen.log_name)


    # create time structure for TS
    T0 = pd.to_datetime(f'{scen.T0_prediction}')
    start_loockback = T0 - pd.DateOffset(months = scen.months_lookback) # + pd.DateOffset(hours=1)
    end_prediction = T0 + pd.DateOffset(months = scen.months_prediction) - pd.DateOffset(hours=1)
    date_range = pd.date_range(start=start_loockback, end=end_prediction, freq='h')
    checkpoint_to_logfile(f'import TS: lookback range   {start_loockback} to {T0-pd.DateOffset(hours=1)}', scen.log_name, 2)
    checkpoint_to_logfile(f'import TS: prediction range {T0} to {end_prediction}', scen.log_name, 2)

    Map_daterange = pd.DataFrame({'date_range': date_range, 'DoY': date_range.dayofyear, 'hour': date_range.hour})
    Map_daterange['HoY'] = (Map_daterange['DoY'] - 1) * 24 + (Map_daterange['hour']+1)
    Map_daterange['t'] = Map_daterange['HoY'].apply(lambda x: f't_{x}')


    # IMPORT ----------------------------------------------------------------------------

    # demand types --------
    demandtypes_tformat = pd.read_parquet(f'{scen.name_dir_import_path}/demandtypes.parquet')
    demandtypes_ts = demandtypes_tformat.copy()

    nas =   sum([demandtypes_ts[col].isna().sum() for col in demandtypes_ts.columns])
    nulls = sum([demandtypes_ts[col].isnull().sum() for col in demandtypes_ts.columns])
    checkpoint_to_logfile(f'sanity check demand_ts: {nas} NaNs or {nulls} Nulls for any column in df', scen.log_name)


    # meteo (radiation & temperature) --------
    meteo_col_dir_radiation =  scen.WEAspec_meteo_col_dir_radiation
    meteo_col_diff_radiation = scen.WEAspec_meteo_col_diff_radiation
    meteo_col_temperature =    scen.WEAspec_meteo_col_temperature
    weater_year =              scen.WEAspec_weater_year

    meteo = pd.read_parquet(f'{scen.name_dir_import_path}/meteo.parquet')
    meteo_cols = ['timestamp', meteo_col_dir_radiation, meteo_col_diff_radiation, meteo_col_temperature]
    meteo = meteo.loc[:,meteo_cols]

    # get radiation
    meteo['rad_direct'] = meteo[meteo_col_dir_radiation]
    meteo['rad_diffuse'] = meteo[meteo_col_diff_radiation]
    meteo.drop(columns=[meteo_col_dir_radiation, meteo_col_diff_radiation], inplace=True)

    # get temperature
    meteo['temperature'] = meteo[meteo_col_temperature]
    meteo.drop(columns=meteo_col_temperature, inplace=True)

    start_wy, end_wy = pd.to_datetime(f'{weater_year}-01-01 00:00:00'), pd.to_datetime(f'{weater_year}-12-31 23:00:00')
    meteo = meteo.loc[(meteo['timestamp'] >= start_wy) & (meteo['timestamp'] <= end_wy)]

    meteo['t']= meteo['timestamp'].apply(lambda x: f't_{(x.dayofyear -1) * 24 + x.hour +1}')
    meteo_ts = meteo.copy()



    # # grid premium --------
    # setup 
    if os.path.exists(f'{scen.name_dir_export_path}/gridprem_ts.parquet'):
        os.remove(f'{scen.name_dir_export_path}/gridprem_ts.parquet')    

    # import 
    dsonodes_df = pd.read_parquet(f'{scen.name_dir_import_path}/dsonodes_df.parquet')
    t_range = [f't_{t}' for t in range(1,8760 + 1)]

    gridprem_ts = pd.DataFrame(np.repeat(dsonodes_df.values, len(t_range), axis=0), columns=dsonodes_df.columns)  
    gridprem_ts['t'] = np.tile(t_range, len(dsonodes_df))
    gridprem_ts['prem_Rp_kWh'] = 0

    gridprem_ts = gridprem_ts[['t', 'grid_node', 'kVA_threshold', 'prem_Rp_kWh']]
    gridprem_ts.drop(columns='kVA_threshold', inplace=True)

    # export 
    gridprem_ts.to_parquet(f'{scen.name_dir_export_path}/gridprem_ts.parquet')

    

    # EXPORT ----------------------------------------------------------------------------
    ts_names = ['Map_daterange', 'demandtypes_ts', 'meteo_ts', 'gridprem_ts' ]
    ts_list =  [ Map_daterange,   demandtypes_ts,   meteo_ts,   gridprem_ts]
    for i, ts in enumerate(ts_list):
        ts.to_parquet(f'{scen.name_dir_export_path}/{ts_names[i]}.parquet')


    # RETURN ----------------------------------------------------------------------------
    return ts_list, ts_names



# ------------------------------------------------------------------------------------------------------
# CONSTRUCTION CAPACITY for pv installations
# ------------------------------------------------------------------------------------------------------
def define_construction_capacity(
        scen, 
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
    topo = topo_func
    ts_list, ts_names = ts_list_func, ts_names_func
    df_list, df_names = df_list_func, df_names_func
    print_to_logfile('run function: define_construction_capacity.py', scen.log_name)


    # create monthly time structure
    T0 = pd.to_datetime(f'{scen.T0_prediction}')
    start_loockback = T0 - pd.DateOffset(months=scen.months_lookback) #+ pd.DateOffset(hours=1)
    end_prediction = T0 + pd.DateOffset(months=scen.months_prediction) - pd.DateOffset(hours=1)
    months_lookback = pd.date_range(start=start_loockback, end=T0, freq='ME').to_period('M')
    months_prediction = pd.date_range(start=(T0 + pd.DateOffset(days=1)), end=end_prediction, freq='ME').to_period('M')


    # IMPORT ----------------------------------------------------------------------------
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
        # fig.show()
        fig.write_html(f'{scen.name_dir_export_path}/pv_total_power_over_time.html')


    # CAPACITY ASSIGNMENT ----------------------------------------------------------------------------
    capacity_growth = scen.CSTRspec_ann_capacity_growth
    month_constr_capa_tuples = scen.CSTRspec_month_constr_capa_tuples

    sum_TP_kW_lookback = pv_sub['TotalPower'].sum()

    constrcapa = pd.DataFrame({'date': months_prediction, 'year': months_prediction.year, 'month': months_prediction.month})
    years_prediction = months_prediction.year.unique()
    i, y = 0, years_prediction[0]
    for i,y in enumerate(years_prediction):

        TP_y = sum_TP_kW_lookback * (1 + capacity_growth)**(i+1)
        for m, TP_m in month_constr_capa_tuples:
            constrcapa.loc[(constrcapa['year'] == y) & 
                           (constrcapa['month'] == m), 'constr_capacity_kw'] = TP_y * TP_m
        
    months_prediction_df = pd.DataFrame({'date': months_prediction, 'year': months_prediction.year, 'month': months_prediction.month})

    # PRINTs to LOGFILE ----------------------------------------------------------------------------
    checkpoint_to_logfile(f'constr_capacity month lookback, between :                {months_lookback[0]} to {months_lookback[-1]}', scen.log_name, 2)
    checkpoint_to_logfile(f'constr_capacity KW built in period (sum_TP_kW_lookback): {round(sum_TP_kW_lookback,2)} kW', scen.log_name, 2)
    print_to_logfile('\n', scen.log_name)
    checkpoint_to_logfile(f'constr_capacity: month prediction {months_prediction[0]} to {months_prediction[-1]}', scen.log_name, 2)
    checkpoint_to_logfile(f'sum_TP_kw_lookback {round(sum_TP_kW_lookback,3)} kW to distribute across months_prediction', scen.log_name, 2)
    print_to_logfile('\n', scen.log_name)
    checkpoint_to_logfile(f'sum_TP_kW_lookback (T0: {round(sum_TP_kW_lookback,2)} kW) increase by {capacity_growth*100}% per year', scen.log_name, 2)


    # EXPORT ----------------------------------------------------------------------------
    constrcapa.to_parquet(f'{scen.name_dir_export_path}/constrcapa.parquet')
    constrcapa.to_csv(f'{scen.name_dir_export_path}/constrcapa.csv', index=False)

    months_prediction_df.to_parquet(f'{scen.name_dir_export_path}/months_prediction.parquet')
    months_prediction_df.to_csv(f'{scen.name_dir_export_path}/months_prediction.csv', index=False)

    return constrcapa, months_prediction, months_lookback






# NOT IN USE ANYMORE ********************************************************************************************************************


# ------------------------------------------------------------------------------------------------------
# import existing topology
# ------------------------------------------------------------------------------------------------------
def import_exisitng_topology(
        scen,
        df_search_names ):
    # name_dir_export_def = pvalloc_settings['name_dir_export']
    # data_path = pvalloc_settings['data_path']
    # scen.log_name = pvalloc_settings['scen.log_name']
    # interim_path_def = pvalloc_settings['interim_path']

    print_to_logfile('run function: import_existing_topology', scen.log_name)

    # import existing topo & Mappings ---------
    # topo = json.load(open(f'{interim_path_def}/topo_egid.json', 'r'))
        
    # df_names = df_search_names
    # df_list = []
    # for m in df_names:
    #     f = glob.glob(f'{interim_path_def}/{m}.*')
    #     if len(f) == 1:
    #         if f[0].split('.')[1] == 'json':
    #             df_list.append(json.load(open(f[0], 'r')))
    #         elif f[0].split('.')[1] == 'parquet':
    #             df_list.append(pd.read_parquet(f[0]))
    
    # return topo, df_list, df_names


