import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import json
import plotly.express as px
import copy

from shapely.ops import unary_union

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr
from api_keys.api_keys import get_primeo_path


# ------------------------------------------------------------------------------------------------------
# BY SB_UUID - FALSE - IMPORT LOCAL DATA + create SPATIAL MAPPINGS
# ------------------------------------------------------------------------------------------------------
def get_earlier_api_import_data(scen,):
    """
    Function to import all api input data, previously downloaded and stored through various API calls
    """
    # SETUP --------------------------------------
    print_to_logfile('run function: get_earlier_api_import_data.py', scen.log_name)

    # IMPORT + STORE DATA in preprep folder --------------------------------------
    # Map_gm_ewr
    Map_gm_ewr = pd.read_parquet(f'{scen.data_path}/input_api/Map_gm_ewr.parquet')
    Map_gm_ewr.to_parquet(f'{scen.preprep_path}/Map_gm_ewr.parquet')
    Map_gm_ewr.to_csv(f'{scen.preprep_path}/Map_gm_ewr.csv', sep=';', index=False)
    checkpoint_to_logfile('Map_gm_ewr stored in prepreped data', scen.log_name, 2, scen.show_debug_prints)
    
    # pvtarif
    pvtarif_all = pd.read_parquet(f'{scen.data_path}/input_api/pvtarif.parquet')
    year_range_2int = [str(year % 100).zfill(2) for year in range(scen.year_range[0], scen.year_range[1]+1)]
    pvtarif = copy.deepcopy(pvtarif_all.loc[pvtarif_all['year'].isin(year_range_2int), :])
    pvtarif.to_parquet(f'{scen.preprep_path}/pvtarif.parquet')
    pvtarif.to_csv(f'{scen.preprep_path}/pvtarif.csv', sep=';', index=False)
    checkpoint_to_logfile('pvtarif stored in prepreped data', scen.log_name, 2, scen.show_debug_prints)
        

# ------------------------------------------------------------------------------------------------------
# BY SB_UUID - FALSE - IMPORT LOCAL DATA + create SPATIAL MAPPINGS
# ------------------------------------------------------------------------------------------------------
def local_data_AND_spatial_mappings(scen, ):
    """
    Function to import all the local data sources, remove and transform data where necessary and store only
    the required data that is in range with the BFS municipality selection. When applicable, create mapping
    files, so that so that different data sets can be matched and spatial data can be reidentified to their 
    geometry if necessary. 
    """

    # SETUP --------------------------------------
    print_to_logfile('run function: local_data_AND_spatial_mappings.py', scen.log_name)

    # IMPORT DATA ---------------------------------------------------------------------------------
    gm_shp_gdf = gpd.read_file(f'{scen.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    

    # PV ====================
    pv_all_pq = pd.read_parquet(f'{scen.data_path}/input_split_data_geometry/pv_pq.parquet')
    checkpoint_to_logfile(f'import pv_pq, {pv_all_pq.shape[0]} rows', scen.log_name, 2, scen.show_debug_prints)
    pv_all_geo = gpd.read_file(f'{scen.data_path}/input_split_data_geometry/pv_geo.geojson')
    checkpoint_to_logfile(f'import pv_geo, {pv_all_geo.shape[0]} rows', scen.log_name, 2, scen.show_debug_prints)

    # transformations
    pv_all_pq['xtf_id'] = pv_all_pq['xtf_id'].astype(str)
    pv_all_geo['xtf_id'] = pv_all_geo['xtf_id'].astype(str)

    pv = pv_all_pq[pv_all_pq['BFS_NUMMER'].isin(scen.bfs_numbers)]  # select and export pq for BFS numbers
    pv_wgeo = pv.merge(pv_all_geo[['xtf_id', 'geometry']], how = 'left', on = 'xtf_id') # merge geometry for later use
    pv_gdf = gpd.GeoDataFrame(pv_wgeo, geometry='geometry')


    # GWR ====================
    gwr = pd.read_parquet(f'{scen.preprep_path}/gwr.parquet')
    gwr_gdf = gpd.read_file(f'{scen.preprep_path}/gwr_gdf.geojson')
    gwr_all_building_gdf = gpd.read_file(f'{scen.preprep_path}/gwr_all_building_gdf.geojson')
    checkpoint_to_logfile(f'import gwr, {gwr.shape[0]} rows', scen.log_name, 5, scen.show_debug_prints)


    # SOLKAT ====================
    solkat_all_pq = pd.read_parquet(f'{scen.data_path}/input_split_data_geometry/solkat_pq.parquet')
    checkpoint_to_logfile(f'import solkat_pq, {solkat_all_pq.shape[0]} rows', scen.log_name,  1, scen.show_debug_prints)

    bsblso_bfs_numbers = get_bfs_from_ktnr([11,12,13], scen.data_path, scen.log_name)
    bsblso_bfs_numbers_TF = all([bfs in bsblso_bfs_numbers for bfs in scen.bfs_numbers])
    if (bsblso_bfs_numbers_TF) & (os.path.exists(f'{scen.data_path}/input_split_data_geometry/solkat_bsblso_geo.geojson')):
        solkat_all_geo = gpd.read_file(f'{scen.data_path}/input_split_data_geometry/solkat_bsblso_geo.geojson')
    else:  
        solkat_all_geo = gpd.read_file(f'{scen.data_path}/input_split_data_geometry/solkat_geo.geojson')
    checkpoint_to_logfile(f'import solkat_geo, {solkat_all_geo.shape[0]} rows', scen.log_name,  1, scen.show_debug_prints)    
    

    # minor transformations to str (with removing nan values)
    solkat_all_geo['DF_UID'] = solkat_all_geo['DF_UID'].astype(str)
    print('transform solkat_geo')
    
    solkat_all_pq['DF_UID'] = solkat_all_pq['DF_UID'].astype(str)    
    solkat_all_pq['SB_UUID'] = solkat_all_pq['SB_UUID'].astype(str)

    # solkat_all_pq['GWR_EGID'] = solkat_all_pq['GWR_EGID'].fillna('NAN').astype(str)
    solkat_all_pq['GWR_EGID'] = solkat_all_pq['GWR_EGID'].fillna(0).astype(int).astype(str)
    solkat_all_pq.loc[solkat_all_pq['GWR_EGID'] == '0', 'GWR_EGID'] = 'NAN'

    solkat_all_pq.rename(columns={'GWR_EGID': 'EGID'}, inplace=True)
    solkat_all_pq = solkat_all_pq.dropna(subset=['DF_UID'])
    
    solkat_all_pq['EGID_count'] = solkat_all_pq.groupby('EGID')['EGID'].transform('count')
    
    
    
    # add omitted EGIDs to SOLKAT ---------------------------------------------------------------------------------
    # old version, no EGIDs matched to solkat
    """
    if not scen.SOLKAT_match_missing_EGIDs_to_solkat_TF:
        solkat_v1 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(scen.bfs_numbers)])
        solkat_v1_wgeo = solkat_v1.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID') # merge geometry for later use
        solkat_v1_gdf = gpd.GeoDataFrame(solkat_v1_wgeo, geometry='geometry')
        solkat, solkat_gdf = copy.deepcopy(solkat_v1), copy.deepcopy(solkat_v1_gdf)
    elif scen.SOLKAT_match_missing_EGIDs_to_solkat_TF:
  """
    
    # the solkat df has missing EGIDs, for example row houses where the entire roof is attributed to one EGID. Attempt to 
    # 1 - add roof (perfectly overlapping roofpartitions) to solkat for all the EGIDs within the unions shape
    # 2- reduce the FLAECHE for all theses partitions by dividing it through the number of EGIDs in the union shape
    print_to_logfile('\nMatch missing EGIDs to solkat (where gwrEGIDs overlapp solkat shape but are not present as a single solkat_row)', scen.summary_name)
    cols_adjust_for_missEGIDs_to_solkat = scen.SOLKAT_cols_adjust_for_missEGIDs_to_solkat

    solkat_v2 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(scen.bfs_numbers)])
    solkat_v2_wgeo = solkat_v2.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID')
    solkat_v2_gdf = gpd.GeoDataFrame(solkat_v2_wgeo, geometry='geometry')
    solkat_v2_gdf = solkat_v2_gdf[solkat_v2_gdf['EGID'] != 'NAN']

    # create mapping of solkatEGIDs and missing gwrEGIDs 
    # union all shapes with the same EGID 
    solkat_union_v2EGID = solkat_v2_gdf.groupby('EGID').agg({
        'geometry': lambda x: unary_union(x),  # Combine all geometries into one MultiPolygon
        'DF_UID': lambda x: '_'.join(map(str, x))  # Concatenate DF_UID values as a single string
        }).reset_index()
    solkat_union_v2EGID = gpd.GeoDataFrame(solkat_union_v2EGID, geometry='geometry')
    

    # match gwrEGID through sjoin to solkat
    solkat_union_v2EGID = solkat_union_v2EGID.rename(columns = {'EGID': 'EGID_old_solkat'})  # rename EGID colum because gwr_EGIDs are now matched to union_shapes
    solkat_union_v2EGID.set_crs(gwr_gdf.crs, allow_override=True, inplace=True)
    join_gwr_solkat_union = gpd.sjoin(solkat_union_v2EGID, gwr_gdf, how='left')
    join_gwr_solkat_union.rename(columns = {'EGID': 'EGID_gwradded'}, inplace = True)
    checkpoint_to_logfile(f'nrows \n\tsolkat_all_pq: {solkat_all_pq.shape[0]}\t\t\tsolkat_v2_gdf: {solkat_v2_gdf.shape[0]} (remove EGID.NANs)\n\tsolkat_union_v2EGID: {solkat_union_v2EGID.shape[0]}\t\tjoin_gwr_solkat_union: {join_gwr_solkat_union.shape[0]}', scen.log_name, 3, scen.show_debug_prints)
    checkpoint_to_logfile(f'nEGID \n\tsolkat_all_pq: {solkat_all_pq["EGID"].nunique()}\t\t\tsolkat_v2_gdf: {solkat_v2_gdf["EGID"].nunique()} (remove EGID.NANs)\n\tsolkat_union_v2EGID_EGID_old: {solkat_union_v2EGID["EGID_old_solkat"].nunique()}\tjoin_gwr_solkat_union_EGID_old: {join_gwr_solkat_union["EGID_old_solkat"].nunique()}\tjoin_gwr_solkat_union_EGID_gwradded: {join_gwr_solkat_union["EGID_gwradded"].nunique()}', scen.log_name, 3, scen.show_debug_prints)


    # check EGID mapping case by case, add missing gwrEGIDs to solkat -------------------
    EGID_old_solkat_list = join_gwr_solkat_union['EGID_old_solkat'].unique()
    new_solkat_append_list = []
    add_solkat_counter, add_solkat_partition = 1, 4
    print_counter_max, i_print = 50, 0
    # n_egid, egid = 0, EGID_old_solkat_list[0]
    for n_egid, egid in enumerate(EGID_old_solkat_list):

        egid_join_union = join_gwr_solkat_union.loc[join_gwr_solkat_union['EGID_old_solkat'] == egid,]
        egid_join_union = egid_join_union.reset_index(drop = True)

        # Shapes of building that will not be included given GWR filter settings
        if any(egid_join_union['EGID_gwradded'].isna()):  
            solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
            solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']


        elif all(egid_join_union['EGID_gwradded'] != np.nan): 

            # a gwrEGID can be picked up by the union shape, but still be present in the solkat df, drop theses rows
            # => currently disabled, because most likely the smaller issue. Added gwrEGIDs that still have a solkatEGID outside the current selection will just also be added
            # to the extended solkat df, with 1/n share of the original solkatEGID for gwrEGID extension and full share of the otherwise already present EGID in solkatEGID. 
            if False:
                egid_to_add = egid_join_union['EGID_gwradded'].unique()[0]
                for egid_to_add in egid_join_union['EGID_gwradded'].unique():
                    if egid_join_union.shape[0] > 1:
                        if egid_to_add != egid:
                            if egid_to_add in EGID_old_solkat_list:
                                egid_join_union = egid_join_union.drop(egid_join_union.loc[egid_join_union['EGID_gwradded'] == egid_to_add].index)
                            elif egid_to_add == egid:
                                print('')


            # cases
            case1_TF = (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] == egid)
            case2_TF = (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] != egid)
            case3_TF = (egid_join_union.shape[0] > 1) & any(egid_join_union['EGID_gwradded'].isna())
            case4_TF = (egid_join_union.shape[0] > 1) & (egid in egid_join_union['EGID_gwradded'].to_list())
            case5_TF = (egid_join_union.shape[0] > 1) & (egid not in egid_join_union['EGID_gwradded'].to_list())

            # "Best" case (unless step above applies): Shapes of building that only has 1 GWR EGID
            if case1_TF:        # (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] == egid): 
                solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']
                
            # Not best case but for consistency better to keep individual solkatEGIs matches (otherwise missmatch of newer buildings with old shape partitions possible)
            elif case2_TF:      # (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] != egid): 
                solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']

            elif case3_TF:      # (egid_join_union.shape[0] > 1) & any(egid_join_union['EGID_gwradded'].isna()):
                checkpoint_to_logfile(f'**MAJOR ERROR**: EGID {egid}, np.nan in egid_join_union[EGID_gwradded] column', scen.log_name, 3, scen.show_debug_prints)

            # Intended case: Shapes of building that has multiple GWR EGIDs within the shape boundaries
            elif case4_TF:      # (egid_join_union.shape[0] > 1) & (egid in egid_join_union['EGID_gwradded'].to_list()):
                
                solkat_subdf_addedEGID_list = []
                n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):
                    
                    # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                    solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                    solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                    solkat_addedEGID['EGID'] = egid_to_add
                    
                    #extend the DF_UID with some numbers to have truely unique DF_UIDs
                    if scen.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                        str_suffix = str(n+1).zfill(5)
                        if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                        elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                    # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                    for col in cols_adjust_for_missEGIDs_to_solkat:
                        solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                    
                    # shrink topology to see which partitions are affected by EGID extensions
                    # solkat_addedEGID['geometry'] =solkat_addedEGID['geometry'].buffer(-0.5, resolution=16)

                    solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                
                # concat all EGIDs within the same shape that were previously missing
                solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)
                
            # Error case: Shapes of building that has multiple gwrEGIDs but does not overlap with the assigned / identical solkatEGID. 
            # Not proper solution, but best for now: add matching gwrEGID to solkatEGID selection, despite the acutall gwrEGID being placed in another shape. 
            elif case5_TF:      # (egid_join_union.shape[0] > 1) & (egid not in egid_join_union['EGID_gwradded'].to_list()):

                # attach a copy of one solkatEGID partition and set the EGID to the gwrEGID
                gwrEGID_row = copy.deepcopy(egid_join_union.iloc[0])
                # solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                gwrEGID_row['EGID_gwradded'] = egid
                egid_join_union = pd.concat([egid_join_union, gwrEGID_row.to_frame().T], ignore_index=True)

                # next follow all steps as in "Intended Case" above (solkat_shape with solkatEGID and gwrEGIDs)
                solkat_subdf_addedEGID_list = []
                n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                
                for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):

                    # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                    solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                    solkat_addedEGID['EGID'] = egid_to_add
                    
                    #extend the DF_UID with some numbers to have truely unique DF_UIDs
                    if scen.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                        str_suffix = str(n+1).zfill(3)
                        if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                        elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                    # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                    for col in cols_adjust_for_missEGIDs_to_solkat:
                        solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                    
                    # shrink topology to see which partitions are affected by EGID extensions
                    # solkat_addedEGID['geometry'] =solkat_addedEGID['geometry'].buffer(-0.5, resolution=16)

                    solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                
                # concat all EGIDs within the same shape that were previously missing
                solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)

                if i_print < print_counter_max:
                    checkpoint_to_logfile(f'ERROR: EGID {egid}: multiple gwrEGIDs, outside solkatEGID / without solkatEGID amongst them', scen.log_name, 1, scen.show_debug_prints)
                    i_print += 1
                elif i_print == print_counter_max:
                    checkpoint_to_logfile(f'ERROR: EGID {egid}: {print_counter_max}+ ... more cases of multiple gwrEGIDs, outside solkatEGID / without solkatEGID amongst them', scen.log_name, 1, scen.show_debug_prints)
                    i_print += 1

        if n_egid == int(len(EGID_old_solkat_list)/add_solkat_partition):
            checkpoint_to_logfile(f'Match gwrEGID to solkat: {add_solkat_counter}/{add_solkat_partition} partition', scen.log_name, 3, scen.show_debug_prints)
            
        # merge all solkat partitions to new solkat df
        new_solkat_append_list.append(solkat_subdf) 

    new_solkat_gdf = gpd.GeoDataFrame(pd.concat(new_solkat_append_list, ignore_index=True), geometry='geometry')
    new_solkat = new_solkat_gdf.drop(columns = ['geometry'])
    checkpoint_to_logfile(f'Extended solkat_df by {new_solkat.shape[0] - solkat_v2_gdf.shape[0]} rows (before matching: {solkat_v2_gdf.shape[0]}, after: {new_solkat.shape[0]} rows)', scen.summary_name, 3, scen.show_debug_prints)

    solkat, solkat_gdf = copy.deepcopy(new_solkat), copy.deepcopy(new_solkat_gdf)      
    

    # SOLKAT_MONTH ====================
    solkat_month_all_pq = pd.read_parquet(f'{scen.data_path}/input_split_data_geometry/solkat_month_pq.parquet')
    checkpoint_to_logfile(f'import solkat_month_pq, {solkat_month_all_pq.shape[0]} rows,', scen.log_name, 1, scen.show_debug_prints)

    # transformations
    solkat_month_all_pq['SB_UUID'] = solkat_month_all_pq['SB_UUID'].astype(str)
    solkat_month_all_pq['DF_UID'] = solkat_month_all_pq['DF_UID'].astype(str)
    solkat_month_all_pq = solkat_month_all_pq.merge(solkat_all_pq[['DF_UID', 'BFS_NUMMER']], how = 'left', on = 'DF_UID')
    solkat_month = solkat_month_all_pq[solkat_month_all_pq['BFS_NUMMER'].isin(scen.bfs_numbers)]


    # GRID_NODE ====================
    Map_egid_dsonode = pd.read_excel(f'{get_primeo_path()}/Daten_Primeo_x_UniBasel_V2.0.xlsx')
    # transformations
    Map_egid_dsonode.rename(columns={'ID_Trafostation': 'grid_node', 'Trafoleistung_kVA': 'kVA_threshold'}, inplace=True)
    Map_egid_dsonode['EGID'] = Map_egid_dsonode['EGID'].astype(str)
    Map_egid_dsonode['grid_node'] = Map_egid_dsonode['grid_node'].astype(str)

    egid_counts = Map_egid_dsonode['EGID'].value_counts()
    multip_egid_dsonode = egid_counts[egid_counts > 1].index
    single_egid_dsonode = []
    egid = multip_egid_dsonode[1]
    for egid in multip_egid_dsonode:
        subegid = Map_egid_dsonode.loc[Map_egid_dsonode['EGID'] == egid,]

        if subegid['grid_node'].nunique() == 1:
            single_egid_dsonode.append([egid, subegid['grid_node'].iloc[0], subegid['kVA_threshold'].iloc[0]])
        elif subegid['grid_node'].nunique() > 1:
            subegid = subegid.loc[subegid['kVA_threshold'] == subegid['kVA_threshold'].max(),]
            single_egid_dsonode.append([egid, subegid['grid_node'].iloc[0], subegid['kVA_threshold'].iloc[0]])

    single_egid_dsonode_df = pd.DataFrame(single_egid_dsonode, columns = ['EGID', 'grid_node', 'kVA_threshold'])
    # drop duplicates and append single_egid_dsonode_df
    Map_egid_dsonode.drop(Map_egid_dsonode[Map_egid_dsonode['EGID'].isin(multip_egid_dsonode)].index, inplace = True)
    Map_egid_dsonode = pd.concat([Map_egid_dsonode, single_egid_dsonode_df], ignore_index=True)
    

    # MAP: solkatdfuid > egid ---------------------------------------------------------------------------------
    Map_solkatdfuid_egid = solkat_gdf.loc[:,['DF_UID', 'DF_NUMMER', 'SB_UUID', 'EGID']].copy()
    Map_solkatdfuid_egid.rename(columns = {'GWR_EGID': 'EGID'}, inplace = True)
    Map_solkatdfuid_egid = Map_solkatdfuid_egid.loc[Map_solkatdfuid_egid['EGID'] != '']


    # MAP: egid > pv ---------------------------------------------------------------------------------
    def set_crs_to_gm_shp(gdf_CRS, gdf_a, gdf_b = None):
        gdf_a.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
        if gdf_b is not None:
            gdf_b.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
        
        if gdf_b is None: 
            return gdf_a
        if gdf_b is not None:
            return gdf_a, gdf_b
        

    # find optimal buffer size ====================
    if scen.SOLKAT_test_loop_optim_buff_size_TF:
        print_to_logfile('\n\n Check different buffersizes!', scen.log_name)
        arange_start, arange_end, arange_step = scen.SOLKAT_test_loop_optim_buff_arang[0], scen.SOLKAT_test_loop_optim_buff_arang[1], scen.SOLKAT_test_loop_optim_buff_arang[2]
        buff_range = np.arange(arange_start, arange_end, arange_step)
        shares_xtf_duplicates = []
        for i in buff_range:# [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 2]:
            print_to_logfile(f'buffer size: {i}', scen.log_name)

            gwr_loop = copy.deepcopy(gwr_gdf)
            gwr_loop.set_crs("EPSG:32632", allow_override=True, inplace=True)
            gwr_loop['geometry'] = gwr_loop['geometry'].buffer(i)
            pv_loop = copy.deepcopy(pv_gdf)
            gwr_loop, pv_loop = set_crs_to_gm_shp(gm_shp_gdf, gwr_loop, pv_loop)
            gwregid_pvid_loop = gpd.sjoin(pv_loop,gwr_loop, how="left", predicate="within")
            gwregid_pvid_loop.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)

            shares = [i, 
                    round(sum(gwregid_pvid_loop['xtf_id'].isna())           /gwregid_pvid_loop['xtf_id'].nunique(),2), 
                    round(sum(gwregid_pvid_loop['xtf_id'].value_counts()==1)/gwregid_pvid_loop['xtf_id'].nunique(),2), 
                    round(sum(gwregid_pvid_loop['xtf_id'].value_counts()==2)/gwregid_pvid_loop['xtf_id'].nunique(),2), 
                    round(sum(gwregid_pvid_loop['xtf_id'].value_counts()>2) /gwregid_pvid_loop['xtf_id'].nunique(),2) ]
            shares_xtf_duplicates.append(shares)
            
            print_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid_loop["EGID"].isna().sum() / gwregid_pvid_loop.shape[0] *100,2)} % of pv rows ({gwregid_pvid_loop.shape[0]}) are missing EGID', scen.log_name)
            print_to_logfile(f'Duplicate shares: \tNANs\tunique\t2x\t>2x \n \t\t\t{shares[0]}\t{shares[1]}\t{shares[2]}\t{shares[3]}\t{sum(shares)}\n', scen.log_name)
        
        # plot shares of successful mappings
        # shares_xtf_duplicates_df = pd.DataFrame(shares_xtf_duplicates, columns = ['buffer_size', 'NANs', 'unique', '2x', '>2x'])
        # not plotted because over-exaggerated buffer is later corrected with closest neighbour matching
        # fig = px.line(shares_xtf_duplicates_df, 
        #               x='buffer_size', y=['NANs', 'unique', '2x', '>2x'],
        #               title = 'Shares of xtf_id duplicates', labels = {'buffer_size': 'Buffer Size', 'value': 'Share'}, width = 800, height = 400)
        # fig.show()
        # fig.write_html(f'{data_path_def}/output/preprep_data/by_buffersize_share_xtf_id_duplicates.html')
        checkpoint_to_logfile('buffer size optimisation finished', scen.log_name, 2, scen.show_debug_prints)


    # (continued MAP: egid > pv) ----------
    gwr_buff_gdf = copy.deepcopy(gwr_gdf)
    gwr_buff_gdf.set_crs("EPSG:32632", allow_override=True, inplace=True)
    gwr_buff_gdf['geometry'] = gwr_buff_gdf['geometry'].buffer(scen.SOLKAT_GWR_EGID_buffer_size)
    gwr_buff_gdf, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, gwr_buff_gdf, pv_gdf)
    checkpoint_to_logfile(f'gwr_gdf.crs == pv_gdf.crs: {gwr_buff_gdf.crs == pv_gdf.crs}', scen.log_name, 6, scen.show_debug_prints)

    gwregid_pvid_all = gpd.sjoin(pv_gdf,gwr_buff_gdf, how="left", predicate="within")
    gwregid_pvid_all.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)

    # keep only unique xtf_ids 
    gwregid_pvid_unique = copy.deepcopy(gwregid_pvid_all.loc[~gwregid_pvid_all.duplicated(subset='xtf_id', keep=False)])
    xtf_duplicates =      copy.deepcopy(gwregid_pvid_all.loc[ gwregid_pvid_all.duplicated(subset='xtf_id', keep=False)])
    checkpoint_to_logfile(f'sum n_unique xtf_ids: {gwregid_pvid_unique["xtf_id"].nunique()} (unique df) +{xtf_duplicates["xtf_id"].nunique()} (duplicates df) = {gwregid_pvid_unique["xtf_id"].nunique()+xtf_duplicates["xtf_id"].nunique() }; n_unique in pv_gdf: {pv_gdf["xtf_id"].nunique()}', scen.log_name, 6, scen.show_debug_prints)
   
   # match duplicates with nearest neighbour
    xtf_nearestmatch_list = []
    xtf_id = xtf_duplicates['xtf_id'].unique()[0]
    for xtf_id in xtf_duplicates['xtf_id'].unique():
        gwr_sub = copy.deepcopy(gwr_buff_gdf.loc[gwr_buff_gdf['EGID'].isin(xtf_duplicates.loc[xtf_duplicates['xtf_id'] == xtf_id, 'EGID'])])
        pv_sub = copy.deepcopy(pv_gdf.loc[pv_gdf['xtf_id'] == xtf_id])
        
        assert pv_sub.crs == gwr_sub.crs
        gwr_sub['distance_to_pv'] = gwr_sub['geometry'].centroid.distance(pv_sub['geometry'].values[0])
        pv_sub['EGID'] = gwr_sub.loc[gwr_sub['distance_to_pv'].idxmin()]['EGID']

        xtf_nearestmatch_list.append(pv_sub)
    
    xtf_nearestmatches_df = pd.concat(xtf_nearestmatch_list, ignore_index=True)
    gwregid_pvid = pd.concat([gwregid_pvid_unique, xtf_nearestmatches_df], ignore_index=True).drop_duplicates()
    checkpoint_to_logfile(f'total unique xtf: {pv_gdf["xtf_id"].nunique()} (pv_gdf); {gwregid_pvid_unique["xtf_id"].nunique()+xtf_nearestmatches_df["xtf_id"].nunique()} (unique + nearest match)', scen.log_name, 6, scen.show_debug_prints)

    checkpoint_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid["EGID"].isna().sum() / gwregid_pvid.shape[0] *100,2)} % of pv rows ({gwregid_pvid.shape[0]}) are missing EGID', scen.log_name, 6, scen.show_debug_prints)
    # Map_egid_pv = gwregid_pvid.loc[gwregid_pvid['EGID'].notna(), ['EGID', 'xtf_id']].copy()
    Map_egid_pv = gwregid_pvid[['EGID', 'xtf_id']].copy()


    # CHECK SELECTION: - OMITTED SPATIAL POINTS / POLYS ---------------------------------------------------------------------------------
    print_to_logfile('\nnumber of omitted buildings because EGID is (not) / present in all of GWR / Solkat / PV / Grid_Node', scen.summary_name)
    print_to_logfile(f'>gwr settings: \n n bfs_numbers: {len(scen.bfs_numbers)}, \n year_range: {scen.year_range}, \n building class GKLAS: {scen.GWR_GKLAS}, \n building status GSTAT: {scen.GWR_GSTAT}, \n year of construction GBAUJ: {scen.GWR_GBAUJ_minmax}', scen.summary_name)
    omitt_gwregid_gdf = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(solkat_gdf['EGID'])])
    checkpoint_to_logfile(f'omitt_gwregid_gdf (gwr not in solkat): {omitt_gwregid_gdf.shape[0]} rows ({round((omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%), gwr[EGID].unique: {gwr_gdf["EGID"].nunique()})', scen.summary_name, 2, True)

    omitt_solkat_all_gwr_gdf = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_all_building_gdf['EGID'])])
    omitt_solkat_gdf = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_gdf['EGID'])])
    checkpoint_to_logfile(f'omitt_solkat_gdf (solkat not in gwr): {omitt_solkat_gdf.shape[0]} rows ({round((omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0])*100, 2)}%), solkat[EGID].unique: {solkat_gdf["EGID"].nunique()})', scen.summary_name, 2, True)

    omitt_pv_gdf = copy.deepcopy(pv_gdf.loc[~pv_gdf['xtf_id'].isin(gwregid_pvid['xtf_id'])])
    checkpoint_to_logfile(f'omitt_pv_gdf (pv not in gwr): {omitt_pv_gdf.shape[0]} rows ({round((omitt_pv_gdf.shape[0]/pv_gdf.shape[0])*100, 2)}%, pv[xtf_id].unique: {pv_gdf["xtf_id"].nunique()})', scen.summary_name, 2, True)

    omitt_gwregid_gridnode_gdf = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(Map_egid_dsonode['EGID'])])
    checkpoint_to_logfile(f'omitt_gwregid_gridnode_gdf (gwr not in gridnode): {omitt_gwregid_gridnode_gdf.shape[0]} rows ({round((omitt_gwregid_gridnode_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%), gwr[EGID].unique: {gwr_gdf["EGID"].nunique()})', scen.summary_name, 2, True)

    omitt_gridnodeegid_gwr_df = copy.deepcopy(Map_egid_dsonode.loc[~Map_egid_dsonode['EGID'].isin(gwr_gdf['EGID'])])
    checkpoint_to_logfile(f'omitt_gridnodeegid_gwr_df (gridnode not in gwr): {omitt_gridnodeegid_gwr_df.shape[0]} rows ({round((omitt_gridnodeegid_gwr_df.shape[0]/Map_egid_dsonode.shape[0])*100, 2)}%), gridnode[EGID].unique: {Map_egid_dsonode["EGID"].nunique()})', scen.summary_name, 2, True)
    

    # CHECK SELECTION: - PRINTS TO SUMMARY LOG FILE ---------------------------------------------------------------------------------
    print_to_logfile('\n\nHow well does GWR cover other data sources', scen.summary_name)
    checkpoint_to_logfile(f'gwr_EGID omitted in solkat: {round(omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0]*100, 2)} %', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'solkat_EGID omitted in gwr_all_bldng: {round(omitt_solkat_all_gwr_gdf.shape[0]/solkat_gdf.shape[0]*100, 2)} %', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'solkat_EGID omitted in gwr: {round(omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0]*100, 2)} %', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'pv_xtf_id omitted in gwr: {round(omitt_pv_gdf.shape[0]/pv_gdf.shape[0]*100, 2)} %', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'gwr_EGID omitted in gridnode: {round(omitt_gwregid_gridnode_gdf.shape[0]/gwr_gdf.shape[0]*100, 2)} %', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'gridnode_EGID omitted in gwr: {round(omitt_gridnodeegid_gwr_df.shape[0]/Map_egid_dsonode.shape[0]*100, 2)} %', scen.summary_name, 2, True)


    # EXPORTS (parquet) ---------------------------------------------------------------------------------
    df_to_export_names = ['pv', 'solkat', 'solkat_month', 'Map_egid_dsonode', 'Map_solkatdfuid_egid', 'Map_egid_pv']
    df_to_export_list = [pv, solkat, solkat_month,  Map_egid_dsonode, Map_solkatdfuid_egid, Map_egid_pv] 
    for i, df in enumerate(df_to_export_list):
        df.to_parquet(f'{scen.preprep_path}/{df_to_export_names[i]}.parquet')
        # df.to_csv(f'{scen.preprep_path}/{df_to_export_names[i]}.csv', sep=';', index=False)
        checkpoint_to_logfile(f'{df_to_export_names[i]} exported to prepreped data', scen.log_name, 1, scen.show_debug_prints)


    # EXPORT SPATIAL DATA ---------------------------------------------------------------------------------
    gdf_to_export_names = ['gm_shp_gdf', 'pv_gdf', 'solkat_gdf', 'gwr_gdf','gwr_buff_gdf', 'gwr_all_building_gdf', 
                           'omitt_gwregid_gdf', 'omitt_solkat_gdf', 'omitt_pv_gdf', 'omitt_gwregid_gridnode_gdf' ]
    gdf_to_export_list = [gm_shp_gdf, pv_gdf, solkat_gdf, gwr_gdf, gwr_buff_gdf, gwr_all_building_gdf, 
                          omitt_gwregid_gdf, omitt_solkat_gdf, omitt_pv_gdf, omitt_gwregid_gridnode_gdf]
    
    for i,g in enumerate(gdf_to_export_list):
        cols_DATUM = [col for col in g.columns if 'DATUM' in col]
        g.drop(columns = cols_DATUM, inplace = True)
        # for each gdf export needs to be adjusted so it is carried over into the geojson file.
        g.set_crs("EPSG:2056", allow_override = True, inplace = True)   

        print_to_logfile(f'CRS for {gdf_to_export_names[i]}: {g.crs}', scen.log_name)
        checkpoint_to_logfile(f'exported {gdf_to_export_names[i]}', scen.log_name , 4, scen.show_debug_prints)

        with open(f'{scen.preprep_path}/{gdf_to_export_names[i]}.geojson', 'w') as f:
            f.write(g.to_json()) 


# ------------------------------------------------------------------------------------------------------
# IMPORT ELECTRICITY DEMAND TS + MATCH TO HOUSEHOLDS
# ------------------------------------------------------------------------------------------------------

def import_demand_TS_AND_match_households(
        scen, ):
    """
    1) Import demand time series data and aggregate it to 4 demand archetypes.
    2) Match the time series to the households IDs dependent on building characteristics (e.g. flat/house size, electric heating, etc.)
       Export all the mappings and time series data.
    """
    # SETUP --------------------------------------
    print_to_logfile('run function: import_demand_TS_AND_match_households.py', scen.log_name)


    # IMPORT CONSUMER DATA -----------------------------------------------------------------
    print_to_logfile(f'\nIMPORT CONSUMER DATA {10*"*"}', scen.log_name) 
       

    # DEMAND DATA SOURCE: NETFLEX ============================================================
    if scen.DEMAND_input_data_source == "NETFLEX" :
        # import demand TS --------
        netflex_consumers_list = glob.glob(f'{scen.data_path}/input/NETFLEX_consumers/ID*') # os.listdir(f'{scen.data_path}/input/NETFLEX_consumers')
                
        all_assets_list = []
        # c = netflex_consumers_list[1]
        for path in netflex_consumers_list:
            f = open(path)
            data = json.load(f)
            assets = data['assets']['list'] 
            all_assets_list.extend(assets)
        
        without_id = [a.split('_ID')[0] for a in all_assets_list]
        all_assets_unique = list(set(without_id))
        checkpoint_to_logfile(f'consumer demand TS contains assets: {all_assets_unique}', scen.log_name, 2, scen.show_debug_prints)

        # aggregate demand for each consumer
        agg_demand_df = pd.DataFrame()
        # netflex_consumers_list = netflex_consumers_list if not smaller_import_def else netflex_consumers_list[0:40]

        # for c, c_n in enumerate() netflex_consumers_list:
        for i_path, path in enumerate(netflex_consumers_list):
            c_demand_id, c_demand_tech, c_demand_asset, c_demand_t, c_demand_values = [], [], [], [], []

            f = open(path)
            data = json.load(f)
            assets = data['assets']['list'] 

            a = assets[0]
            for a in assets:
                if 'asset_time_series' in data['assets'][a].keys():
                    demand = data['assets'][a]['asset_time_series']
                    c_demand_id.extend([f'ID{a.split("_ID")[1]}']*len(demand))
                    c_demand_tech.extend([a.split('_ID')[0]]*len(demand))
                    c_demand_asset.extend([a]*len(demand))
                    c_demand_t.extend(demand.keys())
                    c_demand_values.extend(demand.values())

            c_demand_df = pd.DataFrame({'id': c_demand_id, 'tech': c_demand_tech, 'asset': c_demand_asset, 't': c_demand_t, 'value': c_demand_values})
            agg_demand_df = pd.concat([agg_demand_df, c_demand_df])
            
            if (i_path + 1) % (len(netflex_consumers_list) // 4) == 0:
                id_name = f'ID{path.split("ID")[-1].split(".json")[0]}'
                checkpoint_to_logfile(f'exported demand TS for consumer {id_name}, {i_path+1} of {len(netflex_consumers_list)}', scen.log_name, 2, scen.show_debug_prints)
        
        # remove pv assets because they also have negative values
        agg_demand_df = agg_demand_df[agg_demand_df['tech'] != 'pv']

        agg_demand_df['value'] = agg_demand_df['value'] * 1000 # it appears that values are calculated in MWh, need kWh

        # plot TS for certain consumers by assets
        plot_ids =['ID100', 'ID101', 'ID102', ]
        plot_df = agg_demand_df[agg_demand_df['id'].isin(plot_ids)]
        fig = px.line(plot_df, x='t', y='value', color='asset', title='Demand TS for selected consumers')
        # fig.show()

        # export aggregated demand for all NETFLEX consumer assets
        agg_demand_df.to_parquet(f'{scen.preprep_path}/demand_ts.parquet')
        checkpoint_to_logfile('exported demand TS for all consumers', scen.log_name, scen.show_debug_prints)
        

        # AGGREGATE DEMAND TYPES -----------------------------------------------------------------
        # aggregate demand TS for defined consumer types 
        # demand upper/lower 50 percentile, with/without heat pump
        # get IDs for each subcatergory
        print_to_logfile(f'\nAGGREGATE DEMAND TYPES {10*"*"}', scen.log_name)
        def get_IDs_upper_lower_totalconsumpttion_by_hp(df, hp_TF = True,  up_low50percent = "upper"):
            id_with_hp = df[df['tech'] == 'hp']['id'].unique()
            if hp_TF: 
                filtered_df = df[df['id'].isin(id_with_hp)]
            elif not hp_TF:
                filtered_df = df[~df['id'].isin(id_with_hp)]

            filtered_df = filtered_df.loc[filtered_df['tech'] != 'pv']

            total_consumption = filtered_df.groupby('id')['value'].sum().reset_index()
            mean_value = total_consumption['value'].mean()
            id_upper_half = total_consumption.loc[total_consumption['value'] > mean_value, 'id']
            id_lower_half = total_consumption.loc[total_consumption['value'] < mean_value, 'id']

            if up_low50percent == "upper":
                return id_upper_half
            elif up_low50percent == "lower":
                return id_lower_half
        
        # classify consumers to later aggregate them into demand types
        ids_high_wiHP = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = True, up_low50percent = "upper")
        ids_low_wiHP  = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = True, up_low50percent = "lower")
        ids_high_noHP = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = False, up_low50percent = "upper")
        ids_low_noHP  = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = False, up_low50percent = "lower")

        # aggregate demand types
        demandtypes = pd.DataFrame()
        t_sequence = agg_demand_df['t'].unique()

        demandtypes['t'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().keys()
        # demandtypes['high_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().values
        # demandtypes['low_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_wiHP)].groupby('t')['value'].mean().values
        # demandtypes['high_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_noHP)].groupby('t')['value'].mean().values
        # demandtypes['low_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_noHP)].groupby('t')['value'].mean().values
        demandtypes['high_DEMANDprox_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().values
        demandtypes['low_DEMANDprox_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_wiHP)].groupby('t')['value'].mean().values
        demandtypes['high_DEMANDprox_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_noHP)].groupby('t')['value'].mean().values
        demandtypes['low_DEMANDprox_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_noHP)].groupby('t')['value'].mean().values

        demandtypes['t'] = pd.Categorical(demandtypes['t'], categories=t_sequence, ordered=True)
        demandtypes = demandtypes.sort_values(by = 't')
        demandtypes = demandtypes.reset_index(drop=True)

        demandtypes.to_parquet(f'{scen.preprep_path}/demandtypes.parquet')
        demandtypes.to_csv(f'{scen.preprep_path}/demandtypes.csv', sep=';', index=False)
        checkpoint_to_logfile('exported demand types', scen.log_name, scen.show_debug_prints)

        # plot demand types with plotly
        fig = px.line(demandtypes, x='t', y=['high_DEMANDprox_wiHP', 'low_DEMANDprox_wiHP', 'high_DEMANDprox_noHP', 'low_DEMANDprox_noHP'], title='Demand types')
        # fig.show()
        fig.write_html(f'{scen.preprep_path}/demandtypes.html')
        demandtypes['high_DEMANDprox_wiHP'].sum(), demandtypes['low_DEMANDprox_wiHP'].sum(), demandtypes['high_DEMANDprox_noHP'].sum(), demandtypes['low_DEMANDprox_noHP'].sum()


        # MATCH DEMAND TYPES TO HOUSEHOLDS -----------------------------------------------------------------
        print_to_logfile(f'\nMATCH DEMAND TYPES TO HOUSEHOLDS {10*"*"}', scen.log_name)

        # import GWR and PV --------
        gwr_all = pd.read_parquet(f'{scen.preprep_path}/gwr.parquet')
        checkpoint_to_logfile('imported gwr data', scen.log_name, scen.show_debug_prints)
        
        # transformations
        gwr_all[scen.GWR_DEMAND_proxy] = pd.to_numeric(gwr_all[scen.GWR_DEMAND_proxy], errors='coerce')
        gwr_all['GBAUJ'] = pd.to_numeric(gwr_all['GBAUJ'], errors='coerce')
        gwr_all.dropna(subset = ['GBAUJ'], inplace = True)
        gwr_all['GBAUJ'] = gwr_all['GBAUJ'].astype(int)

        # selection based on GWR specifications -------- 
        # select columns GSTAT that are within list ['1110','1112'] and GKLAS in ['1234','2345']
        gwr = gwr_all[(gwr_all['GSTAT'].isin(scen.GWR_GSTAT)) & 
                    (gwr_all['GKLAS'].isin(scen.GWR_GKLAS)) & 
                    (gwr_all['GBAUJ'] >= scen.GWR_GBAUJ_minmax[0]) &
                    (gwr_all['GBAUJ'] <= scen.GWR_GBAUJ_minmax[1])]
        checkpoint_to_logfile(f'filtered vs unfiltered gwr: shape ({gwr.shape[0]} vs {gwr_all.shape[0]}), EGID.nunique ({gwr["EGID"].nunique()} vs {gwr_all ["EGID"].nunique()})', scen.log_name, 2, scen.show_debug_prints)
        
        def get_IDs_upper_lower_DEMAND_by_hp(df, DEMAND_col = scen.GWR_DEMAND_proxy,  hp_TF = True,  up_low50percent = "upper"):
            id_with_hp = df[df['GWAERZH1'].isin(scen.GWR_GWAERZH)]['EGID'].unique()
            if hp_TF: 
                filtered_df = df[df['EGID'].isin(id_with_hp)]
            elif not hp_TF:
                filtered_df = df[~df['EGID'].isin(id_with_hp)]
            
            mean_value = filtered_df[DEMAND_col].mean()
            id_upper_half = filtered_df.loc[filtered_df[DEMAND_col] > mean_value, 'EGID']
            id_lower_half = filtered_df.loc[filtered_df[DEMAND_col] < mean_value, 'EGID']
            if up_low50percent == "upper":
                return id_upper_half.tolist()
            elif up_low50percent == "lower":
                return id_lower_half.tolist()
            
        high_DEMANDprox_wiHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = True, up_low50percent = "upper")
        low_DEMANDprox_wiHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = True, up_low50percent = "lower")
        high_DEMANDprox_noHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = False, up_low50percent = "upper")
        low_DEMANDprox_noHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = False, up_low50percent = "lower")


        # sanity check --------
        print_to_logfile('sanity check gwr classifications', scen.log_name)
        gwr_classified_list = [high_DEMANDprox_wiHP_list, low_DEMANDprox_wiHP_list, high_DEMANDprox_noHP_list, low_DEMANDprox_noHP_list]
        gwr_classified_names= ['high_DEMANDprox_wiHP_list', 'low_DEMANDprox_wiHP_list', 'high_DEMANDprox_noHP_list', 'low_DEMANDprox_noHP_list']

        for chosen_lst_idx, chosen_list in enumerate(gwr_classified_list):
            chosen_set = set(chosen_list)

            for i, lst in enumerate(gwr_classified_list):
                if i != chosen_lst_idx:
                    other_set = set(lst)
                    common_ids = chosen_set.intersection(other_set)
                    print_to_logfile(f"No. of common IDs between {gwr_classified_names[chosen_lst_idx]} and {gwr_classified_names[i]}: {len(common_ids)}", scen.log_name)
            print_to_logfile('\n', scen.log_name)

        # precent of classified buildings
        n_classified = sum([len(lst) for lst in gwr_classified_list])
        n_all = len(gwr['EGID'])
        print_to_logfile(f'{n_classified} of {n_all} ({round(n_classified/n_all*100, 2)}%) gwr rows are classfied', scen.log_name)
        

        # export to JSON --------
        Map_demandtype_EGID ={
            'high_DEMANDprox_wiHP': high_DEMANDprox_wiHP_list,
            'low_DEMANDprox_wiHP': low_DEMANDprox_wiHP_list,
            'high_DEMANDprox_noHP': high_DEMANDprox_noHP_list,
            'low_DEMANDprox_noHP': low_DEMANDprox_noHP_list,
        }
        with open(f'{scen.preprep_path}/Map_demandtype_EGID.json', 'w') as f:
            json.dump(Map_demandtype_EGID, f)
        checkpoint_to_logfile('exported Map_demandtype_EGID.json', scen.log_name, scen.show_debug_prints)

        Map_EGID_demandtypes = {}
        for type, egid_list in Map_demandtype_EGID.items():
            for egid in egid_list:
                Map_EGID_demandtypes[egid] = type
        with open(f'{scen.preprep_path}/Map_EGID_demandtypes.json', 'w') as f:
            json.dump(Map_EGID_demandtypes, f)
        checkpoint_to_logfile('exported Map_EGID_demandtypes.json', scen.log_name, scen.show_debug_prints)


    # DEMAND DATA SOURCE: SwissStore ============================================================
    elif scen.DEMAND_input_data_source == "SwissStore" :
        print("STUCK")     # follow up call with Hector. => for match all houses to archetypes of Swisstore and then later extract demand profile
        
        swstore_demand_inclnan = pd.read_excel(f'{scen.data_path}/input/SwissStore_DemandData/Electricity_demand_SFH_MFH.xlsx')
        swstore_demand = swstore_demand_inclnan.loc[~swstore_demand_inclnan['time'].isna()]
        swstore_demand['SFH'].sum(), swstore_demand['MFH'].sum()
        swstore_demand.head()
        swstore_demand.shape


        os.listdir(f'{scen.data_path}/input/SwissStore_DemandData/Electricity_demand_SFH_MFH.xlsx')


# ------------------------------------------------------------------------------------------------------
# IMPORT METEO DATA
# ------------------------------------------------------------------------------------------------------

def import_meteo_data(
        scen, ):
    """
    Import meteo data from a source, select only the relevant time frame store data to prepreped data folder.
    """
    
    # SETUP --------------------------------------
    print_to_logfile('run function: import_demand_TS_AND_match_households.py', scen.log_name)


    # IMPORT METEO DATA ============================================================================
    print_to_logfile(f'\nIMPORT METEO DATA {10*"*"}', scen.log_name)

    # import meteo data --------
    meteo = pd.read_csv(f'{scen.data_path}/input/Meteoblue_BSBL/Meteodaten_Basel_2018_2024_reduziert_bereinigt.csv')

    # transformations
    meteo['timestamp'] = pd.to_datetime(meteo['timestamp'], format = '%d.%m.%Y %H:%M:%S')

    # select relevant time frame
    start_stamp = pd.to_datetime(f'01.01.{scen.year_range[0]}', format = '%d.%m.%Y')
    end_stamp = pd.to_datetime(f'31.12.{scen.year_range[1]}', format = '%d.%m.%Y')
    meteo = meteo[(meteo['timestamp'] >= start_stamp) & (meteo['timestamp'] <= end_stamp)]
    
    # export --------
    meteo.to_parquet(f'{scen.preprep_path}/meteo.parquet')
    checkpoint_to_logfile('exported meteo data', scen.log_name, scen.show_debug_prints)

    # MATCH WEATHER STATIONS TO HOUSEHOLDS ============================================================================


