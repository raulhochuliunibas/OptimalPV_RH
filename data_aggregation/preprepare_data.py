import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import winsound
import json
import plotly.express as px
import copy

from datetime import datetime
from shapely.geometry import Point
from shapely.ops import unary_union

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# BY SB_UUID - FALSE - IMPORT LOCAL DATA + create SPATIAL MAPPINGS
# ------------------------------------------------------------------------------------------------------
def get_earlier_api_import_data(dataagg_settings_def):
    """
    Function to import all api input data, previously downloaded and stored through various API calls
    """

    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    data_path_def = dataagg_settings_def['data_path']
    year_range_def = dataagg_settings_def['year_range']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    input_api_path = f'{data_path_def}/input_api'

    # import and store data in preprep data -------------------
    # Map_gm_ewr
    Map_gm_ewr = pd.read_parquet(f'{input_api_path}/Map_gm_ewr.parquet')
    Map_gm_ewr.to_parquet(f'{data_path_def}/output/preprep_data/Map_gm_ewr.parquet')
    Map_gm_ewr.to_csv(f'{data_path_def}/output/preprep_data/Map_gm_ewr.csv', sep=';', index=False)
    checkpoint_to_logfile(f'Map_gm_ewr stored in prepreped data', dataagg_settings_def['log_file_name'],2, show_debug_prints_def)

    # pvtarif
    pvtarif_all = pd.read_parquet(f'{input_api_path}/pvtarif.parquet')
    year_range_2int = [str(year % 100).zfill(2) for year in range(year_range_def[0], year_range_def[1]+1)]
    pvtarif = copy.deepcopy(pvtarif_all.loc[pvtarif_all['year'].isin(year_range_2int), :])
    pvtarif.to_parquet(f'{data_path_def}/output/preprep_data/pvtarif.parquet')
    pvtarif.to_csv(f'{data_path_def}/output/preprep_data/pvtarif.csv', sep=';', index=False)
    checkpoint_to_logfile(f'pvtarif stored in prepreped data', dataagg_settings_def['log_file_name'], 2, show_debug_prints_def)

        

# ------------------------------------------------------------------------------------------------------
# BY SB_UUID - FALSE - IMPORT LOCAL DATA + create SPATIAL MAPPINGS
# ------------------------------------------------------------------------------------------------------
def local_data_AND_spatial_mappings(
        dataagg_settings_def, ):
    """
    Function to import all the local data sources, remove and transform data where necessary and store only
    the required data that is in range with the BFS municipality selection. When applicable, create mapping
    files, so that so that different data sets can be matched and spatial data can be reidentified to their 
    geometry if necessary. 
    """

    # import settings + setup ---------------------------------------------------------------------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']
    summary_file_name = dataagg_settings_def['summary_file_name']

    gwr_selection_specs_def = dataagg_settings_def['gwr_selection_specs']
    solkat_selection_specs_def = dataagg_settings_def['solkat_selection_specs']
    print_to_logfile(f'run function: local_data_AND_spatial_mappings.py', log_file_name_def = log_file_name_def)


    # IMPORT DATA ---------------------------------------------------------------------------------
    gm_shp_gdf = gpd.read_file(f'{data_path_def}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    
    def cols_to_str(col_list, df1, df2 = None, repl_val = ''):
        for col in col_list:
            if isinstance(df1[col].iloc[0], float):
                df1[col] = df1[col].replace(np.nan, 0).astype(int).astype(str)
                df1[col] = df1[col].replace('0', repl_val)
            elif isinstance(df1[col].iloc[0], int):
                df1[col] = df1[col].replace(np.nan, 0).astype(str)
                df1[col] = df1[col].replace('0', repl_val)
            if df2 is not None:
                if isinstance(df2[col].iloc[0], float):
                    df2[col] = df2[col].replace(np.nan, 0).astype(int).astype(str)
                    df2[col] = df2[col].replace('0', repl_val)
                elif isinstance(df2[col].iloc[0], int):
                    df2[col] = df2[col].replace(np.nan, 0).astype(str)
                    df2[col] = df2[col].replace('0', repl_val)
        if df2 is None:
            return df1
        else:
            return df1, df2
    

    # PV ====================
    pv_all_pq = pd.read_parquet(f'{data_path_def}/split_data_geometry/pv_pq.parquet')
    checkpoint_to_logfile(f'import pv_pq, {pv_all_pq.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  2, show_debug_prints_def)
    pv_all_geo = gpd.read_file(f'{data_path_def}/split_data_geometry/pv_geo.geojson')
    checkpoint_to_logfile(f'import pv_geo, {pv_all_geo.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  2, show_debug_prints_def)

    # transformations
    pv_all_pq, pv_all_geo = cols_to_str(['xtf_id',], pv_all_pq, pv_all_geo)

    pv = pv_all_pq[pv_all_pq['BFS_NUMMER'].isin(bfs_number_def)]  # select and export pq for BFS numbers
    pv_wgeo = pv.merge(pv_all_geo[['xtf_id', 'geometry']], how = 'left', on = 'xtf_id') # merge geometry for later use
    pv_gdf = gpd.GeoDataFrame(pv_wgeo, geometry='geometry')


    # GWR ====================
    gwr = pd.read_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
    gwr_gdf = gpd.read_file(f'{data_path_def}/output/preprep_data/gwr_gdf.geojson')
    gwr_all_building_gdf = gpd.read_file(f'{data_path_def}/output/preprep_data/gwr_all_building_gdf.geojson')
    checkpoint_to_logfile(f'import gwr, {gwr.shape[0]} rows', log_file_name_def, 5, show_debug_prints_def = show_debug_prints_def)


    # SOLKAT ====================
    solkat_all_pq = pd.read_parquet(f'{data_path_def}/split_data_geometry/solkat_pq.parquet')
    checkpoint_to_logfile(f'import solkat_pq, {solkat_all_pq.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  1, show_debug_prints_def)
    
    bsblso_kt_numbers_TF = any([kt in [11,12,13] for kt in dataagg_settings_def['kt_numbers']])
    if (bsblso_kt_numbers_TF) & (os.path.exists(f'{data_path_def}/split_data_geometry/solkat_bsblso_geo.geojson')):
        solkat_all_geo = gpd.read_file(f'{data_path_def}/split_data_geometry/solkat_bsblso_geo.geojson')
    else:
        solkat_all_geo = gpd.read_file(f'{data_path_def}/split_data_geometry/solkat_geo.geojson')
    checkpoint_to_logfile(f'import solkat_geo, {solkat_all_geo.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  1, show_debug_prints_def)

    # minor transformations
    solkat_all_pq = cols_to_str(['SB_UUID', 'DF_UID', 'GWR_EGID'], solkat_all_pq)
    solkat_all_geo = cols_to_str(['DF_UID',], solkat_all_geo)
    solkat_all_pq.rename(columns = {'GWR_EGID': 'EGID'}, inplace = True)
    solkat_all_pq['EGID'] = solkat_all_pq['EGID'].replace('', 'NAN')
    solkat_all_pq['EGID_count'] = solkat_all_pq.groupby('EGID')['EGID'].transform('count')
    solkat_all_pq.dtypes

    # add omitted EGIDs to SOLKAT ====================
    # old version, no EGIDs matched to solkat
    if not solkat_selection_specs_def['match_missing_EGIDs_to_solkat_TF']:
        solkat_v1 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(bfs_number_def)])  # select and export pq for BFS numbers
        solkat_v1_wgeo = solkat_v1.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID') # merge geometry for later use
        solkat_v1_gdf = gpd.GeoDataFrame(solkat_v1_wgeo, geometry='geometry')
        solkat, solkat_gdf = copy.deepcopy(solkat_v1), copy.deepcopy(solkat_v1_gdf)
    
    # the solkat df has missing EGIDs, for example row houses where the entire roof is attributed to one EGID. Attempt to 
    # 1 - add roof (perfectly overlapping roofpartitions) to solkat for all the EGIDs within the unions shape
    # 2- reduce the FLAECHE for all theses partitions by dividing it through the number of EGIDs in the union shape
    elif solkat_selection_specs_def['match_missing_EGIDs_to_solkat_TF']:
        print_to_logfile(f'\n\n Match missing EGIDs to solkat', log_file_name_def)
        print_to_logfile(f'\n\n Match missing EGIDs to solkat (where gwrEGIDs overlapp solkat shape but are not present as a single solkat_row)', summary_file_name)
        cols_adjust_for_missEGIDs_to_solkat = solkat_selection_specs_def['cols_adjust_for_missEGIDs_to_solkat']

        solkat_v2 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(bfs_number_def)])
        solkat_v2_wgeo = solkat_v2.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID')
        solkat_v2_gdf = gpd.GeoDataFrame(solkat_v2_wgeo, geometry='geometry')
        solkat_v2_gdf = solkat_v2_gdf[solkat_v2_gdf['EGID'] != 'NAN']
        

        # create mapping of solkatEGIDs and missing gwrEGIDs -------------------
        # union all shapes with the same EGID 
        solkat_union_v2EGID = solkat_v2_gdf.groupby('EGID').agg({
            'geometry': lambda x: unary_union(x),  # Combine all geometries into one MultiPolygon
            'DF_UID': lambda x: '_'.join(map(str, x))  # Concatenate DF_UID values as a single string
            }).reset_index()
        solkat_union_v2EGID = gpd.GeoDataFrame(solkat_union_v2EGID, geometry='geometry')
        
        # rename EGID colum because gwr_EGIDs are now matched to union_shapes
        solkat_union_v2EGID.rename(columns = {'EGID': 'EGID_old_solkat'}, inplace = True)
        # match gwrEGID through sjoin to solkat
        solkat_union_v2EGID.set_crs(gwr_gdf.crs, allow_override=True, inplace=True)
        join_gwr_solkat_union = gpd.sjoin(solkat_union_v2EGID, gwr_gdf, how='left')
        join_gwr_solkat_union.rename(columns = {'EGID': 'EGID_gwradded'}, inplace = True)


        # check EGID mapping case by case, add missing gwrEGIDs to solkat -------------------
        EGID_old_solkat_list = join_gwr_solkat_union['EGID_old_solkat'].unique()
        new_solkat_append_list = []
        add_solkat_counter = 1
        for n_egid, egid in enumerate(EGID_old_solkat_list):
            egid_join_union = join_gwr_solkat_union.loc[join_gwr_solkat_union['EGID_old_solkat'] == egid,]
            egid_join_union = egid_join_union.reset_index(drop = True)

            # Shapes of building that will not be included given GWR filter settings
            if egid_join_union['EGID_gwradded'].isna().any():  
                solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid])

            elif all(egid_join_union['EGID_gwradded'] != np.nan): 

                # a gwrEGID can be picked up by the union shape, but still be present in the solkat df, drop theses rows
                egid_to_add = egid_join_union['EGID_gwradded'].unique()[0]
                for egid_to_add in egid_join_union['EGID_gwradded'].unique():
                    if egid_join_union.shape[0] > 1:
                        if egid_to_add != egid:
                            if egid_to_add in EGID_old_solkat_list:
                                egid_join_union = egid_join_union.drop(egid_join_union.loc[egid_join_union['EGID_gwradded'] == egid_to_add].index)
                            elif egid_to_add == egid:
                                print('')


                # "Best" case (unless step above applies): Shapes of building that only has 1 GWR EGID
                if (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] == egid): 
                    solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                # Not best case but for consistency better to keep individual solkatEGIs matches (otherwise missmatch of newer buildings with old shape partitions possible)
                elif (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] != egid): 
                    solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])

                # Intended case: Shapes of building that has multiple GWR EGIDs within the shape boundaries
                elif (egid_join_union.shape[0] > 1) & (egid in egid_join_union['EGID_gwradded'].to_list()):
                    
                    solkat_subdf_addedEGID_list = []
                    n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                    for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):
                        
                        # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                        solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                        solkat_addedEGID['EGID'] = egid_to_add
                        
                        #extend the DF_UID with some numbers to have truely unique DF_UIDs
                        str_suffix = str(n+1).zfill(3)
                        solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}-{str_suffix}')

                        # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                        solkat_addedEGID[cols_adjust_for_missEGIDs_to_solkat] =  solkat_addedEGID[cols_adjust_for_missEGIDs_to_solkat] / egid_join_union['EGID_gwradded'].nunique()
                        
                        # shrink topology to see which partitions are affected by EGID extensions
                        solkat_addedEGID.buffer(-0.5, resolution=16)  

                        solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                    
                    # concat all EGIDs within the same shape that were previously missing
                    solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)
                    # checkpoint_to_logfile(f'EGID {egid}: added {n+1} gwrEGIDs to solkat', log_file_name_def, 5, show_debug_prints_def = show_debug_prints_def)

            if n_egid == int(len(EGID_old_solkat_list)/4):

                checkpoint_to_logfile(f'{add_solkat_counter}/4 part of solkat extended with gwrEGIDS', log_file_name_def, 3, show_debug_prints_def = show_debug_prints_def)
            # merge all solkat partitions to new solkat df
            new_solkat_append_list.append(solkat_subdf) 

        new_solkat_gdf = gpd.GeoDataFrame(pd.concat(new_solkat_append_list, ignore_index=True), geometry='geometry')
        new_solkat = new_solkat_gdf.drop(columns = ['geometry'])
        checkpoint_to_logfile(f'Extended solkat_df by {new_solkat.shape[0] - solkat_v2_gdf.shape[0]} rows (before matching: {solkat_v2_gdf.shape[0]}, after: {new_solkat.shape[0]} rows)', summary_file_name, 1, show_debug_prints_def = show_debug_prints_def)

        solkat, solkat_gdf = copy.deepcopy(new_solkat), copy.deepcopy(new_solkat_gdf)      
    


    # SOLKAT_MONTH ====================
    solkat_month_all_pq = pd.read_parquet(f'{data_path_def}/split_data_geometry/solkat_month_pq.parquet')
    checkpoint_to_logfile(f'import solkat_month_pq, {solkat_month_all_pq.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  1, show_debug_prints_def)

    # transformations
    solkat_month_all_pq = cols_to_str(['SB_UUID', 'DF_UID',], solkat_month_all_pq)
    solkat_month_all_pq = solkat_month_all_pq.merge(solkat_all_pq[['DF_UID', 'BFS_NUMMER']], how = 'left', on = 'DF_UID')
    solkat_month = solkat_month_all_pq[solkat_month_all_pq['BFS_NUMMER'].isin(bfs_number_def)]  

    # GRID_NODE ====================
    Map_egid_dsonode = pd.read_excel(f'{data_path_def}/input/Daten_Primeo_x_UniBasel_V2.0.xlsx')
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
    if solkat_selection_specs_def['test_loop_optim_buff_size_TF']: #
        print_to_logfile(f'\n\n Check different buffersizes!', log_file_name_def)
        arange_start, arange_end, arange_step = solkat_selection_specs_def['test_loop_optim_buff_arang'][0], solkat_selection_specs_def['test_loop_optim_buff_arang'][1], solkat_selection_specs_def['test_loop_optim_buff_arang'][2]
        buff_range = np.arange(arange_start, arange_end, arange_step)
        shares_xtf_duplicates = []
        for i in buff_range:# [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 2]:
            print_to_logfile(f'buffer size: {i}', log_file_name_def)

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
            
            print_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid_loop["EGID"].isna().sum() / gwregid_pvid_loop.shape[0] *100,2)} % of pv rows ({gwregid_pvid_loop.shape[0]}) are missing EGID', log_file_name_def)
            print_to_logfile(f'Duplicate shares: \tNANs\tunique\t2x\t>2x \n \t\t\t{shares[0]}\t{shares[1]}\t{shares[2]}\t{shares[3]}\t{sum(shares)}\n', log_file_name_def)
        
        # plot shares of successful mappings
        shares_xtf_duplicates_df = pd.DataFrame(shares_xtf_duplicates, columns = ['buffer_size', 'NANs', 'unique', '2x', '>2x'])
        # not plotted because over-exaggerated buffer is later corrected with closest neighbour matching
        # fig = px.line(shares_xtf_duplicates_df, 
        #               x='buffer_size', y=['NANs', 'unique', '2x', '>2x'],
        #               title = 'Shares of xtf_id duplicates', labels = {'buffer_size': 'Buffer Size', 'value': 'Share'}, width = 800, height = 400)
        # fig.show()
        # fig.write_html(f'{data_path_def}/output/preprep_data/by_buffersize_share_xtf_id_duplicates.html')
        checkpoint_to_logfile(f'buffer size optimisation finished', log_file_name_def, 2, show_debug_prints_def)


    # (continued MAP: egid > pv) ----------
    gwr_buff_gdf = copy.deepcopy(gwr_gdf)
    gwr_buff_gdf.set_crs("EPSG:32632", allow_override=True, inplace=True)
    gwr_buff_gdf['geometry'] = gwr_buff_gdf['geometry'].buffer(solkat_selection_specs_def['GWR_EGID_buffer_size'])
    gwr_buff_gdf, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, gwr_buff_gdf, pv_gdf)
    checkpoint_to_logfile(f'gwr_gdf.crs == pv_gdf.crs: {gwr_buff_gdf.crs == pv_gdf.crs}', log_file_name_def, 6, show_debug_prints_def)

    gwregid_pvid_all = gpd.sjoin(pv_gdf,gwr_buff_gdf, how="left", predicate="within")
    gwregid_pvid_all.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)

    # keep only unique xtf_ids 
    gwregid_pvid_unique = copy.deepcopy(gwregid_pvid_all.loc[~gwregid_pvid_all.duplicated(subset='xtf_id', keep=False)])
    xtf_duplicates =      copy.deepcopy(gwregid_pvid_all.loc[ gwregid_pvid_all.duplicated(subset='xtf_id', keep=False)])
    checkpoint_to_logfile(f'sum n_unique xtf_ids: {gwregid_pvid_unique["xtf_id"].nunique()} (unique df) +{xtf_duplicates["xtf_id"].nunique()} (duplicates df) = {gwregid_pvid_unique["xtf_id"].nunique()+xtf_duplicates["xtf_id"].nunique() }; n_unique in pv_gdf: {pv_gdf["xtf_id"].nunique()}', log_file_name_def, 6, show_debug_prints_def)
   
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
    checkpoint_to_logfile(f'total unique xtf: {pv_gdf["xtf_id"].nunique()} (pv_gdf); {gwregid_pvid_unique["xtf_id"].nunique()+xtf_nearestmatches_df["xtf_id"].nunique()} (unique + nearest match)', log_file_name_def, 6, show_debug_prints_def)

    checkpoint_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid["EGID"].isna().sum() / gwregid_pvid.shape[0] *100,2)} % of pv rows ({gwregid_pvid.shape[0]}) are missing EGID', log_file_name_def, 2, show_debug_prints_def)
    # Map_egid_pv = gwregid_pvid.loc[gwregid_pvid['EGID'].notna(), ['EGID', 'xtf_id']].copy()
    Map_egid_pv = gwregid_pvid[['EGID', 'xtf_id']].copy()



    # EXPORTS (parquet) ---------------------------------------------------------------------------------
    df_to_export_names = ['pv', 'solkat', 'solkat_month', 'Map_egid_dsonode', 'Map_solkatdfuid_egid', 'Map_egid_pv']
    df_to_export_list = [pv, solkat, solkat_month,  Map_egid_dsonode, Map_solkatdfuid_egid, Map_egid_pv] 
    for i, df in enumerate(df_to_export_list):
        df.to_parquet(f'{data_path_def}/output/preprep_data/{df_to_export_names[i]}.parquet')
        df.to_csv(f'{data_path_def}/output/preprep_data/{df_to_export_names[i]}.csv', sep=';', index=False)
        checkpoint_to_logfile(f'{df_to_export_names[i]} exported to prepreped data', log_file_name_def, 2, show_debug_prints_def)
    


    # OMITTED SPATIAL POINTS / POLYS ---------------------------------------------------------------------------------
    print_to_logfile(f'\nnumber of omitted buildings because EGID is (not) / present in all of GWR / Solkat / PV / Grid_Node', summary_file_name)
    print_to_logfile(f'>gwr settings: \n n bfs_numbers: {len(bfs_number_def)}, \n year_range: {year_range_def}, \n building class GKLAS: {gwr_selection_specs_def["GKLAS"]}, \n building status GSTAT: {gwr_selection_specs_def["GSTAT"]}, \n year of construction GBAUJ: {gwr_selection_specs_def["GBAUJ_minmax"]}', summary_file_name)
    omitt_gwregid_gdf = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(solkat_gdf['EGID'])])
    checkpoint_to_logfile(f'omitt_gwregid_gdf (gwr not in solkat): {omitt_gwregid_gdf.shape[0]} rows ({round((omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%), gwr[EGID].unique: {gwr_gdf["EGID"].nunique()})', summary_file_name, 2, True) 

    omitt_solkat_all_gwr_gdf = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_all_building_gdf['EGID'])])
    omitt_solkat_gdf = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_gdf['EGID'])])
    checkpoint_to_logfile(f'omitt_solkat_gdf (solkat not in gwr): {omitt_solkat_gdf.shape[0]} rows ({round((omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0])*100, 2)}%), solkat[EGID].unique: {solkat_gdf["EGID"].nunique()})', summary_file_name, 2, True)

    omitt_pv_gdf = copy.deepcopy(pv_gdf.loc[~pv_gdf['xtf_id'].isin(gwregid_pvid['xtf_id'])])
    checkpoint_to_logfile(f'omitt_pv_gdf (pv not in gwr): {omitt_pv_gdf.shape[0]} rows ({round((omitt_pv_gdf.shape[0]/pv_gdf.shape[0])*100, 2)}%, pv[xtf_id].unique: {pv_gdf["xtf_id"].nunique()})', summary_file_name, 2, True)

    omitt_gwregid_gridnode_gdf = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(Map_egid_dsonode['EGID'])])
    checkpoint_to_logfile(f'omitt_gwregid_gridnode_gdf (gwr not in gridnode): {omitt_gwregid_gridnode_gdf.shape[0]} rows ({round((omitt_gwregid_gridnode_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%), gwr[EGID].unique: {gwr_gdf["EGID"].nunique()})', summary_file_name, 2, True)

    omitt_gridnodeegid_gwr_df = copy.deepcopy(Map_egid_dsonode.loc[~Map_egid_dsonode['EGID'].isin(gwr_gdf['EGID'])])
    checkpoint_to_logfile(f'omitt_gridnodeegid_gwr_df (gridnode not in gwr): {omitt_gridnodeegid_gwr_df.shape[0]} rows ({round((omitt_gridnodeegid_gwr_df.shape[0]/Map_egid_dsonode.shape[0])*100, 2)}%), gridnode[EGID].unique: {Map_egid_dsonode["EGID"].nunique()})', summary_file_name, 2, True)

    

    # PRINTS TO SUMMARY LOG FILE ---------------------------------------------------------------------------------
    print_to_logfile(f'\n\nHow well does GWR cover other data sources', summary_file_name)
    checkpoint_to_logfile(f'gwr_EGID omitted in solkat: {round(omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0]*100, 2)} %', summary_file_name, 2, True)
    checkpoint_to_logfile(f'solkat_EGID omitted in gwr_all_bldng: {round(omitt_solkat_all_gwr_gdf.shape[0]/solkat_gdf.shape[0]*100, 2)} %', summary_file_name, 2, True)
    checkpoint_to_logfile(f'solkat_EGID omitted in gwr: {round(omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0]*100, 2)} %', summary_file_name, 2, True)
    checkpoint_to_logfile(f'pv_xtf_id omitted in gwr: {round(omitt_pv_gdf.shape[0]/pv_gdf.shape[0]*100, 2)} %', summary_file_name, 2, True)
    checkpoint_to_logfile(f'gwr_EGID omitted in gridnode: {round(omitt_gwregid_gridnode_gdf.shape[0]/gwr_gdf.shape[0]*100, 2)} %', summary_file_name, 2, True)
    checkpoint_to_logfile(f'gridnode_EGID omitted in gwr: {round(omitt_gridnodeegid_gwr_df.shape[0]/Map_egid_dsonode.shape[0]*100, 2)} %', summary_file_name, 2, True)


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

        print_to_logfile(f'CRS for {gdf_to_export_names[i]}: {g.crs}', log_file_name_def,)
        checkpoint_to_logfile(f'exported {gdf_to_export_names[i]}', log_file_name_def , 4, show_debug_prints_def)

        with open(f'{data_path_def}/output/preprep_data/{gdf_to_export_names[i]}.geojson', 'w') as f:
            f.write(g.to_json())



# ------------------------------------------------------------------------------------------------------
# IMPORT ELECTRICITY DEMAND TS + MATCH TO HOUSEHOLDS
# ------------------------------------------------------------------------------------------------------

def import_demand_TS_AND_match_households(
        dataagg_settings_def, ):
    """
    1) Import demand time series data and aggregate it to 4 demand archetypes.
    2) Match the time series to the households IDs dependent on building characteristics (e.g. flat/house size, electric heating, etc.)
       Export all the mappings and time series data.
    """

    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']

    gwr_selection_specs_def = dataagg_settings_def['gwr_selection_specs']
    demand_specs_def = dataagg_settings_def['demand_specs']

    print_to_logfile(f'run function: import_demand_TS_AND_match_households.py', log_file_name_def = log_file_name_def)


    # IMPORT CONSUMER DATA -----------------------------------------------------------------
    print_to_logfile(f'\nIMPORT CONSUMER DATA {10*"*"}', log_file_name_def = log_file_name_def) 
       

    # DEMAND DATA SOURCE: NETFLEX ============================================================
    if demand_specs_def['input_data_source'] == "NETFLEX" :
        # import demand TS --------
        netflex_consumers_list = os.listdir(f'{data_path_def}/input/NETFLEX_consumers')
        
        all_assets_list = []
        # c = netflex_consumers_list[1]
        for c in netflex_consumers_list:
            f = open(f'{data_path_def}/input/NETFLEX_consumers/{c}')
            data = json.load(f)
            assets = data['assets']['list'] 
            all_assets_list.extend(assets)
        
        without_id = [a.split('_ID')[0] for a in all_assets_list]
        all_assets_unique = list(set(without_id))
        checkpoint_to_logfile(f'consumer demand TS contains assets: {all_assets_unique}', log_file_name_def, 2, show_debug_prints_def)

        # aggregate demand for each consumer
        agg_demand_df = pd.DataFrame()
        netflex_consumers_list = netflex_consumers_list if not smaller_import_def else netflex_consumers_list[0:40]

        c = netflex_consumers_list[2]
        # for c, c_n in enumerate() netflex_consumers_list:
        for c_number, c in enumerate(netflex_consumers_list):
            c_demand_id, c_demand_tech, c_demand_asset, c_demand_t, c_demand_values = [], [], [], [], []

            f = open(f'{data_path_def}/input/NETFLEX_consumers/{c}')
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
            
            if (c_number + 1) % (len(netflex_consumers_list) // 4) == 0:
                checkpoint_to_logfile(f'exported demand TS for consumer {c}, {c_number+1} of {len(netflex_consumers_list)}', log_file_name_def, 2, show_debug_prints_def)
        
        # remove pv assets because they also have negative values
        agg_demand_df = agg_demand_df[agg_demand_df['tech'] != 'pv']

        agg_demand_df['value'] = agg_demand_df['value'] * 1000 # it appears that values are calculated in MWh, need kWh

        # plot TS for certain consumers by assets
        plot_ids =['ID100', 'ID101', 'ID102', ]
        plot_df = agg_demand_df[agg_demand_df['id'].isin(plot_ids)]
        fig = px.line(plot_df, x='t', y='value', color='asset', title='Demand TS for selected consumers')
        # fig.show()

        # export aggregated demand for all NETFLEX consumer assets
        agg_demand_df.to_parquet(f'{data_path_def}/output/preprep_data/demand_ts.parquet')
        checkpoint_to_logfile(f'exported demand TS for all consumers', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        

        # AGGREGATE DEMAND TYPES -----------------------------------------------------------------
        # aggregate demand TS for defined consumer types 
        # demand upper/lower 50 percentile, with/without heat pump
        # get IDs for each subcatergory
        print_to_logfile(f'\nAGGREGATE DEMAND TYPES {10*"*"}', log_file_name_def = log_file_name_def)
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

        demandtypes.to_parquet(f'{data_path_def}/output/preprep_data/demandtypes.parquet')
        demandtypes.to_csv(f'{data_path_def}/output/preprep_data/demandtypes.csv', sep=';', index=False)
        checkpoint_to_logfile(f'exported demand types', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

        # plot demand types with plotly
        fig = px.line(demandtypes, x='t', y=['high_DEMANDprox_wiHP', 'low_DEMANDprox_wiHP', 'high_DEMANDprox_noHP', 'low_DEMANDprox_noHP'], title='Demand types')
        # fig.show()
        fig.write_html(f'{data_path_def}/output/preprep_data/demandtypes.html')
        demandtypes['high_DEMANDprox_wiHP'].sum(), demandtypes['low_DEMANDprox_wiHP'].sum(), demandtypes['high_DEMANDprox_noHP'].sum(), demandtypes['low_DEMANDprox_noHP'].sum()


        # MATCH DEMAND TYPES TO HOUSEHOLDS -----------------------------------------------------------------
        print_to_logfile(f'\nMATCH DEMAND TYPES TO HOUSEHOLDS {10*"*"}', log_file_name_def = log_file_name_def)

        # import GWR and PV --------
        gwr_all = pd.read_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
        checkpoint_to_logfile(f'imported gwr data', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        
        # transformations
        gwr_all[gwr_selection_specs_def['DEMAND_proxy']] = pd.to_numeric(gwr_all[gwr_selection_specs_def['DEMAND_proxy']], errors='coerce')
        gwr_all['GBAUJ'] = pd.to_numeric(gwr_all['GBAUJ'], errors='coerce')
        gwr_all.dropna(subset = ['GBAUJ'], inplace = True)
        gwr_all['GBAUJ'] = gwr_all['GBAUJ'].astype(int)

        # selection based on GWR specifications -------- 
        # select columns GSTAT that are within list ['1110','1112'] and GKLAS in ['1234','2345']
        gwr = gwr_all[(gwr_all['GSTAT'].isin(gwr_selection_specs_def['GSTAT'])) & 
                    (gwr_all['GKLAS'].isin(gwr_selection_specs_def['GKLAS'])) & 
                    (gwr_all['GBAUJ'] >= gwr_selection_specs_def['GBAUJ_minmax'][0]) &
                    (gwr_all['GBAUJ'] <= gwr_selection_specs_def['GBAUJ_minmax'][1])]
        checkpoint_to_logfile(f'filtered vs unfiltered gwr: shape ({gwr.shape[0]} vs {gwr_all.shape[0]}), EGID.nunique ({gwr["EGID"].nunique()} vs {gwr_all ["EGID"].nunique()})', log_file_name_def, 2, show_debug_prints_def)
        
        def get_IDs_upper_lower_DEMAND_by_hp(df, DEMAND_col = gwr_selection_specs_def['DEMAND_proxy'],  hp_TF = True,  up_low50percent = "upper"):
            id_with_hp = df[df['GWAERZH1'].isin(gwr_selection_specs_def['GWAERZH'])]['EGID'].unique()
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
        print_to_logfile(f'sanity check gwr classifications', log_file_name_def = log_file_name_def)
        gwr_egid_list = gwr['EGID'].tolist()
        gwr_classified_list = [high_DEMANDprox_wiHP_list, low_DEMANDprox_wiHP_list, high_DEMANDprox_noHP_list, low_DEMANDprox_noHP_list]
        gwr_classified_names= ['high_DEMANDprox_wiHP_list', 'low_DEMANDprox_wiHP_list', 'high_DEMANDprox_noHP_list', 'low_DEMANDprox_noHP_list']

        for chosen_lst_idx, chosen_list in enumerate(gwr_classified_list):
            chosen_set = set(chosen_list)

            for i, lst in enumerate(gwr_classified_list):
                if i != chosen_lst_idx:
                    other_set = set(lst)
                    common_ids = chosen_set.intersection(other_set)
                    print_to_logfile(f"No. of common IDs between {gwr_classified_names[chosen_lst_idx]} and {gwr_classified_names[i]}: {len(common_ids)}", log_file_name_def = log_file_name_def)
            print_to_logfile(f'\n', log_file_name_def = log_file_name_def)

        # precent of classified buildings
        n_classified = sum([len(lst) for lst in gwr_classified_list])
        n_all = len(gwr['EGID'])
        print_to_logfile(f'{n_classified} of {n_all} ({round(n_classified/n_all*100, 2)}%) gwr rows are classfied', log_file_name_def = log_file_name_def)
        

        # export to JSON --------
        Map_demandtype_EGID ={
            'high_DEMANDprox_wiHP': high_DEMANDprox_wiHP_list,
            'low_DEMANDprox_wiHP': low_DEMANDprox_wiHP_list,
            'high_DEMANDprox_noHP': high_DEMANDprox_noHP_list,
            'low_DEMANDprox_noHP': low_DEMANDprox_noHP_list,
        }
        with open(f'{data_path_def}/output/preprep_data/Map_demandtype_EGID.json', 'w') as f:
            json.dump(Map_demandtype_EGID, f)
        checkpoint_to_logfile(f'exported Map_demandtype_EGID.json', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

        Map_EGID_demandtypes = {}
        for type, egid_list in Map_demandtype_EGID.items():
            for egid in egid_list:
                Map_EGID_demandtypes[egid] = type
        with open(f'{data_path_def}/output/preprep_data/Map_EGID_demandtypes.json', 'w') as f:
            json.dump(Map_EGID_demandtypes, f)
        checkpoint_to_logfile(f'exported Map_EGID_demandtypes.json', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)


    # DEMAND DATA SOURCE: SwissStore ============================================================
    elif demand_specs_def['input_data_source'] == "SwissStore" :
        print("STUCK")     # follow up call with Hector. => for match all houses to archetypes of Swisstore and then later extract demand profile
        
        swstore_demand_inclnan = pd.read_excel(f'{data_path_def}/input/SwissStore_DemandData/Electricity_demand_SFH_MFH.xlsx')
        swstore_demand = swstore_demand_inclnan.loc[~swstore_demand_inclnan['time'].isna()]
        swstore_demand['SFH'].sum(), swstore_demand['MFH'].sum()
        swstore_demand.head()
        swstore_demand.shape


        os.listdir(f'{data_path_def}/input/SwissStore_DemandData/Electricity_demand_SFH_MFH.xlsx')


# ------------------------------------------------------------------------------------------------------
# IMPORT METEO DATA
# ------------------------------------------------------------------------------------------------------

def import_meteo_data(
        dataagg_settings_def, ):
    """
    Import meteo data from a source, select only the relevant time frame store data to prepreped data folder.
    """
    
    # import settings + setup --------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']

    gwr_selection_specs_def = dataagg_settings_def['gwr_selection_specs']
    print_to_logfile(f'run function: import_demand_TS_AND_match_households.py', log_file_name_def = log_file_name_def)



    # IMPORT METEO DATA ============================================================================
    print_to_logfile(f'\nIMPORT METEO DATA {10*"*"}', log_file_name_def = log_file_name_def)

    # import meteo data --------
    meteo = pd.read_csv(f'{data_path_def}/input/Meteoblue_BSBL/Meteodaten_Basel_2018_2024_reduziert_bereinigt.csv')

    # transformations
    meteo['timestamp'] = pd.to_datetime(meteo['timestamp'], format = '%d.%m.%Y %H:%M:%S')

    # select relevant time frame
    start_stamp = pd.to_datetime(f'01.01.{year_range_def[0]}', format = '%d.%m.%Y')
    end_stamp = pd.to_datetime(f'31.12.{year_range_def[1]}', format = '%d.%m.%Y')
    meteo = meteo[(meteo['timestamp'] >= start_stamp) & (meteo['timestamp'] <= end_stamp)]
    
    # export --------
    meteo.to_parquet(f'{data_path_def}/output/preprep_data/meteo.parquet')
    checkpoint_to_logfile(f'exported meteo data', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

    # MATCH WEATHER STATIONS TO HOUSEHOLDS ============================================================================


