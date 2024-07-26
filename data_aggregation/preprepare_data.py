import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import winsound
from datetime import datetime
from shapely.geometry import Point

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# IMPORT LOCAL DATA + create SPATIAL MAPPINGS
# ------------------------------------------------------------------------------------------------------

def local_data_to_parquet_AND_create_spatial_mappings(
        dataagg_settings_def, ):
    """
    1) Function to import all the local data sources, remove and transform data where necessary and store the prepared data as parquet file. 
    2) When applicable, create mapping files, so that spatial data can be reidentified to their geometry if necessary. 
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
    print_to_logfile(f'run function: local_data_to_parquet_AND_create_spatial_mappings.py', log_file_name_def = log_file_name_def)

    # import sys
    # if not script_run_on_server_def:
    #     sys.path.append('C:/Models/OptimalPV_RH') 
    # elif script_run_on_server_def:
    #     sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')
    # import auxiliary_functions
    # from auxiliary_functions import chapter_to_logfile, checkpoint_to_logfile, print_to_logfile



    # IMPORT DATA AND STORE TO PARQUET ============================================================================
    print_to_logfile(f'\nIMPORT DATA AND STORE TO PARQUET {10*"*"}', log_file_name_def = log_file_name_def) 
    if True: 
        gm_shp_gdf = gpd.read_file(f'{data_path_def}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
        
        # Function: Merge GM BFS numbers to spatial data sources
        def attach_bfs_to_spatial_data(gdf, gm_shp_gdf, keep_cols = ['BFS_NUMMER', 'geometry' ]):
            """
            Function to attach BFS numbers to spatial data sources
            """
            gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
            gdf = gpd.sjoin(gdf, gm_shp_gdf, how="left", predicate="within")
            checkpoint_to_logfile('sjoin complete', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
            dele_cols = ['index_right'] + [col for col in gm_shp_gdf.columns if col not in keep_cols]
            gdf.drop(columns = dele_cols, inplace = True)

            return gdf 

        # SOLAR KATASTER --------------------------------------------------------------------
        if not smaller_import_def:  
            solkat_all_gdf = gpd.read_file(f'{data_path_def}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        elif smaller_import_def:
            solkat_all_gdf = gpd.read_file(f'{data_path_def}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 1000)
        checkpoint_to_logfile(f'import solkat, {solkat_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # attach bfs
        solkat_all_gdf = attach_bfs_to_spatial_data(solkat_all_gdf, gm_shp_gdf, )

        # drop unnecessary columns ------------------
        # transformations ------------------
        solkat_all_gdf = solkat_all_gdf.rename(columns={'GWR_EGID': 'EGID'})
        solkat_all_gdf['EGID'] = solkat_all_gdf['EGID'].replace('nan', np.nan)  # convert EGID into numbers with no decimal points
        solkat_all_gdf['EGID'] = solkat_all_gdf['EGID'].astype('Int64')
        

        # filter by bfs_nubmers_def ------------------
        solkat_gdf = solkat_all_gdf[solkat_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]

        # export ------------------
        solkat_gdf.to_parquet(f'{data_path_def}/output/preprep_data/solkat.parquet')
        solkat_gdf.to_csv(f'{data_path_def}/output/preprep_data/solkat.csv', sep=';', index=False)
        print_to_logfile(f'exported solkat data', log_file_name_def = log_file_name_def)


        # HEATING + COOLING DEMAND ---------------------------------------------------------------------
        if not smaller_import_def:
            heat_all_gdf = gpd.read_file(f'{data_path_def}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
        elif smaller_import_def:
            heat_all_gdf = gpd.read_file(f'{data_path_def}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 100)
        checkpoint_to_logfile(f'import heat, {heat_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # attach bfs
        heat_all_gdf = attach_bfs_to_spatial_data(heat_all_gdf, gm_shp_gdf)

        # drop unnecessary columns ------------------
        # transformations ------------------

        # filter by bfs_nubmers_def ------------------
        heat_gdf = heat_all_gdf[heat_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]
        
        # export ------------------
        heat_gdf.to_parquet(f'{data_path_def}/output/preprep_data/heat.parquet')
        heat_gdf.to_csv(f'{data_path_def}/output/preprep_data/heat.csv', sep=';', index=False)
        print_to_logfile(f'exported heat data', log_file_name_def = log_file_name_def)


        # PV ---------------------------------------------------------------------
        if not smaller_import_def:
            elec_prod_gdf = gpd.read_file(f'{data_path_def}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
            pv_all_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        elif smaller_import_def:
            elec_prod_gdf = gpd.read_file(f'{data_path_def}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 1000)
            pv_all_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile(f'import pv, {pv_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # attach bfs
        pv_all_gdf = attach_bfs_to_spatial_data(pv_all_gdf, gm_shp_gdf)

        # drop unnecessary columns ------------------
        # transformations ------------------

        # filter by bfs_nubmers_def ------------------
        pv_gdf = pv_all_gdf[pv_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]

        # export ------------------
        pv_gdf.to_parquet(f'{data_path_def}/output/preprep_data/pv.parquet')
        pv_gdf.to_csv(f'{data_path_def}/output/preprep_data/pv.csv', sep=';', index=False)
        print_to_logfile(f'exported pv data', log_file_name_def = log_file_name_def)



    # MAPPINGS & SPATIAL MAPPIGNS ============================================================================
    print_to_logfile(f'MAPPINGS & SPATIAL MAPPIGNS {10*"*"}', log_file_name_def = log_file_name_def)
    # if True: 

    # only keep certain cols and remove those that are not relevant for spatial/geographic mapping
    def keep_columns (col_names, gdf):
        keep_cols = col_names
        dele_cols = [col for col in gdf.columns if col not in keep_cols]
        gdf.drop(columns = dele_cols, inplace = True)
        return gdf
    
    def set_crs_to_gm_shp(gdf_CRS, gdf_a, gdf_b = None):
        gdf_a.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
        if gdf_b is not None:
            gdf_b.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
        
        if gdf_b is None: 
            return gdf_a
        if gdf_b is not None:
            return gdf_a, gdf_b
          
    # Create House Shapes ---------------------------------------------------------------------
    solkat_gdf_mapping = solkat_gdf.copy()
    solkat_gdf_mapping = set_crs_to_gm_shp(gm_shp_gdf, solkat_gdf_mapping)
    # solkat_gdf_mapping = keep_columns(['SB_UUID', 'EGID', 'DF_NUMMER', 'geometry'], solkat_gdf_mapping)
    solkat_gdf_mapping = solkat_gdf_mapping.loc[:,['SB_UUID', 'EGID', 'DF_NUMMER', 'geometry']]
    solkat_union_srs = solkat_gdf_mapping.groupby('EGID')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
    solkat_egidunion = gpd.GeoDataFrame(solkat_union_srs, geometry='geometry')


    # MAP: solkat_egid > solkat_sbuuid ---------------------------------------------------------------------
    Map_solkategid_sbuuid = solkat_gdf_mapping[['SB_UUID', 'EGID']].drop_duplicates().copy()
    Map_solkategid_sbuuid.dropna(subset = ['EGID'], inplace = True)
    Map_solkategid_sbuuid = Map_solkategid_sbuuid.sort_values(by = ['EGID', 'SB_UUID'])

    Map_solkategid_sbuuid.to_parquet(f'{data_path_def}/output/preprep_data/Map_solkategid_sbuuid.parquet')
    Map_solkategid_sbuuid.to_csv(f'{data_path_def}/output/preprep_data/Map_solkategid_sbuuid.csv', sep=';', index=False)
    checkpoint_to_logfile(f'exported Map_egid_sbuuid', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

    # MAP: solkat_egid > pv id ---------------------------------------------------------------------
    solkat_egidunion.reset_index(inplace = True)
    solkat_egidunion, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, solkat_egidunion, pv_gdf)
    Map_solkategid_pv = gpd.sjoin(solkat_egidunion, pv_gdf, how="left", predicate="within")
    # Map_solkategid_pv = keep_columns(['EGID','xtf_id', ], Map_solkategid_pv)
    Map_solkategid_pv = Map_solkategid_pv.loc[:,['EGID','xtf_id', ]]

    Map_solkategid_pv.to_parquet(f'{data_path_def}/output/preprep_data/Map_solkategid_pv.parquet')
    Map_solkategid_pv.to_csv(f'{data_path_def}/output/preprep_data/Map_solkategid_pv.csv', sep=';', index=False)
    checkpoint_to_logfile(f'exported Map_egid_pv', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

    # MAP: solkat_egid > heat id ---------------------------------------------------------------------
    solkat_egidunion.reset_index(inplace = True)
    solkat_egidunion, heat_gdf = set_crs_to_gm_shp(gm_shp_gdf, solkat_egidunion, heat_gdf)
    Map_solkategid_heat = gpd.sjoin(solkat_egidunion, heat_gdf, how="left", predicate="within")
    # Map_solkategid_heat = keep_columns(['EGID','NEEDHOME', ], Map_solkategid_heat)
    Map_solkategid_heat = Map_solkategid_heat.loc[:,['EGID','NEEDHOME', ]]

    Map_solkategid_heat.to_parquet(f'{data_path_def}/output/preprep_data/Map_solkategid_heat.parquet')
    Map_solkategid_heat.to_csv(f'{data_path_def}/output/preprep_data/Map_solkategid_heat.csv', sep=';', index=False)
    checkpoint_to_logfile(f'exported Map_egid_heat', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

    # MAP: solkat_egidunion > geometry ---------------------------------------------------------------------
    GEOM_solkat_union = solkat_egidunion.copy()
    GEOM_solkat_union.to_file(f'{data_path_def}/output/preprep_data/GEOM_solkat_union.geojson', driver='GeoJSON')

    # MAP: pv > geometry ---------------------------------------------------------------------
    # GEOM_pv = keep_columns(['xtf_id', 'geometry'], pv_gdf).copy()
    GEOM_pv = pv_gdf.loc[:,['xtf_id', 'geometry']]
    GEOM_pv.to_file(f'{data_path_def}/output/preprep_data/GEOM_pv.geojson', driver='GeoJSON')

    # MAP: heat > geometry ---------------------------------------------------------------------
    # GEOM_heat = keep_columns(['NEEDHOME', 'geometry'], heat_gdf).copy()
    GEOM_heat = heat_gdf.loc[:,['NEEDHOME', 'geometry']]
    GEOM_heat.to_file(f'{data_path_def}/output/preprep_data/GEOM_heat.geojson', driver='GeoJSON')




