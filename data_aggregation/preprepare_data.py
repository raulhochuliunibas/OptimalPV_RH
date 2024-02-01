import sys
import os as os
import pandas as pd
import geopandas as gpd
import winsound
from datetime import datetime
from shapely.geometry import Point

sys.path.append('..')
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# SPATIAL DATA TO PARQUET BY GM
# ------------------------------------------------------------------------------------------------------

# SOLAR KATASTER ---------------------------------------------------------------------
def solkat_spatial_toparquet(
    script_run_on_server_def = False,
    smaller_import_def = False,
    log_file_name_def = None,
    wd_path_def = None,
    data_path_def = None,
    show_debug_prints_def = None, 
    ):
    """
    Function to intersect spatial data with gm_shp and export to parquet
    """

    # setup -------------------
    wd_path = wd_path_def if wd_path_def else "C:/Models/OptimalPV_RH"
    data_path = data_path_def if data_path_def else f'{wd_path}_data'

    import sys
    if not script_run_on_server_def:
        sys.path.append('C:/Models/OptimalPV_RH') 
    elif script_run_on_server_def:
        sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')
    import functions
    from functions import chapter_to_logfile, checkpoint_to_logfile, print_to_logfile

    # set directory if necessary    
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')

    # create first checkpoint
    checkpoint_to_logfile('run function: solkat_spatial_toparquet.py', log_file_name_def=log_file_name_def, n_tabs_def = 5) 


    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import_def:
        solkat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        checkpoint_to_logfile('import solkat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    elif smaller_import_def:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        solkat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 1000)
        checkpoint_to_logfile('import solkat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        
    # sjoin to gm_shp ------------------
    solkat_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    solkat = gpd.sjoin(solkat_gdf, gm_shp_gdf, how="left", predicate="within")
    checkpoint_to_logfile('sjoin solkat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # drop unnecessary columns ------------------
    keep_cols = ["BFS_NUMMER", ]
    dele_cols = ['index_right'] + [col for col in gm_shp_gdf.columns if col not in keep_cols]
    solkat.drop(columns = dele_cols, inplace = True)

    # export ------------------
    solkat.to_parquet(f'{data_path}/output/preprep_data/solkat_by_gm.parquet')
    checkpoint_to_logfile('export solkat.parquet', log_file_name_def = log_file_name_def, n_tabs_def = 5)
    print_to_logfile(f'\n', log_file_name_def = log_file_name_def)


# HEATING + COOLING DEMAND ---------------------------------------------------------------------
def heat_spatial_toparquet(
    script_run_on_server_def = False,
    smaller_import_def = False,
    log_file_name_def = None,
    wd_path_def = None,
    data_path_def = None,
    show_debug_prints_def = None,
    ):
    """
    Function to intersect spatial data with gm_shp and export to parquet
    """

    # setup -------------------
    wd_path = wd_path_def if wd_path_def else "C:/Models/OptimalPV_RH"
    data_path = data_path_def if data_path_def else f'{wd_path}_data'

    import sys
    if not script_run_on_server_def:
        sys.path.append('C:/Models/OptimalPV_RH')
    elif script_run_on_server_def:
        sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')
    import functions
    from functions import chapter_to_logfile, checkpoint_to_logfile, print_to_logfile

    # set directory if necessary
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')

    # create first checkpoint
    checkpoint_to_logfile('run function: heat_spatial_toparquet.py', log_file_name_def=log_file_name_def, n_tabs_def = 5)
    
    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import_def:
        heat_gdf = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')       
        checkpoint_to_logfile('import heat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    elif smaller_import_def:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def=show_debug_prints_def)
        heat_gdf = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 10)
        checkpoint_to_logfile('import heat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    
    # sjoin to gm_shp ------------------
    heat_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    heat = gpd.sjoin(heat_gdf, gm_shp_gdf, how="left", predicate="within")
    checkpoint_to_logfile('sjoin heat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # drop unnecessary columns ------------------
    keep_cols = ["BFS_NUMMER", ]
    dele_cols = ['index_right'] + [col for col in gm_shp_gdf.columns if col not in keep_cols]
    heat.drop(columns = dele_cols, inplace = True)

    # export ------------------
    heat.to_parquet(f'{data_path}/output/preprep_data/heat_by_gm.parquet')
    checkpoint_to_logfile('export heat.parquet', log_file_name_def = log_file_name_def, n_tabs_def = 5)
    print_to_logfile(f'\n', log_file_name_def = log_file_name_def)


# PV ---------------------------------------------------------------------
def pv_spatial_toparquet(
    script_run_on_server_def = False,
    smaller_import_def = False,
    log_file_name_def = None,
    wd_path_def = None,
    data_path_def = None,
    show_debug_prints_def = None,
    ):
    """
    Function to intersect spatial data with gm_shp and export to parquet
    """

    # setup -------------------
    wd_path = wd_path_def if wd_path_def else "C:/Models/OptimalPV_RH"
    data_path = data_path_def if data_path_def else f'{wd_path}_data'

    import sys
    if not script_run_on_server_def:
        sys.path.append('C:/Models/OptimalPV_RH')
    elif script_run_on_server_def:
        sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')
    import functions
    from functions import chapter_to_logfile, checkpoint_to_logfile, print_to_logfile

    # set directory if necessary
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')

    # create first checkpoint
    checkpoint_to_logfile('run function: pv_spatial_toparquet.py', log_file_name_def=log_file_name_def, n_tabs_def = 5)

    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import_def:
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        checkpoint_to_logfile('import elec_prod', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    elif smaller_import_def:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 1000)
        checkpoint_to_logfile('import elec_prod', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # sjoin to gm_shp ------------------
    pv_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    pv = gpd.sjoin(pv_gdf, gm_shp_gdf, how="left", predicate="within")
    checkpoint_to_logfile('sjoin pv', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # drop unnecessary columns ------------------
    keep_cols = ["BFS_NUMMER", ]
    dele_cols = ['index_right'] + [col for col in gm_shp_gdf.columns if col not in keep_cols]
    pv.drop(columns = dele_cols, inplace = True)

    # export ------------------
    pv.to_parquet(f'{data_path}/output/preprep_data/pv_by_gm.parquet')
    checkpoint_to_logfile('export pv.parquet', log_file_name_def = log_file_name_def, n_tabs_def = 5)
    print_to_logfile(f'\n', log_file_name_def = log_file_name_def)


# ------------------------------------------------------------------------------------------------------
# MAPPINGS FOR SPATIAL DATA
# ------------------------------------------------------------------------------------------------------

# MAP ROOF PV ---------------------------------------------------------------------
def create_spatial_mappings(
    script_run_on_server_def = False,
    smaller_import_def = False,
    log_file_name_def = None,
    wd_path_def = None,
    data_path_def = None,
    show_debug_prints_def = None,
    ):
    """
    Function to create a mapping from solkat to pv, SB_UUID to xtf_id but also other data sources
    """    

    # setup -------------------
    wd_path = wd_path_def if wd_path_def else "C:/Models/OptimalPV_RH"
    data_path = data_path_def if data_path_def else f'{wd_path}_data'

    import sys
    if not script_run_on_server_def:
        sys.path.append('C:/Models/OptimalPV_RH')
    elif script_run_on_server_def:
        sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')
    import functions
    from functions import chapter_to_logfile, checkpoint_to_logfile, print_to_logfile

    # set directory if necessary
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')

    # create first checkpoint
    checkpoint_to_logfile('run function: create_spatial_mappings.py', log_file_name_def=log_file_name_def, n_tabs_def = 5)

    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import_def:
        solkat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        checkpoint_to_logfile('import solkat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        heat_gdf = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
        checkpoint_to_logfile('import heat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        checkpoint_to_logfile('import elec_prod', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()

        # gwr_gdf = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
        # checkpoint_to_logfile('import gwr', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        gwr = pd.read_parquet(f'{data_path}/output/preprep_data/gwr.parquet')
        
    elif smaller_import_def:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        solkat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 1000)
        heat_gdf = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 100)
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 1000)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('import data SMALLER IMPORT', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        gwr = pd.read_parquet(f'{data_path}/output/preprep_data/gwr.parquet')
        gwr = gwr.head(1000)
    
    # changes ------------------
    gwr['GKODE'] = pd.to_numeric(gwr['GKODE'], errors='coerce')
    gwr['GKODN'] = pd.to_numeric(gwr['GKODN'], errors='coerce')
    gwr_gdf = gpd.GeoDataFrame(gwr, geometry=gpd.points_from_xy(gwr['GKODE'], gwr['GKODN']))

    solkat_gdf = solkat_gdf.rename(columns={'GWR_EGID': 'EGID'})
        

    # drop unnecessary columns ------------------
    def keep_columns (col_names, gdf):
        keep_cols = col_names
        dele_cols = [col for col in gdf.columns if col not in keep_cols]
        gdf.drop(columns = dele_cols, inplace = True)
        return gdf

    solkat_gdf = keep_columns(["SB_UUID", "EGID", "DF_NUMMER", "geometry" ], solkat_gdf)
    gwr_gdf = keep_columns(["EGID", "geometry" ], gwr_gdf)
    heat_gdf = keep_columns(["OBJECTID", "geometry" ], heat_gdf)
    pv_gdf = keep_columns(["xtf_id", "geometry"], pv_gdf)

    # create ID mapping files ------------------
    def set_crs_to_gm_shp(gdf_a, gdf_b):
        gdf_a.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
        gdf_b.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
        return gdf_a, gdf_b
    
    def drop_index_right_cols(gdf):
        gdf.drop(columns = ['geometry', 'index_right'], inplace = True)
        return gdf
    

    # prepare Mappings ------------------
    # create house shapes 
    solkat_union_srs = solkat_gdf.groupby('SB_UUID')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
    solkat_union = gpd.GeoDataFrame(solkat_union_srs, geometry='geometry')
    checkpoint_to_logfile('created solkat_union', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    gwr_union_srs = solkat_gdf.groupby('EGID')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
    gwr_union = gpd.GeoDataFrame(gwr_union_srs, geometry='geometry')
    checkpoint_to_logfile('created gwr_union', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    
    # GWR_EGID SOLKAT to X Mappings ------------------
    # map eg_solkat > sb_solkat
    unique_combo = solkat_gdf[["SB_UUID", "EGID"]].drop_duplicates()
    unique_combo.dropna(subset=['EGID'], inplace=True)
    Map_egroof_sbroof = unique_combo
    Map_egroof_sbroof['EGID'] = Map_egroof_sbroof['EGID'].astype(int).astype(str)
    # Map_egroof_sbroof.reset_index(inplace = True)

    checkpoint_to_logfile('created Map_egroof_sbroof', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # map eg_solkat > PV
    gwr_union, pv_gdf = set_crs_to_gm_shp(gwr_union, pv_gdf)
    Map_egroof_pv = gpd.sjoin(gwr_union, pv_gdf, how="left", predicate="intersects")
    Map_egroof_pv = drop_index_right_cols(Map_egroof_pv)
    Map_egroof_pv.reset_index(inplace = True)
    Map_egroof_pv['EGID'] = Map_egroof_pv['EGID'].astype(int).astype(str)
    Map_egroof_pv['xtf_id'] = Map_egroof_pv['xtf_id'].astype(int).astype(str)
    checkpoint_to_logfile('created Map_egroof_pv', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # map eg_solkat > heat
    gwr_union, heat_gdf = set_crs_to_gm_shp(gwr_union, heat_gdf)
    Map_egroof_heat = gpd.sjoin(gwr_union, heat_gdf, how="left", predicate="intersects")
    Map_egroof_heat = drop_index_right_cols(Map_egroof_heat)
    Map_egroof_heat.reset_index(inplace = True)
    Map_egroof_heat['EGID'] = Map_egroof_heat['EGID'].astype(int).astype(str)
    Map_egroof_heat['OBJECTID'] = Map_egroof_heat['OBJECTID'].astype(int).astype(str)
    checkpoint_to_logfile('created Map_egroof_heat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)


    # GEOMETRY mapping files ------------------
    Map_sbroof_geom = solkat_union
    Map_sbroof_geom.reset_index(inplace = True)
    
    Map_egroof_geom = gwr_union
    Map_egroof_geom.reset_index(inplace = True)
    
    Map_solkat_geom = solkat_gdf[['SB_UUID', 'geometry']]
    Map_gwr_geom = gwr_gdf[['EGID', 'geometry']]    
    Map_heat_geom = heat_gdf[['OBJECTID', 'geometry']]
    Map_pv_geom = pv_gdf[['xtf_id', 'geometry']]

    # SB_UUID SOLKAT to X Mappings ------------------
    if False:

        # sjoin map SOLKAT > PV
        solkat_union, pv_gdf = set_crs_to_gm_shp(solkat_union, pv_gdf)
        Map_roof_pv = gpd.sjoin(solkat_union, pv_gdf, how="left", predicate="intersects")
        Map_roof_pv = drop_index_right_cols(Map_roof_pv)
        Map_roof_pv.reset_index(inplace = True)
        checkpoint_to_logfile(f'created Map_roof_pv', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # sjoin SOLKAT > GWR 
        solkat_union, gwr_gdf = set_crs_to_gm_shp(solkat_union, gwr_gdf)
        Map_roof_gwr = gpd.sjoin(solkat_union, gwr_gdf, how="left", predicate="intersects")
        Map_roof_gwr = drop_index_right_cols(Map_roof_gwr)
        Map_roof_gwr.reset_index(inplace = True)
        checkpoint_to_logfile(f'created Map_roof_gwr', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # sjoin SOLKAT > HEAT
        solkat_union, heat_gdf = set_crs_to_gm_shp(solkat_union, heat_gdf)
        Map_roof_heat = gpd.sjoin(solkat_union, heat_gdf, how="left", predicate="intersects")
        Map_roof_heat = drop_index_right_cols(Map_roof_heat)
        Map_roof_heat.reset_index(inplace = True)
        checkpoint_to_logfile(f'created Map_roof_heat', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

        # sjoin SOLKAT > GM 
        solkat_union, gm_shp_gdf = set_crs_to_gm_shp(solkat_union, gm_shp_gdf)
        Map_roof_gm = gpd.sjoin(solkat_union, gm_shp_gdf, how="left", predicate="intersects")
        Map_roof_gm = drop_index_right_cols(Map_roof_gm)
        Map_roof_gm.reset_index(inplace = True)
        checkpoint_to_logfile(f'created Map_roof_gm', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    
    # transform all Mappings to str ------------------
    

    Map_egroof_sbroof.to_parquet(f'{data_path}/output/preprep_data/Map_egroof_sbroof.parquet')
    Map_egroof_pv.to_parquet(f'{data_path}/output/preprep_data/Map_egroof_pv.parquet')
    Map_egroof_heat.to_parquet(f'{data_path}/output/preprep_data/Map_egroof_heat.parquet')

    Map_egroof_sbroof.to_csv(f'{data_path}/output/preprep_data/Map_egroof_sbroof.csv')
    Map_egroof_pv.to_csv(f'{data_path}/output/preprep_data/Map_egroof_pv.csv')
    Map_egroof_heat.to_csv(f'{data_path}/output/preprep_data/Map_egroof_heat.csv')

    Map_egroof_geom.to_file(f'{data_path}/output/preprep_data/Map_egroof_geom.geojson', driver='GeoJSON')
    Map_sbroof_geom.to_file(f'{data_path}/output/preprep_data/Map_sbroof_geom.geojson', driver='GeoJSON')
    Map_solkat_geom.to_file(f'{data_path}/output/preprep_data/Map_solkat_geom.geojson', driver='GeoJSON')
    # export ------------------
    checkpoint_to_logfile('exported all Map*.parquet files', log_file_name_def = log_file_name_def, n_tabs_def = 5)



##############################################################################################################################
##############################################################################################################################
# No Longer Used, to be deleted in April 2024
#   V   V   V
##############################################################################################################################

# BUILDING + DWELLING REGISTERY ---------------------------------------------------------------------
def gwr_spatial_toparquet(
    script_run_on_server_def = False,
    smaller_import_def = False,
    log_file_name_def = None,
    wd_path_def = None,
    data_path_def = None,
    show_debug_prints_def = None,
    ):
    """
    Function to intersect spatial data with gm_shp and export to parquet
    """

    # setup -------------------
    wd_path = wd_path_def if wd_path_def else "C:/Models/OptimalPV_RH"
    data_path = data_path_def if data_path_def else f'{wd_path}_data'

    import sys
    if not script_run_on_server_def:
        sys.path.append('C:/Models/OptimalPV_RH')
    elif script_run_on_server_def:
        sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')
    import functions
    from functions import chapter_to_logfile, checkpoint_to_logfile, print_to_logfile

    # set directory if necessary
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')

    # create first checkpoint
    checkpoint_to_logfile('run function: gwr_spatial_toparquet.py', log_file_name_def=log_file_name_def, n_tabs_def = 5)
    
    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import_def:
        # gwr_gdf = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
        gwr_pq = pd.read_parquet(f'{data_path}/output/preprep_data/gwr.parquet')  
        checkpoint_to_logfile('import gwr', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    elif smaller_import_def:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
        # gwr_gdf = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson', rows = 1000)
        gwr_pq = pd.read_parquet(f'{data_path}/output/preprep_data/gwr.parquet')  
        checkpoint_to_logfile('import gwr', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    
    # create gdf ------------------
    gwr_pq['GKODE'] = pd.to_numeric(gwr_pq['GKODE'], errors='coerce')
    gwr_pq['GKODN'] = pd.to_numeric(gwr_pq['GKODN'], errors='coerce')

    gwr_gdf = gpd.GeoDataFrame(gwr_pq, geometry=gpd.points_from_xy(gwr_pq['GKODE'], gwr_pq['GKODN']))


    # sjoin to gm_shp ------------------
    gwr_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    gwr = gpd.sjoin(gwr_gdf, gm_shp_gdf, how="left", predicate="within")
    checkpoint_to_logfile('sjoin gwr', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # drop unnecessary columns ------------------
    keep_cols = ["BFS_NUMMER", ]
    dele_cols = ['index_right'] + [col for col in gm_shp_gdf.columns if col not in keep_cols]
    gwr.drop(columns = dele_cols, inplace = True)

    # export ------------------
    gwr.to_parquet(f'{data_path}/output/preprep_data/gwr_by_gm.parquet')
    checkpoint_to_logfile('export gwr.parquet', log_file_name_def = log_file_name_def, n_tabs_def = 5)
    print_to_logfile(f'\n', log_file_name_def = log_file_name_def)


##############################################################################################################################
##############################################################################################################################
# No Longer Used, to be deleted in March 2024
#   V   V   V
##############################################################################################################################

def create_Mappings(
    script_run_on_server_def = 0,
    # buffer_size = [0.05,],
    smaller_import = False,):
    """
    Function to create a mapping from solkat to pv, SB_UUID to xtf_id
    """
    print(f'\n\n > call function create_Mappings()')
    

    # setup -------------------
    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # set directory if necessary
    if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
        os.makedirs(f'{data_path}/spatial_intersection_by_gm')

    # create log file for checkpoint comments
    log_file_name = f'{data_path}/spatial_intersection_by_gm/spatial_data_toparquet_by_gm_log.txt'

    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import:
        roof_kat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 1)
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 1)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name = log_file_name, n_tabs = 1)
    elif smaller_import:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name = log_file_name, n_tabs = 5)
        roof_kat_exists_TF, pv_exists_TF = os.path.exists(f'{data_path}/spatial_intersection_by_gm/roof_kat_by_gm.parquet'), os.path.exists(f'{data_path}/spatial_intersection_by_gm/pv_by_gm.parquet')
        
        roof_kat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 30000)
        checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 7)
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = int(194022/2))
        checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 7)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name = log_file_name, n_tabs = 8)

    # drop unnecessary columns ------------------
    keep_cols_roof_kat = ["SB_UUID", "DF_NUMMER", "geometry" ]
    dele_cols_roof_kat = [col for col in roof_kat_gdf.columns if col not in keep_cols_roof_kat]
    roof_kat_gdf.drop(columns = dele_cols_roof_kat, inplace = True)

    keep_cols_pv = ["xtf_id", "geometry"]
    dele_cols_pv = [col for col in pv_gdf.columns if col not in keep_cols_pv]
    pv_gdf.drop(columns = dele_cols_pv, inplace = True)

    keep_cols_gm = ["BFS_NUMMER", "geometry"]
    dele_cols_gm = [col for col in gm_shp_gdf.columns if col not in keep_cols_gm]
    gm_shp_gdf.drop(columns = dele_cols_gm, inplace = True)

        
    # convex_hull by SB_UUID ------------------
    roof_kat_union_srs = roof_kat_gdf.groupby('SB_UUID')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
    roof_kat_union = gpd.GeoDataFrame(roof_kat_union_srs, geometry='geometry')  
    roof_kat_hull = gpd.GeoDataFrame(roof_kat_union_srs, geometry='geometry') # gpd.GeoDataFrame(roof_kat_union, crs = roof_kat_gdf.crs).reset_index()
    roof_kat_hull['geometry'] = roof_kat_hull['geometry'].convex_hull

    checkpoint_to_logfile('-', log_file_name = log_file_name, n_tabs = 9)
    checkpoint_to_logfile('created roof_kat_hull + roof_kat_union', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'roof_kat_hull area is {(roof_kat_hull["geometry"].area.sum() - roof_kat_gdf["geometry"].area.sum() )/ roof_kat_gdf["geometry"].area.sum()} %  larger than roof_kat_gdf', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'roof_kat_union area is {(roof_kat_union["geometry"].area.sum() - roof_kat_gdf["geometry"].area.sum() )/ roof_kat_gdf["geometry"].area.sum()} %  larger than roof_kat_gdf', log_file_name = log_file_name, n_tabs = 1)
    
    
    # sjoin ------------------
    roof_kat_hull.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    roof_kat_union.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    pv_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    # join_roof_kat_pv = gpd.sjoin(roof_kat_hull, pv_gdf, how="left", predicate="within")
    # join_roof_kat_pv.drop(columns = ['index_right'], inplace = True)

    Map_roof_pv_hull = gpd.sjoin(roof_kat_hull, pv_gdf, how="left", predicate="intersects")
    Map_roof_pv_hull.drop(columns = ['index_right'], inplace = True)
    Map_roof_pv_hull.reset_index(inplace = True)

    Map_roof_pv_union = gpd.sjoin(roof_kat_union, pv_gdf, how="left", predicate="intersects")
    Map_roof_pv_union.drop(columns = ['index_right'], inplace = True)
    Map_roof_pv_union.reset_index(inplace = True)

    checkpoint_to_logfile('-', log_file_name = log_file_name, n_tabs = 9)
    checkpoint_to_logfile(f'shape: Map_roof_pv_hull: {Map_roof_pv_hull.shape}, Map_roof_pv_union: {Map_roof_pv_union.shape}', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'nunique SB_UUID in Map_roof_pv_hull: {Map_roof_pv_hull["SB_UUID"].nunique()}, Map_roof_pv_union: {Map_roof_pv_union["SB_UUID"].nunique()}', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'nunique xtf_id in Map_roof_pv_hull: {Map_roof_pv_hull["xtf_id"].nunique()}, Map_roof_pv_union: {Map_roof_pv_union["xtf_id"].nunique()}', log_file_name = log_file_name, n_tabs = 1)

    Map_roof_gm_hull = gpd.sjoin(roof_kat_hull, gm_shp_gdf, how="left", predicate="intersects")
    Map_roof_gm_hull.drop(columns = ['index_right'], inplace = True)
    Map_roof_gm_hull.reset_index(inplace = True)

    Map_roof_gm_union = gpd.sjoin(roof_kat_union, gm_shp_gdf, how="left", predicate="intersects")
    Map_roof_gm_union.drop(columns = ['index_right'], inplace = True)
    Map_roof_gm_union.reset_index(inplace = True)

    checkpoint_to_logfile('-', log_file_name = log_file_name, n_tabs = 9)
    checkpoint_to_logfile(f'shape: Map_roof_gm_hull: {Map_roof_gm_hull.shape}, Map_roof_gm_union: {Map_roof_gm_union.shape}', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'nunique SB_UUID in Map_roof_gm_hull: {Map_roof_gm_hull["SB_UUID"].nunique()}, Map_roof_gm_union: {Map_roof_gm_union["SB_UUID"].nunique()}', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'nunique BFS_NUMMER in Map_roof_gm_hull: {Map_roof_gm_hull["BFS_NUMMER"].nunique()}, Map_roof_gm_union: {Map_roof_gm_union["BFS_NUMMER"].nunique()}', log_file_name = log_file_name, n_tabs = 1)

    # export ------------------
    # Map_roof_pv.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv.parquet')
    # Map_roof_pv.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv.csv')

    Map_roof_pv_hull.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_hull.parquet')
    Map_roof_pv_hull.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_hull.csv')
    Map_roof_pv_union.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_union.parquet')
    Map_roof_pv_union.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_union.csv')

    Map_roof_gm_hull.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm_hull.parquet')
    Map_roof_gm_hull.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm_hull.csv')
    Map_roof_gm_union.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm_union.parquet')
    Map_roof_gm_union.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm_union.csv')

    roof_kat_hull.to_file(f'{data_path}/spatial_intersection_by_gm/roof_kat_hull_IN_PV.shp')
    roof_kat_union.to_file(f'{data_path}/spatial_intersection_by_gm/roof_kat_union_IN_PV.shp')
    
    checkpoint_to_logfile('-', log_file_name = log_file_name, n_tabs = 6)
    checkpoint_to_logfile(f'export Map_roof_pv.parquet', log_file_name = log_file_name, n_tabs = 1)



    # # create roof_pv mapping ------------------
    # def assign_xtf_to_Map_roof_pv(sb_uuid):
    #     selected_rows = Map_roof_pv_all[Map_roof_pv_all['SB_UUID'] == sb_uuid]
    #     non_nan_xtf_id = selected_rows['xtf_id'].dropna().unique()

    #     if selected_rows['xtf_id'].isna().all():
    #         return np.nan
    #     elif len(non_nan_xtf_id) == 1:
    #         return non_nan_xtf_id[0]
    #     else:
    #         return 'multiple_xft_ids'
        
    # for b in buffer_size:
    #     roof_kat_for_Map_buff = roof_kat_gdf.copy()
    #     roof_kat_for_Map_buff['geometry'] = roof_kat_for_Map_buff.buffer(b, resolution = 16)

    #     roof_pv_sjoin = gpd.sjoin(roof_kat_for_Map_buff, pv_gdf, how="left", predicate="intersects")
    #     Map_roof_pv_all = roof_pv_sjoin[['SB_UUID', 'xtf_id']]

    #     # create mapping
    #     Map_roof_pv = pd.DataFrame({'SB_UUID': roof_kat_for_Map_buff['SB_UUID'].unique()})
    #     Map_roof_pv['xtf_id'] = Map_roof_pv['SB_UUID'].apply(assign_xtf_to_Map_roof_pv)
        
    

    chapter_to_logfile('end spatial_data_toparquet.py', log_file_name = log_file_name)





# ------------------------------------------------------------------------------------------------------
# OLD ALL IN ONE FUNCTION
# ------------------------------------------------------------------------------------------------------


def heatcool_dem_spatial_toparquet(
    script_run_on_server_def = 0,
    smaller_import = False,):
    """
    Function to intersect spatial data with gm_shp and export to parquet
    """

    # setup -------------------
    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # set directory if necessary
    if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
        os.makedirs(f'{data_path}/spatial_intersection_by_gm')
    
    # create log file for checkpoint comments
    log_file_name = f'{data_path}/spatial_intersection_by_gm/spatial_data_toparquet_by_gm_log.txt'

    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import:
        heatcool_dem_gdf = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
        checkpoint_to_logfile('import heatcool_dem', log_file_name = log_file_name, n_tabs = 1)
    elif smaller_import:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name = log_file_name, n_tabs = 1)
        heatcool_dem_gdf = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 10)
        checkpoint_to_logfile('import heatcool_dem', log_file_name = log_file_name, n_tabs = 1)

    # drop unnecessary columns ------------------
        
    # sjoin to gm_shp ------------------
    heatcool_dem_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    heatcool_dem = gpd.sjoin(heatcool_dem_gdf, gm_shp_gdf, how="left", predicate="within")
    checkpoint_to_logfile('sjoin heatcool_dem', log_file_name = log_file_name, n_tabs = 1)

    # export ------------------
    heatcool_dem.to_parquet(f'{data_path}/spatial_intersection_by_gm/heatcool_dem_by_gm.parquet')
    checkpoint_to_logfile('export heatcool_dem.parquet', log_file_name = log_file_name, n_tabs = 1)

# PV ---------------------------------------------------------------------
def pv_old_spatial_toparquet(
    script_run_on_server_def = 0,
    smaller_import = False,):
    """
    Function to intersect spatial data with gm_shp and export to parquet
    """

    # setup -------------------
    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # set directory if necessary
    if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
        os.makedirs(f'{data_path}/spatial_intersection_by_gm')

    # create log file for checkpoint comments
    log_file_name = f'{data_path}/spatial_intersection_by_gm/spatial_data_toparquet_by_gm_log.txt'

    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import:
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 1)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name = log_file_name, n_tabs = 1)
    elif smaller_import:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name = log_file_name, n_tabs = 1)
        elec_prod_gdf = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 10000)
        checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 1)
        pv_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name = log_file_name, n_tabs = 1)

    # drop unnecessary columns ------------------

    # sjoin to gm_shp ------------------
    pv_gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    pv = gpd.sjoin(pv_gdf, gm_shp_gdf, how="left", predicate="within")
    checkpoint_to_logfile('sjoin pv', log_file_name = log_file_name, n_tabs = 1)

    # export ------------------
    pv.to_parquet(f'{data_path}/spatial_intersection_by_gm/pv_by_gm.parquet')
    checkpoint_to_logfile('export pv.parquet', log_file_name = log_file_name, n_tabs = 1)





# MAP ROOF GM ---------------------------------------------------------------------
def create_Map_roof_gm(
    script_run_on_server_def = 0,
    smaller_import = False,):
    """
    Function to create a mapping from roof_kat to gm, SB_UUID to BFS_NUMMER
    """

    # setup -------------------
    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # set directory if necessary
    if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
        os.makedirs(f'{data_path}/spatial_intersection_by_gm')

    # create log file for checkpoint comments
    log_file_name = f'{data_path}/spatial_intersection_by_gm/spatial_data_toparquet_by_gm_log.txt'

    # import ------------------
    gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    if not smaller_import:
        roof_kat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 1)
    elif smaller_import:
        checkpoint_to_logfile('USE SMALLER IMPORT for debugging', log_file_name = log_file_name, n_tabs = 4)
        roof_kat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 10000)
        checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 7)

    # drop unnecessary columns ------------------
    keep_cols_roof_kat = ["SB_UUID", "DF_NUMMER", "geometry" ]
    dele_cols_roof_kat = [col for col in roof_kat_gdf.columns if col not in keep_cols_roof_kat]
    roof_kat_gdf.drop(columns = dele_cols_roof_kat, inplace = True)

    keep_cols_gm = ["BFS_NUMMER", "geometry"]
    dele_cols_gm = [col for col in gm_shp_gdf.columns if col not in keep_cols_gm]
    gm_shp_gdf.drop(columns = dele_cols_gm, inplace = True)
    
        
    # convex_hull by SB_UUID ------------------
    roof_kat_union = roof_kat_gdf.groupby('SB_UUID')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
    roof_kat_hull = gpd.GeoDataFrame(roof_kat_union, geometry='geometry') # gpd.GeoDataFrame(roof_kat_union, crs = roof_kat_gdf.crs).reset_index()
    roof_kat_hull['geometry'] = roof_kat_hull['geometry'].convex_hull

    checkpoint_to_logfile('created roof_kat_hull', log_file_name = log_file_name, n_tabs = 1)
    checkpoint_to_logfile(f'roof_kat_hull is {(roof_kat_hull["geometry"].area.sum() - roof_kat_gdf["geometry"].area.sum() )/ roof_kat_gdf["geometry"].area.sum()}% the size of roof_kat_gdf', log_file_name = log_file_name, n_tabs = 1)


    # sjoin ------------------
    roof_kat_hull.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
    join_roof_kat_gm = gpd.sjoin(roof_kat_hull, gm_shp_gdf, how="left", predicate="within")
    join_roof_kat_gm.drop(columns = ['index_right'], inplace = True)
    
    Map_roof_gm = join_roof_kat_gm.copy()

    # export ------------------
    Map_roof_gm.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm.parquet')
    Map_roof_gm.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm.csv')
    roof_kat_hull.to_file(f'{data_path}/spatial_intersection_by_gm/roof_kat_hull_IN_GM.shp')
    checkpoint_to_logfile(f'export Map_roof_gm.parquet', log_file_name = log_file_name, n_tabs = 1)
            
    chapter_to_logfile('end spatial_data_toparquet.py', log_file_name = log_file_name)




# SETTIGNS --------------------------------------------------------------------
"""    
def spatial_toparquet(script_run_on_server_def = 0):

    # SETUP --------------------------------------------------------------------
    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # set directory if necessary
    if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
        os.makedirs(f'{data_path}/spatial_intersection_by_gm')

    # create log file for checkpoint comments
    log_file_name = f'{data_path}/spatial_intersection_by_gm/spatial_data_toparquet_by_gm_log.txt'
    with open(log_file_name, 'w') as log_file:
        log_file.write(f'\n')
    chapter_to_logfile('start spatial_data_toparquet_by_gm.py', log_file_name = log_file_name)


    # IMPORT SHAPES -------------------------------------------------------------

    smaller_import = False
    if not smaller_import:    
        gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
        checkpoint_to_logfile('import gm_shp', log_file_name = log_file_name, n_tabs = 1)
        roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 1)
        # faca_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-fassaden_2056.gdb/SOLKAT_FASS_20230221.gdb', layer ='SOLKAT_CH_FASS')
        # checkpoint_to_logfile('import faca_kat', log_file_name = log_file_name, n_tabs = 1)
        bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
        checkpoint_to_logfile('import bldng_reg', log_file_name = log_file_name, n_tabs = 1)
        heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
        checkpoint_to_logfile('import heatcool_dem', log_file_name = log_file_name, n_tabs = 1)
        elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 1)
        pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
        checkpoint_to_logfile('subset for pv', log_file_name = log_file_name, n_tabs = 1)
    
    elif smaller_import:
        print('USE SMALLER IMPORT for debugging')
        gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
        checkpoint_to_logfile('import gm_shp', log_file_name = log_file_name, n_tabs = 1)
        roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 10000)
        checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 1)
        # faca_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-fassaden_2056.gdb/SOLKAT_FASS_20230221.gdb', layer ='SOLKAT_CH_FASS', rows = 10000)
        # checkpoint_to_logfile('import faca_kat', log_file_name = log_file_name, n_tabs = 1)
        bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson', rows = 2)
        checkpoint_to_logfile('import bldng_reg', log_file_name = log_file_name, n_tabs = 1)
        heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 10)
        checkpoint_to_logfile('import heatcool_dem', log_file_name = log_file_name, n_tabs = 1) 
        elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 10000)
        checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 1)
        pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()


    # COPIES FOR MAPPING DF LATER ------------------------------------------------
        
    roof_kat_for_Map = roof_kat.copy()
    pv_for_Map = pv.copy()


    # DROP UNNECESSARY COLUMNS --------------------------------------------------

    checkpoint_to_logfile('\n\n', log_file_name = log_file_name, n_tabs = 10)
    # sjoin to gm_shp
    roof_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    # faca_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    bldng_reg.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    heatcool_dem.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    pv.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    roof_kat_for_Map.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    pv_for_Map.set_crs(gm_shp.crs, allow_override=True, inplace=True)

    # SJOIN ALL DF TO GM SHP ------------------------------------------------------

    roof_kat = gpd.sjoin(roof_kat, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin roof_kat', log_file_name = log_file_name, n_tabs = 1)
    # faca_kat = gpd.sjoin(faca_kat, gm_shp, how="left", predicate="within")
    # checkpoint_to_logfile('sjoin faca_kat', log_file_name = log_file_name, n_tabs = 1)
    bldng_reg = gpd.sjoin(bldng_reg, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin bldng_reg', log_file_name = log_file_name, n_tabs = 1)
    heatcool_dem = gpd.sjoin(heatcool_dem, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin heatcool_dem', log_file_name = log_file_name, n_tabs = 1)
    pv = gpd.sjoin(pv, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin pv', log_file_name = log_file_name, n_tabs = 1)


    # EXPORT TO PARQUET ---------------------------------------------------------

    roof_kat.to_parquet(f'{data_path}/spatial_intersection_by_gm/roof_kat_by_gm.parquet')
    checkpoint_to_logfile('export roof_kat.parquet', log_file_name = log_file_name, n_tabs = 1)
    # faca_kat.to_parquet(f'{data_path}/spatial_intersection_by_gm/faca_kat_by_gm.parquet')
    # checkpoint_to_logfile('export faca_kat.parquet', log_file_name = log_file_name, n_tabs = 1)
    bldng_reg.to_parquet(f'{data_path}/spatial_intersection_by_gm/bldng_reg_by_gm.parquet')
    checkpoint_to_logfile('export bldng_reg.parquet', log_file_name = log_file_name, n_tabs = 1)
    heatcool_dem.to_parquet(f'{data_path}/spatial_intersection_by_gm/heatcool_dem_by_gm.parquet')
    checkpoint_to_logfile('export heatcool_dem.parquet', log_file_name = log_file_name, n_tabs = 1)
    pv.to_parquet(f'{data_path}/spatial_intersection_by_gm/pv_by_gm.parquet')
    checkpoint_to_logfile('export pv.parquet', log_file_name = log_file_name, n_tabs = 1)
    gm_shp.to_parquet(f'{data_path}/spatial_intersection_by_gm/gm_shp.parquet')
    checkpoint_to_logfile('export gm_shp.parquet', log_file_name = log_file_name, n_tabs = 1)

    # DELETE VARIABLES for MEMORY ------------------------------------------------
    del roof_kat, bldng_reg, heatcool_dem, pv, gm_shp
    import gc
    gc.collect()


    # CREATE ROOF_PV MAPPING with different buffer sizes -------------------------
    buff_size = np.round(np.arange(0.05, 1.3, 0.05), 2)

    def assign_xtf_to_Map_roof_pv(sb_uuid):
        selected_rows = Map_roof_pv_all[Map_roof_pv_all['SB_UUID'] == sb_uuid]
        non_nan_xtf_id = selected_rows['xtf_id'].dropna().unique()

        if selected_rows['xtf_id'].isna().all():
            return np.nan
        elif len(non_nan_xtf_id) == 1:
            return non_nan_xtf_id[0]
        else:
            return 'multiple_xft_ids'

    for b in buff_size:
        roof_kat_for_Map_buff = roof_kat_for_Map.copy()
        roof_kat_for_Map_buff['geometry'] = roof_kat_for_Map_buff.buffer(b, resolution = 16)

        roof_pv_sjoin = gpd.sjoin(roof_kat_for_Map_buff, pv_for_Map, how="left", predicate="intersects")
        Map_roof_pv_all = roof_pv_sjoin[['SB_UUID', 'xtf_id']]

        # create mapping
        Map_roof_pv = pd.DataFrame({'SB_UUID': roof_kat_for_Map_buff['SB_UUID'].unique()})
        Map_roof_pv['xtf_id'] = Map_roof_pv['SB_UUID'].apply(assign_xtf_to_Map_roof_pv)
        
        # export
        Map_roof_pv.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_buff{str(b)}.parquet')
        Map_roof_pv.to_csv(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_buff{str(b)}.csv')
        Map_roof_pv_all.to_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_all_buff{str(b)}.parquet')
        checkpoint_to_logfile(f'export Map_roof_pv_buff{str(b)}.parquet', log_file_name = log_file_name, n_tabs = 1)

    chapter_to_logfile('end spatial_data_toparquet_by_gm.py', log_file_name = log_file_name)
"""

# --- OLD CODE ----------------------------------------------------------------
    # # drop unnecessary columns
# drop_gm_cols = ['index_right', 'UUID', 'DATUM_AEND', 'DATUM_ERST', 'ERSTELL_J', 'ERSTELL_M', 'REVISION_J', 'REVISION_M',
#                 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M', 'OBJEKTART', 'BEZIRKSNUM', 'SEE_FLAECH', 'REVISION_Q', 
#                 'NAME', 'KANTONSNUM', 'ICC', 'EINWOHNERZ', 'HIST_NR', 'GEM_TEIL', 'GEM_FLAECH', 'SHN']
# roof_kat.drop(columns = drop_gm_cols, axis = 1, inplace = True)
# bldng_reg.drop(columns = drop_gm_cols, axis = 1, inplace = True)
# heatcool_dem.drop(columns = drop_gm_cols, axis = 1, inplace = True)
# pv.drop(columns = drop_gm_cols, axis = 1, inplace = True)
# checkpoint_to_logfile('drop unnecessary columns', log_file_name = log_file_name, n_tabs = 1)