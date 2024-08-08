import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import winsound
import json
import plotly.express as px

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

        # drop unnecessary columns --------
        # transformations --------
        solkat_all_gdf = solkat_all_gdf.rename(columns={'GWR_EGID': 'EGID'})
        solkat_all_gdf['EGID'] = solkat_all_gdf['EGID'].replace('nan', np.nan)  # convert EGID into numbers with no decimal points
        solkat_all_gdf['EGID'] = solkat_all_gdf['EGID'].astype('Int64')
        

        # filter by bfs_nubmers_def --------
        solkat_gdf = solkat_all_gdf[solkat_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]

        # export --------
        solkat_gdf.to_parquet(f'{data_path_def}/output/preprep_data/solkat.parquet')
        solkat_gdf.to_csv(f'{data_path_def}/output/preprep_data/solkat.csv', sep=';', index=False)
        print_to_logfile(f'exported solkat data', log_file_name_def = log_file_name_def)


        # HEATING + COOLING DEMAND ---------------------------------------------------------------------
        if False: # Heat data most likely not needed because demand TS is available
            if not smaller_import_def:
                heat_all_gdf = gpd.read_file(f'{data_path_def}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
            elif smaller_import_def:
                heat_all_gdf = gpd.read_file(f'{data_path_def}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 100)
            checkpoint_to_logfile(f'import heat, {heat_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

            # attach bfs
            heat_all_gdf = attach_bfs_to_spatial_data(heat_all_gdf, gm_shp_gdf)

            # drop unnecessary columns --------
            # transformations --------

            # filter by bfs_nubmers_def --------
            heat_gdf = heat_all_gdf[heat_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]
            
            # export --------
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

        # drop unnecessary columns --------
        # transformations --------

        # filter by bfs_nubmers_def --------
        pv_gdf = pv_all_gdf[pv_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]

        # export --------
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
    print_to_logfile(f'run function: import_demand_TS_AND_match_households.py', log_file_name_def = log_file_name_def)



    # IMPORT CONSUMER DATA ============================================================================
    print_to_logfile(f'\nIMPORT CONSUMER DATA {10*"*"}', log_file_name_def = log_file_name_def) 
       
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
    checkpoint_to_logfile(f'consumer demand TS contains assets: {all_assets_unique}', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

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
            checkpoint_to_logfile(f'exported demand TS for consumer {c}, {c_number+1} of {len(netflex_consumers_list)}', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
    
    # remove pv assets because they also have negative values
    agg_demand_df = agg_demand_df[agg_demand_df['tech'] != 'pv']

    # plot TS for certain consumers by assets
    plot_ids =['ID100', 'ID101', 'ID102', ]
    plot_df = agg_demand_df[agg_demand_df['id'].isin(plot_ids)]
    fig = px.line(plot_df, x='t', y='value', color='asset', title='Demand TS for selected consumers')
    # fig.show()

    # export aggregated demand for all NETFLEX consumer assets
    agg_demand_df.to_parquet(f'{data_path_def}/output/preprep_data/demand_ts.parquet')
    checkpoint_to_logfile(f'exported demand TS for all consumers', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
    


    # AGGREGATE DEMAND TYPES ============================================================================
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
    demand_types = pd.DataFrame()
    t_sequence = agg_demand_df['t'].unique()
    demand_types['t'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().keys()
    demand_types['high_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().values
    demand_types['low_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_wiHP)].groupby('t')['value'].mean().values
    demand_types['high_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_noHP)].groupby('t')['value'].mean().values
    demand_types['low_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_noHP)].groupby('t')['value'].mean().values

    demand_types['t'] = pd.Categorical(demand_types['t'], categories=t_sequence, ordered=True)
    demand_types = demand_types.sort_values(by = 't')
    demand_types = demand_types.reset_index(drop=True)

    demand_types.to_parquet(f'{data_path_def}/output/preprep_data/demand_types.parquet')
    checkpoint_to_logfile(f'exported demand types', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

    # plot demand types with plotly
    fig = px.line(demand_types, x='t', y=['high_wiHP', 'low_wiHP', 'high_noHP', 'low_noHP'], title='Demand types')
    # fig.show()
    fig.write_html(f'{data_path_def}/output/preprep_data/demand_types.html')



    # MATCH DEMAND TYPES TO HOUSEHOLDS ============================================================================
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
    Map_demand_type_gwrEGID ={
        'high_DEMANDprox_wiHP': high_DEMANDprox_wiHP_list,
        'low_DEMANDprox_wiHP': low_DEMANDprox_wiHP_list,
        'high_DEMANDprox_noHP': high_DEMANDprox_noHP_list,
        'low_DEMANDprox_noHP': low_DEMANDprox_noHP_list,
    }
    with open(f'{data_path_def}/output/preprep_data/Map_demand_type_gwrEGID.json', 'w') as f:
        json.dump(Map_demand_type_gwrEGID, f)
    checkpoint_to_logfile(f'exported demand types for GWR', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)




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



