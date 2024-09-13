import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import winsound
import json
import plotly.express as px
import glob

from datetime import datetime
from shapely.geometry import Point

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


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
    solkat_selection_specs_def = dataagg_settings_def['solkat_selection_specs']
    print_to_logfile(f'run function: local_data_AND_spatial_mappings.py', log_file_name_def = log_file_name_def)

    # IMPORT DATA -------------------
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
        
    # PV 
    pv_all_pq = pd.read_parquet(f'{data_path_def}/split_data_geometry/pv_pq.parquet')
    checkpoint_to_logfile(f'import pv_pq, {pv_all_pq.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  2, show_debug_prints_def)
    pv_all_geo = gpd.read_file(f'{data_path_def}/split_data_geometry/pv_geo.geojson')
    checkpoint_to_logfile(f'import pv_geo, {pv_all_geo.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  2, show_debug_prints_def)

    # transformations
    pv_all_pq, pv_all_geo = cols_to_str(['xtf_id',], pv_all_pq, pv_all_geo)

    pv = pv_all_pq[pv_all_pq['BFS_NUMMER'].isin(bfs_number_def)]  # select and export pq for BFS numbers
    pv.to_parquet(f'{data_path_def}/output/preprep_data/pv.parquet')
    pv_wgeo = pv.merge(pv_all_geo[['xtf_id', 'geometry']], how = 'left', on = 'xtf_id') # merge geometry for later use
    pv_gdf = gpd.GeoDataFrame(pv_wgeo, geometry='geometry')

    # SOLKAT 
    solkat_all_pq = pd.read_parquet(f'{data_path_def}/split_data_geometry/solkat_pq.parquet')
    checkpoint_to_logfile(f'import solkat_pq, {solkat_all_pq.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  1, show_debug_prints_def)
    
    bsblso_kt_numbers_TF = all([kt in [11,12,13] for kt in dataagg_settings_def['kt_numbers']])
    if (bsblso_kt_numbers_TF) & (os.path.exists(f'{data_path_def}/split_data_geometry/solkat_bsblso_geo.geojson')):
        solkat_all_geo = gpd.read_file(f'{data_path_def}/split_data_geometry/solkat_bsblso_geo.geojson')
    else:
        solkat_all_geo = gpd.read_file(f'{data_path_def}/split_data_geometry/solkat_geo.geojson')
    checkpoint_to_logfile(f'import solkat_geo, {solkat_all_geo.shape[0]} rows, (smaller_import: {smaller_import_def})', log_file_name_def ,  1, show_debug_prints_def)

    # transformations
    solkat_all_pq = cols_to_str(['SB_UUID', 'DF_UID', 'GWR_EGID'], solkat_all_pq)
    solkat_all_geo = cols_to_str(['DF_UID',], solkat_all_geo)
    solkat_all_pq.rename(columns = {'GWR_EGID': 'EGID'}, inplace = True)

    solkat = solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(bfs_number_def)]  # select and export pq for BFS numbers
    solkat.to_parquet(f'{data_path_def}/output/preprep_data/solkat.parquet')
    solkat_wgeo = solkat.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID') # merge geometry for later use
    solkat_gdf = gpd.GeoDataFrame(solkat_wgeo, geometry='geometry')

    # GWR
    gwr = pd.read_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
    checkpoint_to_logfile(f'import gwr, {gwr.shape[0]} rows', log_file_name_def, 5, show_debug_prints_def = show_debug_prints_def)

    # transform to gdf
    gwr = gwr.loc[(gwr['GKODE'] != '') & (gwr['GKODN'] != '')]
    gwr[['GKODE', 'GKODN']] = gwr[['GKODE', 'GKODN']].astype(float)
    gwr['geometry'] = gwr.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
    gwr_gdf = gpd.GeoDataFrame(gwr, geometry='geometry')



    # MAP: solkatdfuid > egid -------------------
    Map_solkatdfuid_egid = solkat_gdf.loc[:,['DF_UID', 'DF_NUMMER', 'SB_UUID', 'EGID']].copy()
    Map_solkatdfuid_egid.rename(columns = {'GWR_EGID': 'EGID'}, inplace = True)
    Map_solkatdfuid_egid = Map_solkatdfuid_egid.loc[Map_solkatdfuid_egid['EGID'] != '']

    Map_solkatdfuid_egid.to_parquet(f'{data_path_def}/output/preprep_data/Map_solkatdfuid_egid.parquet')
    Map_solkatdfuid_egid.to_csv(f'{data_path_def}/output/preprep_data/Map_solkatdfuid_egid.csv', sep=';', index=False)



    # MAP: egid > pv -------------------
    def set_crs_to_gm_shp(gdf_CRS, gdf_a, gdf_b = None):
        gdf_a.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
        if gdf_b is not None:
            gdf_b.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
        
        if gdf_b is None: 
            return gdf_a
        if gdf_b is not None:
            return gdf_a, gdf_b
        
    gwr_buff_gdf = gwr_gdf.copy()
    gwr_buff_gdf.set_crs("EPSG:32632", allow_override=True, inplace=True)
    gwr_buff_gdf['geometry'] = gwr_buff_gdf['geometry'].buffer(solkat_selection_specs_def['GWR_EGID_buffer_size'])
    gwr_buff_gdf, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, gwr_buff_gdf, pv_gdf)
    checkpoint_to_logfile(f'gwr_gdf.crs == pv_gdf.crs: {gwr_buff_gdf.crs == pv_gdf.crs}', log_file_name_def, 6, show_debug_prints_def)

    gwregid_pvid = gpd.sjoin(pv_gdf,gwr_buff_gdf, how="left", predicate="within")
    gwregid_pvid.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)
    checkpoint_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid["EGID"].isna().sum() / gwregid_pvid.shape[0] *100,2)} % of pv rows ({gwregid_pvid.shape[0]}) are missing EGID', log_file_name_def, 2, show_debug_prints_def)

    Map_egid_pv = gwregid_pvid.loc[gwregid_pvid['EGID'].notna(), ['EGID', 'xtf_id']].copy()
    Map_egid_pv.to_parquet(f'{data_path_def}/output/preprep_data/Map_egid_pv.parquet')
    Map_egid_pv.to_csv(f'{data_path_def}/output/preprep_data/Map_egid_pv.csv', sep=';', index=False)


    # OMITTED SPATIAL POINTS / POLYS -------------------
    print_to_logfile(f'number of omitted buildings because egid is (not) present in GWR or Solkat data frame or vice-versa', dataagg_settings_def['summary_file_name'])
    omitt_gwregid_gdf = gwr_gdf.loc[~gwr_gdf['EGID'].isin(solkat_gdf['EGID'])].copy()
    checkpoint_to_logfile(f'omitt_gwregid_gdf: {omitt_gwregid_gdf.shape[0]} rows ({round((omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%, gwr[EGID].unique {gwr_gdf['EGID']})', dataagg_settings_def['summary_file_name'], 2, True) 

    omitt_solkat_gdf = solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_gdf['EGID'])].copy()
    checkpoint_to_logfile(f'omitt_solkat_gdf: {omitt_solkat_gdf.shape[0]} rows ({round((omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0])*100, 2)}%, solkat[EGID].unique {solkat_gdf["EGID"].unique()})', dataagg_settings_def['summary_file_name'], 2, True)

    omitt_pv_gdf = pv_gdf.loc[~pv_gdf['xtf_id'].isin(gwregid_pvid['xtf_id'])].copy()
    checkpoint_to_logfile(f'omitt_pv_gdf: {omitt_pv_gdf.shape[0]} rows ({round((omitt_pv_gdf.shape[0]/pv_gdf.shape[0])*100, 2)}%, pv[xtf_id].unique {pv_gdf["xtf_id"].unique()})', dataagg_settings_def['summary_file_name'], 2, True)


    # EXPORT SPATIAL DATA -------------------
    gdf_to_export_names = ['pv_gdf', 'solkat_gdf', 'gwr_gdf', 'omiitt_gwregid_gdf', 'omitt_solkat_gdf', 'gwr_buff_gdf']
    gdf_to_export_list = [pv_gdf, solkat_gdf, gwr_gdf, omitt_gwregid_gdf, omitt_solkat_gdf, gwr_buff_gdf]
    
    for i,g in enumerate(gdf_to_export_list):
        cols_DATUM = [col for col in g.columns if 'DATUM' in col]
        g.drop(columns = cols_DATUM, inplace = True)

        checkpoint_to_logfile(f'exporting {gdf_to_export_names[i]}', log_file_name_def , 4, show_debug_prints_def)
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



# ------------------------------------------------------------------------------------------------------
# NO LONGER USED

# BY SB_UUID - FALSE - IMPORT LOCAL DATA + create SPATIAL MAPPINGS
# ------------------------------------------------------------------------------------------------------
if False:
    def local_data_to_parquet_AND_create_spatial_mappings_bySBUUID(
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

        gwr_selection_specs_def = dataagg_settings_def['gwr_selection_specs']
        solkat_selection_specs_def = dataagg_settings_def['solkat_selection_specs']
        print_to_logfile(f'run function: local_data_to_parquet_AND_create_spatial_mappings.py', log_file_name_def = log_file_name_def)



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
                if 'BFS_NUMMER' in gdf.columns:
                    # transform BFS_NUMMER to str, np.nan to ''
                    gdf['BFS_NUMMER'] = gdf['BFS_NUMMER'].apply(lambda x: '' if pd.isna(x) else str(int(x)))

                return gdf 
            

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
                elec_prod_gdf = gpd.read_file(f'{data_path_def}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 100000)
                pv_all_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
            checkpoint_to_logfile(f'import pv, {pv_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

            # filter by bfs_nubmers_def --------
            pv_all_gdf = attach_bfs_to_spatial_data(pv_all_gdf, gm_shp_gdf)
            pv_gdf = pv_all_gdf[pv_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]

            # export --------
            pv_gdf.to_parquet(f'{data_path_def}/output/preprep_data/pv.parquet')
            pv_gdf.to_csv(f'{data_path_def}/output/preprep_data/pv.csv', sep=';', index=False)
            print_to_logfile(f'exported pv data', log_file_name_def = log_file_name_def)


            # GWR KATASTER --------------------------------------------------------------------
            gwr_all = pd.read_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
            checkpoint_to_logfile(f'import gwr, {gwr_all.shape[0]} rows', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

            # transform to gdf
            gwr = gwr_all.loc[gwr_all['GGDENR'].isin(bfs_number_def) ].copy()
            gwr = gwr.loc[(gwr['GKODE'] != '') & (gwr['GKODN'] != '')]
            gwr[['GKODE', 'GKODN']] = gwr[['GKODE', 'GKODN']].astype(float)
            gwr['geometry'] = gwr.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
            gwr_gdf = gpd.GeoDataFrame(gwr, geometry='geometry')


            # SOLAR KATASTER --------------------------------------------------------------------
            if not smaller_import_def:  
                solkat_all_gdf = gpd.read_file(f'{data_path_def}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
            elif smaller_import_def:
                solkat_all_gdf = gpd.read_file(f'{data_path_def}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 6000)
            checkpoint_to_logfile(f'import solkat, {solkat_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
            checkpoint_to_logfile(f'No. of unique DF_UID in solkat: {solkat_all_gdf["DF_UID"].nunique()} of {solkat_all_gdf.shape[0]} nrows in solkat ', log_file_name_def, show_debug_prints_def)


            # filter by bfs_nubmers_def --------
            solkat_all_gdf = attach_bfs_to_spatial_data(solkat_all_gdf, gm_shp_gdf, )
            solkat_all_gdf['BFS_NUMMER'] = solkat_all_gdf['BFS_NUMMER'].astype(str)
            cols_to_str = ['BFS_NUMMER', 'DF_UID', 'SB_UUID', 'GWR_EGID']
            for col in cols_to_str:
                if isinstance(solkat_all_gdf[col].iloc[0], float):
                    solkat_all_gdf[col] = solkat_all_gdf[col].replace(np.nan, 0).astype(int).astype(str)
                    solkat_all_gdf[col] = solkat_all_gdf[col].replace('0', '')
                elif isinstance(solkat_all_gdf[col].iloc[0], int):
                    solkat_all_gdf[col] = solkat_all_gdf[col].replace(np.nan, 0).astype(str)
                    solkat_all_gdf[col] = solkat_all_gdf[col].replace('0', '')

            solkat_gdf = solkat_all_gdf[solkat_all_gdf['BFS_NUMMER'].isin(bfs_number_def)]

            # transformations --------
            solkat_gdf = solkat_gdf.rename(columns={'GWR_EGID': 'solkat_EGID'})
            solkat_gdf.dtypes
            solkat_gdf['solkat_EGID'] = solkat_gdf['solkat_EGID'].apply(lambda x: '' if pd.isna(x) else str(int(x))) # convert EGID into str numbers and replace nan with '' empty string

            # export --------
            solkat_gdf.to_parquet(f'{data_path_def}/output/preprep_data/solkat.parquet')
            solkat_gdf.to_csv(f'{data_path_def}/output/preprep_data/solkat.csv', sep=';', index=False)
            print_to_logfile(f'exported solkat data', log_file_name_def = log_file_name_def)



        # MAPPINGS & SPATIAL MAPPIGNS ============================================================================
        print_to_logfile(f'MAPPINGS & SPATIAL MAPPIGNS {10*"*"}', log_file_name_def = log_file_name_def)
        # if True: 

        # only keep certain cols and remove those that are not relevant for spatial/geographic mapping
        def keep_columns (col_names, gdf):
            keep_cols = col_names
            dele_cols = [col for col in gdf.columns if col not in keep_cols]
            gdf.drop(columns = dele_cols, inplace = True)
            return gdf


        # MAP: solkat_dfuid > gwr_egid ---------------------------------------------------------------------
        # attach EGID from GWR data to SOLKAT because solkat EGIDs are largely missing (much aboe 10%)
        buffer = solkat_selection_specs_def['GWR_EGID_buffer_size']
        col_partition_union = solkat_selection_specs_def['col_partition_union']

        print_to_logfile(f'\nGWR_EGID in SOLKAT is largely missing (much above 10%), or faulty (1 EGIDs for multiple houses, contrary to GWR).' , log_file_name_def = log_file_name_def)
        print_to_logfile(f'That is why the roof unions are only considered for overlapping SB_UUIDs and buffered GWR Points', log_file_name_def = log_file_name_def)
        print_to_logfile(f' Applying a buffer of {buffer}m to GWR points to increase the overlap direct neighboring partitions (e.g. Giebeldaecher)', log_file_name_def = log_file_name_def)
        
        checkpoint_to_logfile(f'n of solkat[GWR_EGID] == NaN: {solkat_gdf.loc[solkat_gdf["solkat_EGID"].isna()].shape[0]}', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        checkpoint_to_logfile(f'> {round(solkat_gdf.loc[solkat_gdf["solkat_EGID"] == ""].shape[0] / solkat_gdf.shape[0] *100, 2)}% of solkat rows are missing EGID (% of roof partitions)', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

        def set_crs_to_gm_shp(gdf_CRS, gdf_a, gdf_b = None):
            gdf_a.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
            if gdf_b is not None:
                gdf_b.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
            
            if gdf_b is None: 
                return gdf_a
            if gdf_b is not None:
                return gdf_a, gdf_b


        # Build unions with SB_UUIDs --------
        solkat_sbuuid = solkat_gdf.loc[:,['SB_UUID', 'DF_NUMMER', 'DF_UID', 'BFS_NUMMER', 'geometry']].copy()
        solkat_sbuuid_union = solkat_sbuuid.groupby(col_partition_union)['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
        solkat_sbuuid_gdf = gpd.GeoDataFrame(solkat_sbuuid_union, geometry='geometry').reset_index() 


        # attach GWR EGID to SOLKAT by SB_UUID --------
        gwr_buff_gdf = gwr_gdf.loc[gwr_gdf['GGDENR'].isin(solkat_gdf['BFS_NUMMER'].unique())].copy()
        gwr_buff_gdf.set_crs("EPSG:32632", allow_override=True, inplace=True)
        gwr_buff_gdf['geometry'] = gwr_buff_gdf['geometry'].buffer(buffer)
        gwr_buff_gdf, solkat_sbuuid_gdf = set_crs_to_gm_shp(gm_shp_gdf, gwr_buff_gdf, solkat_sbuuid_gdf)
        checkpoint_to_logfile(f'gwr_gdf.crs == solkat_sbuuid_gdf.crs: {gwr_buff_gdf.crs == solkat_sbuuid_gdf.crs}', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

        solkat_sbuuid_gdf = gpd.sjoin(solkat_sbuuid_gdf, gwr_buff_gdf, how="left", predicate="intersects")
        solkat_sbuuid_gdf.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)
        checkpoint_to_logfile(f'sjoin completed', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        

        # one unique EGID per one or multiple SB_UUID --------
        # create index for different cases (occurences of SB_UUIDs and EGIDs)
        sbuuid_counts_wEGID = solkat_sbuuid_gdf.loc[solkat_sbuuid_gdf['EGID'].notna(), 'SB_UUID'].value_counts()
        a_single_sbuuid = list(sbuuid_counts_wEGID[sbuuid_counts_wEGID == 1].index)
        b_multip_sbuuid = list(sbuuid_counts_wEGID[sbuuid_counts_wEGID > 1].index)
        sbuuid_counts_noEGID = solkat_sbuuid_gdf.loc[solkat_sbuuid_gdf['EGID'].isna(), 'SB_UUID'].value_counts()
        c_egid_nan = list(sbuuid_counts_noEGID[sbuuid_counts_noEGID == 1].index)
        checkpoint_to_logfile(f'sanity check solkat_sbuuid_gdf: {len(a_single_sbuuid) + len(b_multip_sbuuid) + len(c_egid_nan)} = {solkat_gdf["SB_UUID"].nunique()} => a (single sbuuid), b (multiple sbuuid), c (no EGID) = solkat[SBUUID].nunique', log_file_name_def, show_debug_prints_def) 

        # single sbuuid
        submap_oneSBUUID = solkat_gdf.copy()
        submap_oneSBUUID = submap_oneSBUUID.loc[submap_oneSBUUID['SB_UUID'].isin(a_single_sbuuid)]
        submap_oneSBUUID = submap_oneSBUUID.merge(solkat_sbuuid_gdf.loc[:,['SB_UUID', 'EGID']], how = 'left', on = 'SB_UUID')
        submap_oneSBUUID = submap_oneSBUUID.loc[:,['SB_UUID', 'DF_NUMMER', 'DF_UID', 'BFS_NUMMER', 'EGID']]
        submap_oneSBUUID.rename(columns = {'EGID': 'EGID_oneSBUUID'}, inplace = True)

        # multiple sbuuid
        solkat_multipSBUUID = solkat_gdf.copy()
        solkat_multipSBUUID = solkat_multipSBUUID.loc[solkat_multipSBUUID['SB_UUID'].isin(b_multip_sbuuid)]
        gwr_gdf, solkat_multipSBUUID = set_crs_to_gm_shp(gm_shp_gdf, gwr_gdf, solkat_multipSBUUID)
        solkat_df_uid_gdf =  gpd.sjoin(solkat_multipSBUUID, gwr_buff_gdf, how="left", predicate="within")
        solkat_df_uid_gdf.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)
        submap_multipSBUUID = solkat_df_uid_gdf
        submap_multipSBUUID = submap_multipSBUUID.loc[:,['SB_UUID', 'DF_NUMMER', 'DF_UID', 'BFS_NUMMER', 'EGID']]
        submap_multipSBUUID.rename(columns = {'EGID': 'EGID_multipSBUUID'}, inplace = True)


        # merge sub_gdfs to one single mapping gdf --------
        Map_solkatdfuid_egid = solkat_gdf.copy()
        Map_solkatdfuid_egid = Map_solkatdfuid_egid.merge(submap_oneSBUUID.loc[:,['DF_UID', 'EGID_oneSBUUID']], how = 'left', on = 'DF_UID')
        Map_solkatdfuid_egid = Map_solkatdfuid_egid.merge(submap_multipSBUUID.loc[:,['DF_UID', 'EGID_multipSBUUID']], how = 'left', on = 'DF_UID')

        Map_solkatdfuid_egid['EGID'] = np.where(Map_solkatdfuid_egid['EGID_oneSBUUID'].notna(), Map_solkatdfuid_egid['EGID_oneSBUUID'], Map_solkatdfuid_egid['EGID_multipSBUUID'])
        Map_solkatdfuid_egid.drop(columns = ['EGID_oneSBUUID', 'EGID_multipSBUUID'], inplace = True)
        Map_solkatdfuid_egid = Map_solkatdfuid_egid.loc[:,['DF_UID', 'EGID']]
        print_to_logfile(f'attached EGID based one SB_UUID mapping to DF_UID where ever possible', log_file_name_def )
        print_to_logfile(f'In case of double EGIDs per SB_UUID, the buffered EGID from GWR directly with DF_UID', log_file_name_def )
        print_to_logfile(f'\t => EGID mapping for {Map_solkatdfuid_egid["EGID"].notna().sum()} of {Map_solkatdfuid_egid.shape[0]} roof partitions ({(Map_solkatdfuid_egid["EGID"].notna().sum()/Map_solkatdfuid_egid.shape[0])*100} %)', log_file_name_def )
        print_to_logfile(f'\n*ATTENTION*, Map_solkatdfuid_egid.shape[0]: {Map_solkatdfuid_egid.shape[0]}, DF_UID.nunique: {Map_solkatdfuid_egid["DF_UID"].nunique()}', log_file_name_def)
        print_to_logfile(f'It is possible that there is an overcounting of houses, but it is minimal at the testing level', log_file_name_def)


        # export     --------
        Map_solkatdfuid_egid.to_parquet(f'{data_path_def}/output/preprep_data/Map_solkatdfuid_egid.parquet')
        Map_solkatdfuid_egid.to_csv(f'{data_path_def}/output/preprep_data/Map_solkatdfuid_egid.csv', sep=';', index=False)
        checkpoint_to_logfile(f'exported Map_solkatdfuid_egid', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        

        # UPDATE: solkat by gwr_egid ---------------------------------------------------------------------
        solkat_pq = solkat_gdf.copy() #solkat_all_gdf[solkat_all_gdf['BFS_NUMMER'].isin(bfs_number_def)].copy()
        checkpoint_to_logfile(f'{solkat_pq.shape[0]}, {solkat_pq["DF_UID"].nunique()}, {Map_solkatdfuid_egid["DF_UID"].nunique()} > nrows_solkat_pq, solkat_DFUID.nunique, Map_solkatdfuid_egid_DFUID.nunique', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        solkat_pq = solkat_pq.merge(Map_solkatdfuid_egid.loc[:,['DF_UID', 'EGID']], how = 'left', on = 'DF_UID')
        solkat_pq.to_parquet(f'{data_path_def}/output/preprep_data/solkat.parquet')
        solkat_pq.to_csv(f'{data_path_def}/output/preprep_data/solkat.csv', sep=';', index=False)
        checkpoint_to_logfile(f'exported updated solkat_gdf (w EGID)', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
        

        # MAP: solkat_egid > pv id ---------------------------------------------------------------------
        # create shape unions now based on EGID, to get highest possible match and accuracy in overlap with pv installations
        solkat_egid_union = solkat_gdf.copy()
        solkat_egid_union = solkat_egid_union.merge(Map_solkatdfuid_egid, how = 'left', on = 'DF_UID')
        solkat_egid_union = solkat_egid_union.groupby('EGID')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union)
        solkat_egid_gdf = gpd.GeoDataFrame(solkat_egid_union, geometry='geometry').reset_index()

        solkat_egid_gdf, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, solkat_egid_gdf, pv_gdf)
        Map_solkategid_pv = gpd.sjoin(solkat_egid_gdf, pv_gdf, how="left", predicate="within")
        Map_solkategid_pv = Map_solkategid_pv.loc[:,['EGID','xtf_id', ]]
        Map_solkategid_pv['xtf_id'].value_counts()

        Map_solkategid_pv.to_parquet(f'{data_path_def}/output/preprep_data/Map_solkategid_pv.parquet')
        Map_solkategid_pv.to_csv(f'{data_path_def}/output/preprep_data/Map_solkategid_pv.csv', sep=';', index=False)
        checkpoint_to_logfile(f'exported Map_egid_pv', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)


        # MAP: solkat > geometry ---------------------------------------------------------------------
        GEOM_solkat = solkat_gdf.copy()
        GEOM_solkat.to_file(f'{data_path_def}/output/preprep_data/GEOM_solkat.geojson', driver='GeoJSON')
        checkpoint_to_logfile(f'exported GEOM_solkat', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

        GEOM_solkatdfuid = solkat_gdf.copy()
        nonNA_dfuid = Map_solkatdfuid_egid.dropna(subset = ['EGID'])['DF_UID'].unique()
        GEOM_solkatdfuid = GEOM_solkatdfuid.loc[GEOM_solkatdfuid['DF_UID'].isin(nonNA_dfuid)]
        GEOM_solkatdfuid.to_file(f'{data_path_def}/output/preprep_data/GEOM_solkatdfuid.geojson', driver='GeoJSON')
        checkpoint_to_logfile(f'exported GEOM_solkatdfuid', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)


        # MAP: GWR > geometry ---------------------------------------------------------------------
        GEOM_gwr = gwr_gdf.copy()
        GEOM_gwr.to_file(f'{data_path_def}/output/preprep_data/GEOM_gwr.geojson', driver='GeoJSON')


        # MAP: PV > geometry ---------------------------------------------------------------------
        GEOM_pv = pv_gdf.copy()
        GEOM_pv.to_file(f'{data_path_def}/output/preprep_data/GEOM_pv.geojson', driver='GeoJSON')
        checkpoint_to_logfile(f'exported GEOM_pv', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)


        # SHP testing exports ---------------------------------------------------------------------
        if True: 
            if os.path.exists(f'{data_path_def}/output/testing_shp') == False:
                os.makedirs(f'{data_path_def}/output/testing_shp')
            # elif os.path.exists(f'{data_path_def}/output/testing_shp') == True:
            #     files_to_remove = glob.glob(f'{data_path_def}/output/testing_shp/*')
            #     for f in files_to_remove:
            #         os.remove(f)

            gwr_small_gdf = gwr_gdf.loc[gwr_gdf['GGDENR'].isin(solkat_gdf['BFS_NUMMER'].unique())].copy()
            gwr_small_gdf.to_file(f'{data_path_def}/output/testing_shp/gwr_gdf.shp')
            checkpoint_to_logfile(f'exported gwr_gdf', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
            gwr_buff_gdf.to_file(f'{data_path_def}/output/testing_shp/gwr_buff_gdf.shp')
            checkpoint_to_logfile(f'exported gwr_buff_gdf', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)
            if smaller_import_def: 
                solkat_shp = solkat_gdf.copy()
                cols_DATUM = [col for col in solkat_shp.columns if 'DATUM' in col]
                solkat_shp.drop(columns = cols_DATUM, inplace = True)
                solkat_shp.to_file(f'{data_path_def}/output/testing_shp/solkat_gdf.shp')
                checkpoint_to_logfile(f'exported solkat_gdf', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

            solkat_sbuuid_gdf.to_file(f'{data_path_def}/output/testing_shp/solkat_sbuuid_gdf.shp')
            checkpoint_to_logfile(f'exported solkat_sbuuid_gdf', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

            solkat_egid_gdf.to_file(f'{data_path_def}/output/testing_shp/solkat_egid_gdf.shp')
            checkpoint_to_logfile(f'exported solkat_egid_gdf', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def)

            pv_gdf.to_file(f'{data_path_def}/output/testing_shp/pv_gdf.shp')
            checkpoint_to_logfile(f'exported pv_gdf', log_file_name_def = log_file_name_def, show_debug_prints_def = show_debug_prints_def) 

            print('exported shp files')





