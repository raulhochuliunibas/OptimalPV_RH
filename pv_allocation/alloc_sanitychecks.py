import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import copy
import glob
import plotly.graph_objs as go
import plotly.offline as pyo


from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# visualization of PV topology
# ------------------------------------------------------------------------------------------------------
def create_gdf_export_of_topology(
        pvalloc_settings, ):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    bfs_numbers_def = pvalloc_settings['bfs_numbers']

    # create topo_df -----------------------------------------------------
    topo = json.load(open(f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'r'))
    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    topo_df_uid_list = []    
    for k,v in topo.items():
        egid_list.append(k)
        gklas_list.append(v.get('gwr_info').get('gklas'))
        inst_tf_list.append(v.get('pv_inst').get('inst_TF'))
        inst_info_list.append(v.get('pv_inst').get('inst_info'))
        inst_id_list.append(v.get('pv_inst').get('xtf_id'))
        beginop_list.append(v.get('pv_inst').get('BeginOp'))
        power_list.append(v.get('pv_inst').get('TotalPower'))

        for k_sub, v_sub in v.get('solkat_partitions').items():
            topo_df_uid_list.append(k_sub)


    topo_df = pd.DataFrame({'EGID': egid_list,'gklas': gklas_list,
                            'inst_tf': inst_tf_list,'inst_info': inst_info_list,'inst_id': inst_id_list,'beginop': beginop_list,'power': power_list,
    })
    # topo_df['power'] = topo_df['power'].replace('', 0).infer_objects(copy=False).astype(float)
    # topo_df['power'] = topo_df['power'].replace('', 0).astype(object)
    # topo_df['power'] = pd.to_numeric(topo_df['power'], errors='coerce').fillna(0)
    topo_df['power'] = pd.to_numeric(topo_df['power'].replace('', '0'), errors='coerce').fillna(0)


    # import geo data -----------------------------------------------------
    # topo_df = pd.read_parquet(f'{data_path_def}/output/pvalloc_run/topo_egid_df.parquet')

    if pvalloc_settings['fast_debug_run']:
        solkat_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/solkat_gdf.geojson', rows=50)
        gwr_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gwr_gdf.geojson', rows = 50)
        pv_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/pv_gdf.geojson', rows = 50)
    else:
        solkat_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/solkat_gdf.geojson')
        gwr_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gwr_gdf.geojson')
        pv_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/pv_gdf.geojson')

    Map_egid_dsonode = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_dsonode.parquet')
    gwr_bsblso_gdf = gpd.read_file(f'{data_path_def}/split_data_geometry/gwr_bsblso_gdf.geojson')
    gm_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gm_shp_gdf.geojson')


    # transformations
    pv_gdf['xtf_id'] = pv_gdf['xtf_id'].astype(int).replace(np.nan, "").astype(str)
    solkat_gdf['DF_UID'] = solkat_gdf['DF_UID'].astype(int).replace(np.nan, "").astype(str)
    solkat_gdf.rename(columns={'DF_UID': 'df_uid'}, inplace=True)

    # DSO whole gridnet
    dsonodes_withegids_gdf = Map_egid_dsonode.merge(gwr_bsblso_gdf, on='EGID', how='left')
    dsonodes_withegids_gdf = gpd.GeoDataFrame(dsonodes_withegids_gdf, crs='EPSG:2056', geometry='geometry')


    # subset gwr + pv -----------------------------------------------------
    solkat_gdf_in_topo = copy.deepcopy(solkat_gdf.loc[solkat_gdf['df_uid'].isin(topo_df_uid_list)])
    gwr_gdf_in_topo = copy.deepcopy(gwr_gdf.loc[gwr_gdf['EGID'].isin(topo_df['EGID'].unique())])
    pv_gdf_in_topo = copy.deepcopy(pv_gdf.loc[pv_gdf['xtf_id'].isin(topo_df['inst_id'].unique())])

    solkat_gdf_notin_topo = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['df_uid'].isin(topo_df_uid_list)])
    gwr_gdf_notin_topo = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(topo_df['EGID'].unique())])
    pv_gdf_notin_topo = copy.deepcopy(pv_gdf.loc[~pv_gdf['xtf_id'].isin(topo_df['inst_id'].unique())])
    

    topo_gdf = topo_df.merge(gwr_gdf[['EGID', 'geometry']], on='EGID', how='left')
    topo_gdf = gpd.GeoDataFrame(topo_gdf, crs='EPSG:2056', geometry='geometry')


    solkat_in_grid = solkat_gdf.loc[solkat_gdf['EGID'].isin(Map_egid_dsonode['EGID'].unique())]
    solkat_in_grid = solkat_in_grid.loc[solkat_in_grid['BFS_NUMMER'].isin(bfs_numbers_def)]
    single_partition_houses = copy.deepcopy(solkat_in_grid[solkat_in_grid['EGID'].map(solkat_in_grid['EGID'].value_counts()) == 1])
    single_part_houses_w_tilt = copy.deepcopy(single_partition_houses.loc[single_partition_houses['NEIGUNG'] > 0])
    print_to_logfile(f'\n\nSINGLE PARTITION HOUSES WITH TILT for debugging:', log_file_name_def)
    checkpoint_to_logfile(f'First 10 EGID rows: {single_part_houses_w_tilt["EGID"][0:10]}', log_file_name_def) 


    # EXPORT to shp -----------------------------------------------------
    if not os.path.exists(f'{data_path_def}/output/pvalloc_run/topo_spatial_data'):
        os.makedirs(f'{data_path_def}/output/pvalloc_run/topo_spatial_data')

    shp_to_export=[(solkat_gdf_in_topo, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/solkat_gdf_in_topo.shp'),
                   (gwr_gdf_in_topo, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/gwr_gdf_in_topo.shp'),
                   (pv_gdf_in_topo, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/pv_gdf_in_topo.shp'),
                   
                   (solkat_gdf_notin_topo, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/solkat_gdf_notin_topo.shp'),
                   (gwr_gdf_notin_topo, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/gwr_gdf_notin_topo.shp'),
                   (pv_gdf_notin_topo, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/pv_gdf_notin_topo.shp'), 

                   (topo_gdf, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/topo_gdf.shp'), 
                   (single_part_houses_w_tilt, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/single_part_houses_w_tilt.shp'), 

                   (dsonodes_withegids_gdf, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/dsonodes_withegids_gdf.shp'),
                   (gm_gdf, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/gm_gdf.shp')
    ]

    for gdf, path in shp_to_export:
        try:
            gdf.to_file(path)
        except Exception as e:
            print(f"Failed to export {path}. Error: {e}")        


    # subset to > max n partitions -----------------------------------------------------
    max_partitions = pvalloc_settings['gwr_selection_specs']['solkat_max_n_partitions']
    topo_above_npart_gdf = copy.deepcopy(topo_gdf)
    counts = topo_above_npart_gdf['EGID'].value_counts()
    topo_above_npart_gdf['EGID_count'] = topo_above_npart_gdf['EGID'].map(counts)
    topo_above_npart_gdf = topo_above_npart_gdf[topo_above_npart_gdf['EGID_count'] > max_partitions]

    solkat_above_npart_gdf = copy.deepcopy(solkat_gdf_in_topo)
    solkat_gdf_in_topo[solkat_gdf_in_topo['df_uid'].isin(topo_df_uid_list)].copy()

    # export to shp -----------------------------------------------------
    shp_to_export2 = [(topo_above_npart_gdf, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/topo_above_{max_partitions}_npart_gdf.shp'),
                      (solkat_above_npart_gdf, f'{data_path_def}/output/pvalloc_run/topo_spatial_data/solkat_above_{max_partitions}_npart_gdf.shp')]
    for gdf, path in shp_to_export2:
        try:
            gdf.to_file(path)
        except Exception as e:
            print(f"Failed to export {path}. Error: {e}")

    print_to_logfile(f'Exported topo spatial data to shp files (with possible expections, see prints statments).', log_file_name_def)


# ------------------------------------------------------------------------------------------------------
# check multiple BUILT installations per EGID
# ------------------------------------------------------------------------------------------------------
def check_multiple_xtf_ids_per_EGID(
        pvalloc_settings, ):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']

    # import -----------------------------------------------------
    gwr_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gwr_gdf.geojson')
    pv_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/pv_gdf.geojson')
    Map_egid_pv = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_pv.parquet')

    check_egid = json.load(open(f'{data_path_def}/output/pvalloc_run/CHECK_egid_with_problems.json', 'r'))
    egid_list, issue_list= [], []
    for k,v in check_egid.items():
        egid_list.append(k)
        issue_list.append(v)
    check_df = pd.DataFrame({'EGID': egid_list,'issue': issue_list})

    check_df = check_df.loc[check_df['issue'] == 'multiple xtf_ids']

    # Map egid to xtf_id 
    multip_xtf_list = []
    for i, row in Map_egid_pv.iterrows():
        if row['EGID'] in check_df['EGID'].unique():
            multip_xtf_list.append(row['xtf_id'])
    
    multip_xtf_list_unique = list(set(multip_xtf_list))
    

    gwr_gdf_multiple_xtf_id = gwr_gdf[gwr_gdf['EGID'].isin(check_df['EGID'].unique())].copy()
    pv_gdf_multiple_xtf_id = pv_gdf[pv_gdf['xtf_id'].isin(multip_xtf_list_unique)].copy()

    # export to shp -----------------------------------------------------
    if not os.path.exists(f'{data_path_def}/output/pvalloc_run/topo_spatial_data'):
        os.makedirs(f'{data_path_def}/output/pvalloc_run/topo_spatial_data')
    
    gwr_gdf_multiple_xtf_id.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/gwr_gdf_multiple_xtf_id.shp')
    pv_gdf_multiple_xtf_id.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/pv_gdf_multiple_xtf_id.shp')


# ------------------------------------------------------------------------------------------------------
# sanity check for sample of EGIDs
# ------------------------------------------------------------------------------------------------------
def sanity_check_summary_byEGID(
        pvalloc_settings,
        subdir_path ):
    
    # setup -----------------------------------------------------
    name_dir_export_def = pvalloc_settings['name_dir_export']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    subdir_path_def = subdir_path
    log_file_name_def = pvalloc_settings['log_file_name']
    
    sanitycheck_summary_byEGID_specs = pvalloc_settings['sanitycheck_summary_byEGID_specs']

    # not needed because dir created in master file outside of function?
    # if not os.path.exists(f'{data_path_def}/output/pvalloc_run/sanity_check_byEGID'):
    #     os.makedirs(f'{data_path_def}/output/pvalloc_run/sanity_check_byEGID')


    # import -----------------------------------------------------
    topo = json.load(open(f'{subdir_path_def}/topo_egid.json', 'r'))
    npv_df = pd.read_parquet(f'{subdir_path_def}/npv_df.parquet')
    path_pred_inst = glob.glob(f'{subdir_path_def}/pred_npv_inst_by_M/pred_inst_df_*.parquet')
    pred_inst_df = pd.read_parquet(f'{subdir_path_def}/pred_inst_df.parquet')

    # add a EGID of model algorithm to the list
    if pred_inst_df.shape[0]< sanitycheck_summary_byEGID_specs['n_EGIDs_of_alloc_algorithm']:
        n_EGIDs_of_alloc_algorithm = list(np.random.choice(pred_inst_df['EGID'], pred_inst_df.shape[0], replace=False))
    else:
        n_EGIDs_of_alloc_algorithm = list(np.random.choice(pred_inst_df['EGID'], sanitycheck_summary_byEGID_specs['n_EGIDs_of_alloc_algorithm'], replace=False))
    pred_inst_df.loc[pred_inst_df['EGID'].isin(n_EGIDs_of_alloc_algorithm), ['EGID','info_source']]
    
    # remove any duplicates + add to pvalloc_settings
    pvalloc_settings['sanitycheck_summary_byEGID_specs']['egid_list'] = list(set(pvalloc_settings['sanitycheck_summary_byEGID_specs']['egid_list'] + n_EGIDs_of_alloc_algorithm ))
    

    # information extraction -----------------------------------------------------
    colnames = ['key', 'descr', 'partition_id', 'col1', 'col2', 'val', 'unit']
    def get_new_row():
        return {col: None for col in colnames}
    
    summary_toExcel_list = []
    # egid = sanitycheck_summary_byEGID_specs['egid_list'][3]
    for n_egid, egid in enumerate(sanitycheck_summary_byEGID_specs['egid_list']):
        if egid not in topo.keys():
            single_val_list = [row_egid_not_in_topo, ] = [get_new_row(), ]
            row_egid_not_in_topo['key'], row_egid_not_in_topo['descr'], row_egid_not_in_topo['val'] = 'EGID', 'EGID NOT in topo', egid
        elif egid in topo.keys():
            # single values ----------
            if True:
                single_val_list = [
                    row_egid, row_bfs, row_gklas, row_node, row_demand_type, 
                    row_pvinst_info, row_pvinst_BeginOp, row_pvinst_TotalPower,
                    row_elecpri, row_pvtarif, 
                    row_interest_rate, row_years_maturity, row_selfconsumption, row_pvprod_method, 
                    row_panel_efficiency, row_inverter_efficiency, row_kWpeak_per_m2, row_share_roof_area, 
                    empty_row ] = [get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                                get_new_row(), get_new_row(), get_new_row(), get_new_row(), 
                                get_new_row(), get_new_row(), get_new_row(), get_new_row(), 
                                get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                                get_new_row(), get_new_row(), get_new_row(),  ]
                
                # row_egid, row_bfs, row_gklas, row_node, row_demand_type = get_new_row(), get_new_row(), get_new_row(), get_new_row(), get_new_row()
                row_egid['key'], row_egid['descr'], row_egid['val'] = 'EGID', 'house identifier ID', egid
                row_bfs['key'], row_bfs['descr'], row_bfs['val'] = 'BFS', 'municipality identifier ID', topo.get(egid).get('gwr_info').get('bfs')
                row_gklas['key'], row_gklas['descr'], row_gklas['val'] = 'GKLAS', 'building type classification', topo.get(egid).get('gwr_info').get('gklas')
                row_node['key'], row_node['descr'], row_node['val'] = 'node', 'grid node identifier (artificial)', topo.get(egid).get('node')
                row_demand_type['key'], row_demand_type['descr'], row_demand_type['val'] = 'demand_type', 'type of artifical demand profile (Netflex, maybe CKW later)', topo.get(egid).get('demand_type')

                # row_pvinst_info, row_pvinst_BeginOp, row_pvinst_TotalPower = get_new_row(), get_new_row(), get_new_row() 
                row_pvinst_info['key'], row_pvinst_info['descr'], row_pvinst_info['val'] = 'pv_inst > info_source', 'Origin behind pv inst on house (real data or model alloc)', topo.get(egid).get('pv_inst').get('info_source')
                row_pvinst_BeginOp['key'], row_pvinst_BeginOp['descr'], row_pvinst_BeginOp['val'] = 'pv_inst > BeginOp', 'begin of operation', topo.get(egid).get('pv_inst').get('BeginOp')
                row_pvinst_TotalPower['key'], row_pvinst_TotalPower['descr'], row_pvinst_TotalPower['val'], row_pvinst_TotalPower['unit'] = 'pv_inst > TotalPower', 'total power of PV installation', topo.get(egid).get('pv_inst').get('TotalPower'), 'kW'

                # row_elecpri, row_pvtarif = get_new_row(), get_new_row() 
                row_elecpri['key'], row_elecpri['descr'], row_elecpri['val'], row_elecpri['unit'], row_elecpri['col1'], row_elecpri['col2'] = 'elecpri', 'mean electricity price per BFS area', topo.get(egid).get('elecpri_Rp_kWh'), 'Rp/kWh', f"elecpri_info: {topo.get(egid).get('elecpri_info')}",f"year: {pvalloc_settings.get('tech_economic_specs').get('elecpri_year')}"
                row_pvtarif['key'], row_pvtarif['descr'], row_pvtarif['val'], row_pvtarif['unit'], row_pvtarif['col1'], row_pvtarif['col2'] = 'pvtarif', 'tariff for PV feedin to EWR',topo.get(egid).get('pvtarif_Rp_kWh'), 'Rp/kWh', f"EWRs: {topo.get(egid).get('EWR').get('name')}", f"year: {pvalloc_settings.get('tech_economic_specs').get('pvtarif_year')}"
                row_interest_rate['key'], row_interest_rate['descr'],row_interest_rate['val'] = 'interest_rate', 'generic interest rate used for dicsounting NPV calculation', pvalloc_settings.get('tech_economic_specs').get('interest_rate')
                row_years_maturity['key'], row_years_maturity['descr'], row_years_maturity['val'] = 'invst_maturity', 'number of years that consider pv production for NPV calculation', pvalloc_settings.get('tech_economic_specs').get('invst_maturity')

                # row_selfconsumption, row_interest_rate, row_years_maturity, row_kWpeak_per_m2  = get_new_row(), get_new_row(), get_new_row(), get_new_row()
                row_selfconsumption['key'], row_selfconsumption['descr'], row_selfconsumption['val'] = 'self_consumption_ifapplicable', 'amount of production that can be consumed by the house at any hour during the year', pvalloc_settings.get('tech_economic_specs').get('self_consumption_ifapplicable')
                row_pvprod_method['key'], row_pvprod_method['descr'], row_pvprod_method['val'] = 'pvprod_calc_method', 'method used to calculate PV production', pvalloc_settings.get('tech_economic_specs').get('pvprod_calc_method')
                row_panel_efficiency['key'], row_panel_efficiency['descr'], row_panel_efficiency['val'] = 'panel_efficiency', 'transformation factor, how much solar energy can be transformed into electricity', pvalloc_settings.get('tech_economic_specs').get('panel_efficiency')
                row_inverter_efficiency['key'], row_inverter_efficiency['descr'], row_inverter_efficiency['val'] = 'inverter_efficiency', 'transformation factor, how much DC can be transformed into AC', pvalloc_settings.get('tech_economic_specs').get('inverter_efficiency')
                row_kWpeak_per_m2['key'], row_kWpeak_per_m2['descr'], row_kWpeak_per_m2['val'] = 'kWpeak_per_m2', 'transformation factor, how much kWp can be put on a square meter', pvalloc_settings.get('tech_economic_specs').get('kWpeak_per_m2')
                row_share_roof_area['key'], row_share_roof_area['descr'], row_share_roof_area['val'] = 'share_roof_area_available',  'share of roof area that can be effectively used for PV installation', pvalloc_settings.get('tech_economic_specs').get('share_roof_area_available')

            # df_uid (roof partition) values ----------   
            no_pv_TF = not topo.get(egid).get('pv_inst').get('inst_TF') 
            if no_pv_TF:
                npv_sub = npv_df.loc[npv_df['EGID'] == egid]
                npv_val_list = [
                    row_demand_kW,
                    row_FLAECHE_m2_mean, row_FLAECHE_m2_sum, 
                    row_AUSRICHTUNG_mean, row_AUSRICHTUNG_sum,
                    row_NEIGUNG_mean, row_NEIGUNG_sum,
                    row_pvprod_kW_min, row_pvprod_kW_max, row_pvprod_kW_mean, row_pvprod_kW_std, 
                    row_stromertrag_kWh_min, row_stromertrag_kWh_max, row_stromertrag_kWh_mean, row_stromertrag_kWh_std,
                    row_netfeedin_kW_min, row_netfeedin_kW_max, row_netfeedin_kW_mean, row_netfeedin_kW_std,
                    row_econ_inc_chf_min, row_econ_inc_chf_max, row_econ_inc_chf_mean, row_econ_inc_chf_std,
                    row_estim_pvinstcost_chf_min, row_estim_pvinstcost_chf_max, row_estim_pvinstcost_chf_mean, row_estim_pvinstcost_chf_std,
                    row_npv_chf_min, row_npv_chf_max, row_npv_chf_mean, row_npv_chf_std,
                    ] = [get_new_row(), 
                        get_new_row(), get_new_row(), 
                        get_new_row(), get_new_row(), 
                        get_new_row(), get_new_row(), 
                        get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                        get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                        get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                        get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                        get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                        get_new_row(), get_new_row(), get_new_row(), get_new_row(),]
                
                row_demand_kW['key'], row_demand_kW['descr'], row_demand_kW['val'], row_demand_kW['unit'] = 'demand_kW_min', 'total demand of house over 1 year', npv_sub['demand_kW'].mean(), 'kWh'

                row_pvprod_kW_mean['key'], row_pvprod_kW_mean['descr'], row_pvprod_kW_mean['val'], row_pvprod_kW_mean['unit'] = 'pvprod_kW_mean', 'mean of possible production within all partition combinations',  npv_sub['pvprod_kW'].mean(), 'kWh'
                row_pvprod_kW_std['key'], row_pvprod_kW_std['descr'], row_pvprod_kW_std['val'], row_pvprod_kW_std['unit'] = 'pvprod_kW_std', 'std of possible production within all partition combinations',  npv_sub['pvprod_kW'].std(), 'kWh'
                row_pvprod_kW_min['key'], row_pvprod_kW_min['descr'], row_pvprod_kW_min['val'], row_pvprod_kW_min['unit'] = 'pvprod_kW_min', 'min of possible production within all partition combinations',  npv_sub['pvprod_kW'].min(), 'kWh'
                row_pvprod_kW_max['key'], row_pvprod_kW_max['descr'], row_pvprod_kW_max['val'], row_pvprod_kW_max['unit'] = 'pvprod_kW_max', 'max of possible production within all partition combinations',  npv_sub['pvprod_kW'].max(), 'kWh'

                row_FLAECHE_m2_mean['key'], row_FLAECHE_m2_mean['descr'], row_FLAECHE_m2_mean['val'], row_FLAECHE_m2_mean['unit'] = 'FLAECHE_m2_mean', 'mean of possible roof area within all partition combinations',  npv_sub['FLAECHE'].mean(), 'm2'  
                row_FLAECHE_m2_sum['key'], row_FLAECHE_m2_sum['descr'], row_FLAECHE_m2_sum['val'], row_FLAECHE_m2_sum['unit'] = 'FLAECHE_m2_sum', 'sum of possible roof area within all partition combinations',  npv_sub['FLAECHE'].sum(), 'm2'
                row_AUSRICHTUNG_mean['key'], row_AUSRICHTUNG_mean['descr'], row_AUSRICHTUNG_mean['val'], row_AUSRICHTUNG_mean['unit'] = 'AUSRICHTUNG_mean', 'mean of possible orientation within all partition combinations',  npv_sub['AUSRICHTUNG'].mean(), 'degree'
                row_AUSRICHTUNG_sum['key'], row_AUSRICHTUNG_sum['descr'], row_AUSRICHTUNG_sum['val'], row_AUSRICHTUNG_sum['unit'] = 'AUSRICHTUNG_sum', 'sum of possible orientation within all partition combinations',  npv_sub['AUSRICHTUNG'].sum(), 'degree'
                row_NEIGUNG_mean['key'], row_NEIGUNG_mean['descr'], row_NEIGUNG_mean['val'], row_NEIGUNG_mean['unit'] = 'NEIGUNG_mean', 'mean of possible tilt within all partition combinations',  npv_sub['NEIGUNG'].mean(), 'degree'
                row_NEIGUNG_sum['key'], row_NEIGUNG_sum['descr'], row_NEIGUNG_sum['val'], row_NEIGUNG_sum['unit'] = 'NEIGUNG_sum', 'sum of possible tilt within all partition combinations',  npv_sub['NEIGUNG'].sum(), 'degree'

                # row_stromertrag_kWh_mean['key'], row_stromertrag_kWh_mean['descr'], row_stromertrag_kWh_mean['val'], row_stromertrag_kWh_mean['unit'] = 'STROMERTRAG_mean', 'mean of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].mean(), 'kWh/year'
                # row_stromertrag_kWh_std['key'], row_stromertrag_kWh_std['descr'], row_stromertrag_kWh_std['val'], row_stromertrag_kWh_std['unit'] = 'STROMERTRAG_std', 'std of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].std(), 'kWh/year'
                # row_stromertrag_kWh_min['key'], row_stromertrag_kWh_min['descr'], row_stromertrag_kWh_min['val'], row_stromertrag_kWh_min['unit'] = 'STROMERTRAG_min', 'min of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].min(), 'kWh/year'
                # row_stromertrag_kWh_max['key'], row_stromertrag_kWh_max['descr'], row_stromertrag_kWh_max['val'], row_stromertrag_kWh_max['unit'] = 'STROMERTRAG_max', 'max of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].max(), 'kWh/year'
                # BOOKMARK STROMERTRAG IS NOT IN NPV_DFL

                row_netfeedin_kW_mean['key'], row_netfeedin_kW_mean['descr'], row_netfeedin_kW_mean['val'], row_netfeedin_kW_mean['unit'] = 'netfeedin_kW_mean', 'mean of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].mean(), 'kWh'
                row_netfeedin_kW_std['key'], row_netfeedin_kW_std['descr'], row_netfeedin_kW_std['val'], row_netfeedin_kW_std['unit'] = 'netfeedin_kW_std', 'std of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].std(), 'kWh'
                row_netfeedin_kW_min['key'], row_netfeedin_kW_min['descr'], row_netfeedin_kW_min['val'], row_netfeedin_kW_min['unit'] = 'netfeedin_kW_min', 'min of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].min(), 'kWh'
                row_netfeedin_kW_max['key'], row_netfeedin_kW_max['descr'], row_netfeedin_kW_max['val'], row_netfeedin_kW_max['unit'] = 'netfeedin_kW_max', 'max of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].max(), 'kWh'
                
                row_econ_inc_chf_mean['key'], row_econ_inc_chf_mean['descr'], row_econ_inc_chf_mean['val'], row_econ_inc_chf_mean['unit'] = 'econ_inc_chf_mean', 'mean of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].mean(), 'CHF'
                row_econ_inc_chf_std['key'], row_econ_inc_chf_std['descr'], row_econ_inc_chf_std['val'], row_econ_inc_chf_std['unit'] = 'econ_inc_chf_std', 'std of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].std(), 'CHF'
                row_econ_inc_chf_min['key'], row_econ_inc_chf_min['descr'], row_econ_inc_chf_min['val'], row_econ_inc_chf_min['unit'] = 'econ_inc_chf_min', 'min of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].min(), 'CHF'
                row_econ_inc_chf_max['key'], row_econ_inc_chf_max['descr'], row_econ_inc_chf_max['val'], row_econ_inc_chf_max['unit'] = 'econ_inc_chf_max', 'max of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].max(), 'CHF'

                row_estim_pvinstcost_chf_mean['key'], row_estim_pvinstcost_chf_mean['descr'], row_estim_pvinstcost_chf_mean['val'], row_estim_pvinstcost_chf_mean['unit'] = 'estim_pvinstcost_chf_mean', 'mean of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].mean(), 'CHF'
                row_estim_pvinstcost_chf_std['key'], row_estim_pvinstcost_chf_std['descr'], row_estim_pvinstcost_chf_std['val'], row_estim_pvinstcost_chf_std['unit'] = 'estim_pvinstcost_chf_std', 'std of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].std(), 'CHF'
                row_estim_pvinstcost_chf_min['key'], row_estim_pvinstcost_chf_min['descr'], row_estim_pvinstcost_chf_min['val'], row_estim_pvinstcost_chf_min['unit'] = 'estim_pvinstcost_chf_min', 'min of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].min(), 'CHF'
                row_estim_pvinstcost_chf_max['key'], row_estim_pvinstcost_chf_max['descr'], row_estim_pvinstcost_chf_max['val'], row_estim_pvinstcost_chf_max['unit'] = 'estim_pvinstcost_chf_max', 'max of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].max(), 'CHF'

                row_npv_chf_mean['key'], row_npv_chf_mean['descr'], row_npv_chf_mean['val'], row_npv_chf_mean['unit'] = 'npv_chf_mean', 'mean of possible NPV within all partition combinations',  npv_sub['NPV_uid'].mean(), 'CHF'
                row_npv_chf_std['key'], row_npv_chf_std['descr'], row_npv_chf_std['val'], row_npv_chf_std['unit'] = 'npv_chf_std', 'std of possible NPV within all partition combinations',  npv_sub['NPV_uid'].std(), 'CHF'
                row_npv_chf_min['key'], row_npv_chf_min['descr'], row_npv_chf_min['val'], row_npv_chf_min['unit'] = 'npv_chf_min', 'min of possible NPV within all partition combinations',  npv_sub['NPV_uid'].min(), 'CHF'
                row_npv_chf_max['key'], row_npv_chf_max['descr'], row_npv_chf_max['val'], row_npv_chf_max['unit'] = 'npv_chf_max', 'max of possible NPV within all partition combinations',  npv_sub['NPV_uid'].max(), 'CHF'

            alloc_algo_pv_TF = topo.get(egid).get('pv_inst').get('inst_TF') and topo.get(egid).get('pv_inst').get('info_source') == 'alloc_algorithm'
            if alloc_algo_pv_TF:
                pred_inst_sub = pred_inst_df.loc[pred_inst_df['EGID'] == egid]
                npv_val_list = [
                    row_demand_kW,
                    row_df_uid,
                    row_pvprod_kW,
                    row_FLAECHE, 
                    row_FLAECH_angletilt,
                    row_AUSRICHTUNG,
                    row_NEIGUNG, 
                    row_STROMERTRAG_kWh,
                    row_netfeedin_kW, 
                    row_econ_inc_chf, 
                    row_estim_pvinstcost_chf, 
                    row_npv_chf
                    ] = [get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(), get_new_row(), get_new_row(),get_new_row(), ] 
                
                row_demand_kW['key'], row_demand_kW['descr'], row_demand_kW['val'], row_demand_kW['unit'] = 'demand_kW_min', 'total demand of house over 1 year', pred_inst_sub['demand_kW'].values[0], 'kWh'
                row_df_uid['key'], row_df_uid['descr'], row_df_uid['val'], row_df_uid['unit'] = 'df_uid', 'roof partition identifier ID', pred_inst_sub['df_uid_combo'].values[0], 'ID'

                row_pvprod_kW['key'], row_pvprod_kW['descr'], row_pvprod_kW['val'], row_pvprod_kW['unit'] = 'pvprod_kW', 'total production of house over 1 year', pred_inst_sub['pvprod_kW'].values[0], 'kWh'
                row_FLAECHE['key'], row_FLAECHE['descr'], row_FLAECHE['val'], row_FLAECHE['unit'] = 'FLAECHE_m2', 'total roof area of house', pred_inst_sub['FLAECHE'].values[0], 'm2'
                row_FLAECH_angletilt['key'], row_FLAECH_angletilt['descr'], row_FLAECH_angletilt['val'], row_FLAECH_angletilt['unit'] = 'FLAECH_angletilt', 'total roof area of house with angle tilt', pred_inst_sub['FLAECH_angletilt'].values[0], 'm2'
                row_AUSRICHTUNG['key'], row_AUSRICHTUNG['descr'], row_AUSRICHTUNG['val'], row_AUSRICHTUNG['unit'] = 'AUSRICHTUNG', 'total orientation of house', pred_inst_sub['AUSRICHTUNG'].values[0], 'degree'
                row_NEIGUNG['key'], row_NEIGUNG['descr'], row_NEIGUNG['val'], row_NEIGUNG['unit'] = 'NEIGUNG', 'total tilt of house', pred_inst_sub['NEIGUNG'].values[0], 'degree'
                row_STROMERTRAG_kWh['key'], row_STROMERTRAG_kWh['descr'], row_STROMERTRAG_kWh['val'], row_STROMERTRAG_kWh['unit'] = 'STROMERTRAG_kWh', 'total STROMERTRAG of house over 1 year', pred_inst_sub['STROMERTRAG'].values[0], 'kWh/year'
                # BOOKMARK STROMERTRAG IS NOT IN NPV_DFL
                row_netfeedin_kW['key'], row_netfeedin_kW['descr'], row_netfeedin_kW['val'], row_netfeedin_kW['unit'] = 'netfeedin_kW', 'total feedin of house over 1 year', pred_inst_sub['netfeedin_kW'].values[0], 'kWh'
                row_econ_inc_chf['key'], row_econ_inc_chf['descr'], row_econ_inc_chf['val'], row_econ_inc_chf['unit'] = 'econ_inc_chf', 'economic income of house over 1 year', pred_inst_sub['econ_inc_chf'].values[0], 'CHF'
                row_estim_pvinstcost_chf['key'], row_estim_pvinstcost_chf['descr'], row_estim_pvinstcost_chf['val'], row_estim_pvinstcost_chf['unit'] = 'estim_pvinstcost_chf', 'estimated installation costs of house over 1 year', pred_inst_sub['estim_pvinstcost_chf'].values[0], 'CHF'
                row_npv_chf['key'], row_npv_chf['descr'], row_npv_chf['val'], row_npv_chf['unit'] = 'npv_chf', 'net present value of house over 1 year', pred_inst_sub['NPV_uid'].values[0], 'CHF'
            
        
        # attache all rows to summary_df ----------
        summary_rows = []

        for row in single_val_list:
            summary_rows.append(row)
        if egid in topo.keys():
            if no_pv_TF or alloc_algo_pv_TF:
                for row in npv_val_list:
                    summary_rows.append(row)

        egid_summary_df = pd.DataFrame(summary_rows)
        egid_summary_df.to_csv(f'{data_path_def}/output/pvalloc_run/sanity_check_byEGID/summary_{egid}.csv')
        summary_toExcel_list.append(egid_summary_df)
   
    
    with pd.ExcelWriter(f'{data_path_def}/output/pvalloc_run/sanity_check_byEGID/summary_all{name_dir_export_def}.xlsx') as writer:
        for i, df in enumerate(summary_toExcel_list):
            sheet_egid = df.loc[df['key']=='EGID', 'val'].values[0]
            df.to_excel(writer, sheet_name=sheet_egid, index=False)

    checkpoint_to_logfile(f'exported summary for {len(sanitycheck_summary_byEGID_specs["egid_list"])} EGIDs to excel', log_file_name_def)
    
