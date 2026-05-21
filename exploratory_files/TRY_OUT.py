import sys
import os
import pandas as pd
import numpy as np
import json
import glob
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import polars as pl
from scipy import optimize
# import xlsxwriter
import sqlite3
import time


import copy
import plotly.graph_objects as go
import plotly.express as px


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ------------------------------------------------------------------------------------------------------
# parquet to csv
if True:
    path_list = [
        '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max/zMC1/pred_inst_df.parquet', 
        '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max/zMC1/constrcapa.parquet', 
        '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max/zMC1/npv_df.parquet', 
        
        '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_sBs0p8/zMC1/pred_inst_df.parquet', 
        '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_sBs0p8/zMC1/constrcapa.parquet', 
        '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_sBs0p8/zMC1/npv_df.parquet', 

        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_epzb1/zMC1/pred_inst_df.parquet', 
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_epzb1/zMC1/constrcapa.parquet', 
        
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1/zMC1/pred_inst_df.parquet', 
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1/zMC1/pred_inst_df.parquet', 
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2/zMC1/constrcapa.parquet', 
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2/zMC1/constrcapa.parquet', 
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3/zMC1/constrcapa.parquet', 
        # '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc/pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3/zMC1/constrcapa.parquet', 

        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max',                   'zMC1', 'pred_inst_df.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max',                   'zMC1', 'constrcapa.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_epzb1',             'zMC1', 'pred_inst_df.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_epzb1',             'zMC1', 'constrcapa.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1', 'zMC1', 'pred_inst_df.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1', 'zMC1', 'pred_inst_df.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2', 'zMC1', 'constrcapa.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2', 'zMC1', 'constrcapa.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3', 'zMC1', 'constrcapa.parquet' ), 
        # os.path.join(os.getcwd(), 'data', 'pvalloc', 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3', 'zMC1', 'constrcapa.parquet' ), 
                

        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_29nbfs_LRG2_max\zMC1\constrcapa.parquet",
        # r"C:\Users\hocrau00\Downloads\pvalloc_29nbfs_LRG2_max\zMC1\pred_npv_inst_by_M\pred_inst_df_10.parquet", 
        # r"C:\Users\hocrau00\Downloads\pvalloc_29nbfs_LRG2_max\zMC1\pred_npv_inst_by_M\npv_df_1.parquet", 

        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_16nbfs_RUR_max_gridoptim\zMC1_OptimExpa\npv_df.parquet",
        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_16nbfs_RUR_max_gridoptim\zMC1_OptimExpa\pred_inst_df.parquet",
        # r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max_OLDpreprep\zMC1\pred_npv_inst_by_M\npv_df_1.parquet", 
        # r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max_OLDpreprep\zMC1\pred_npv_inst_by_M\pred_inst_df_1.parquet"
        
        # r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max\zMC1\pred_npv_inst_by_M\pred_inst_df_1.parquet", 
        # r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max_OLDpreprep\zMC1\pred_npv_inst_by_M\pred_inst_df_1.parquet", 
        # r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI\gwr_all_building_df.parquet", 
        # r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI\gwr.parquet", 
        # r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr.parquet", 
        # r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr_all_building_df.parquet", 

        ]
    
    for pq_path in path_list:
        print(pq_path)
        export_to_pvalloc = True 
        if 'scicore' in pq_path:
            scen_name = pq_path.split('/')[-3]
            file_name = pq_path.split('/')[-1].split('.parquet')[0]
            if export_to_pvalloc:
                csv_path = '/scicore/home/krysiak/hocrau00/OptimalPV_RH/data/pvalloc'
            else:
                csv_path = "/".join(pq_path.split('/')[0:-1])
        
        else:
            scen_name = pq_path.split('\\')[-3]
            file_name = pq_path.split('\\')[-1].split('.parquet')[0]
            if export_to_pvalloc:
                csv_path = 'C:\Models\OptimalPV_RH\data\pvalloc'
            else:   
                csv_path = "\\".join(pq_path.split('/')[0:-1])

        df  = pd.read_parquet(pq_path)
        if any( [tag in pq_path for tag in ['OLDpreprep', 'COPYpreprep_used_untilFeb26',] ] ):
            export_path = f'{csv_path}\{scen_name}_{file_name}_OLDpreprep.csv'
        else:
            export_path = f'{csv_path}\{scen_name}_{file_name}.csv'

        # df.to_csv(export_path)
        df.to_excel(export_path.replace('.csv', '.xlsx'), index=False)



# ------------------------------------------------------------------------------------------------------
if False: 
    # save deprecated / faulty topo file
    topo_sub = json.load(open(r"C:\Users\hocrau00\Downloads\topo_egid_sub.json", 'r'))
    topo_rur = json.load(open(r"C:\Users\hocrau00\Downloads\topo_egid_DEV2_pvalloc_16nbfs_RUR_max.json", 'r'))
    topo_sub_befFeb26 = json.load(open(r"C:\Users\hocrau00\Downloads\topo_egid_sub_beforeFeb26.json", 'r'))
    topo_rur_befFeb26 = json.load(open(r"C:\Users\hocrau00\Downloads\topo_egid_rur_beforeFeb26.json", 'r'))

    constrcapa_sub = pd.read_parquet(r"C:\Users\hocrau00\Downloads\constrcapa_sub.parquet")
    constrcapa_rur = pd.read_parquet(r"C:\Users\hocrau00\Downloads\constrcapa_rur.parquet")
    constrcapa_sub_befFeb26 = pd.read_parquet(r"C:\Users\hocrau00\Downloads\constrcapa_sub_beforeFeb26.parquet")
    constrcapa_rur_befFeb26 = pd.read_parquet(r"C:\Users\hocrau00\Downloads\constrcapa_rur_beforeFeb26.parquet")

    gridnode_df_rur = pl.read_parquet(r"C:\Users\hocrau00\Downloads\gridnode_df_rur.parquet")
    gridnode_df_sub = pl.read_parquet(r"C:\Users\hocrau00\Downloads\gridnode_df_sub.parquet")
    gridnode_df_rur_befFeb26 = pl.read_parquet(r"C:\Users\hocrau00\Downloads\gridnode_df_rur_beforeFeb26.parquet")
    gridnode_df_sub_befFeb26 = pl.read_parquet(r"C:\Users\hocrau00\Downloads\gridnode_df_sub_beforeFeb26.parquet")


    def n_gridnode_in_topo_scen(topo):
        return pd.Series( [v['grid_node'] for v in topo.values()]).nunique()

    n_gridnode_sub = n_gridnode_in_topo_scen(topo_sub)
    n_gridnode_rur = n_gridnode_in_topo_scen(topo_rur)
    n_gridnode_sub_befFeb26 = n_gridnode_in_topo_scen(topo_sub_befFeb26)
    n_gridnode_rur_befFeb26 = n_gridnode_in_topo_scen(topo_rur_befFeb26)

    def summary_gridthreshold_scen(gridnode_df, t='t_1', round_decimal = 4):
        summary = (
            round(gridnode_df.filter(pl.col('t') == t).select(pl.col(['kW_threshold'])).min()['kW_threshold'][0],  round_decimal),
            round(gridnode_df.filter(pl.col('t') == t).select(pl.col(['kW_threshold'])).median()['kW_threshold'][0], round_decimal),
            round(gridnode_df.filter(pl.col('t') == t).select(pl.col(['kW_threshold'])).mean()['kW_threshold'][0], round_decimal),
            round(gridnode_df.filter(pl.col('t') == t).select(pl.col(['kW_threshold'])).max()['kW_threshold'][0], round_decimal),
        )
        return summary
    summary_gridthresh_sub = summary_gridthreshold_scen(gridnode_df_sub)
    summary_gridthresh_rur = summary_gridthreshold_scen(gridnode_df_rur)
    summary_gridthresh_sub_befFeb26 = summary_gridthreshold_scen(gridnode_df_sub_befFeb26)
    summary_gridthresh_rur_befFeb26 = summary_gridthreshold_scen(gridnode_df_rur_befFeb26)

    def nEGID_in_iter1(topo, date_iter1 = '2024-12-31'):
        return len([k for k, v in topo.items() if v['pv_inst']['BeginOp'] ==  date_iter1]) # and v['pv_inst']['info_source'] == 'alloc_algorithm'])
    nEGID_iter1_sub = nEGID_in_iter1(topo_sub)
    nEGID_iter1_rur = nEGID_in_iter1(topo_rur)
    nEGID_iter1_sub_befFeb26 = nEGID_in_iter1(topo_sub_befFeb26)
    nEGID_iter1_rur_befFeb26 = nEGID_in_iter1(topo_rur_befFeb26)

    def capacity_in_iter(topo, date_iter1 = '2024-12-31'):
            return sum([v['pv_inst']['TotalPower'] for k, v in topo.items() if v['pv_inst']['BeginOp'] ==  date_iter1 ]) #and v['pv_inst']['info_source'] == 'alloc_algorithm'])    
    instCap_iter1_sub = capacity_in_iter(topo_sub)
    instCap_iter1_rur = capacity_in_iter(topo_rur)
    instCap_iter1_sub_befFeb26 = capacity_in_iter(topo_sub_befFeb26)
    instCap_iter1_rur_befFeb26 = capacity_in_iter(topo_rur_befFeb26)


    scens = {
        'SUB': {
            'nEGID': len(topo_sub.keys()),
            'n_grid_node': n_gridnode_sub,
            'constracapa_kw': constrcapa_sub["constr_capacity_kw"][0],
            'summary_grid_tresh': summary_gridthresh_sub,
            'n_EGID_iter1': nEGID_iter1_sub,
            'instCap_iter1': instCap_iter1_sub,
        },
        'RUR': {
            'nEGID': len(topo_rur.keys()),
            'n_grid_node': n_gridnode_rur,
            'constracapa_kw': constrcapa_rur["constr_capacity_kw"][0],
            'summary_grid_tresh': summary_gridthresh_rur,
            'n_EGID_iter1': nEGID_iter1_rur,
            'instCap_iter1': instCap_iter1_rur,
        },
        'SUB_befFeb26': {
            'nEGID': len(topo_sub_befFeb26.keys()),
            'n_grid_node': n_gridnode_sub_befFeb26,
            'constracapa_kw': constrcapa_sub_befFeb26["constr_capacity_kw"][0],
            'summary_grid_tresh': summary_gridthresh_sub_befFeb26,
            'n_EGID_iter1': nEGID_iter1_sub_befFeb26,
            'instCap_iter1': instCap_iter1_sub_befFeb26,
        },
        'RUR_befFeb26': {
            'nEGID': len(topo_rur_befFeb26.keys()),
            'n_grid_node': n_gridnode_rur_befFeb26,
            'constracapa_kw': constrcapa_rur_befFeb26["constr_capacity_kw"][0],
            'summary_grid_tresh': summary_gridthresh_rur_befFeb26,
            'n_EGID_iter1': nEGID_iter1_rur_befFeb26,
            'instCap_iter1': instCap_iter1_rur_befFeb26,
        },
    }


    row_list = []
    for scen in [
        ('abs', 'SUB', '__'),
        # ('abs', 'RUR', '__'),
        # ('compare', 'RUR', 'SUB'),
        ('abs', 'SUB_befFeb26', '__'),
        # ('abs', 'RUR_befFeb26', '__'),
        # ('compare', 'RUR_befFeb26', 'SUB_befFeb26'),
        ('compare', 'SUB_befFeb26', 'SUB'),
    ]:
        
        if scen[0] == 'abs':
            row = {
                'scen' : scen[1],
                'nEGID': scens[scen[1]]['nEGID'],
                'n_grid_node': scens[scen[1]]['n_grid_node'],
                'constracapa_kw': scens[scen[1]]['constracapa_kw'],
                # 'min_gridthresh': scens[scen[1]]['summary_grid_tresh'][0],
                # 'med_gridthresh': scens[scen[1]]['summary_grid_tresh'][1],
                'mean_gridthresh': scens[scen[1]]['summary_grid_tresh'][2],
                # 'max_gridthresh': scens[scen[1]]['summary_grid_tresh'][3],
                'n_EGID_iter1': scens[scen[1]]['n_EGID_iter1'],
                'instCap_iter1': scens[scen[1]]['instCap_iter1'],
            }
        elif scen[0] == 'compare':
            row = {
                'scen' : f'{scen[1]} / {scen[2]}',
                'nEGID': scens[scen[1]]['nEGID'] / scens[scen[2]]['nEGID'],
                'n_grid_node': scens[scen[1]]['n_grid_node'] / scens[scen[2]]['n_grid_node'],
                'constracapa_kw': scens[scen[1]]['constracapa_kw'] / scens[scen[2]]['constracapa_kw'],
                # 'min_gridthresh': scens[scen[1]]['summary_grid_tresh'][0] / scens[scen[2]]['summary_grid_tresh'][0],
                # 'med_gridthresh': scens[scen[1]]['summary_grid_tresh'][1] / scens[scen[2]]['summary_grid_tresh'][1],
                'mean_gridthresh': scens[scen[1]]['summary_grid_tresh'][2] / scens[scen[2]]['summary_grid_tresh'][2],
                # 'max_gridthresh': scens[scen[1]]['summary_grid_tresh'][3] / scens[scen[2]]['summary_grid_tresh'][3],            
                'n_EGID_iter1': scens[scen[1]]['n_EGID_iter1'] / scens[scen[2]]['n_EGID_iter1'],
                'instCap_iter1': scens[scen[1]]['instCap_iter1'] / scens[scen[2]]['instCap_iter1'],
            }
        row_list.append(row)

    summary_df = pl.DataFrame(row_list)

    egid = '390391'
    topo_sub[egid]
    topo_sub_befFeb26[egid]
    topo_sub[egid]['gwr_info']
    topo_sub_befFeb26[egid]['gwr_info']

    topo_sub[egid]['gwr_info']
    topo_sub_befFeb26[egid]['gwr_info']

    topo_sub[egid]['pv_inst']
    topo_sub_befFeb26[egid]['pv_inst']

    topo_sub[egid]['solkat_partitions']
    topo_sub_befFeb26[egid]['solkat_partitions']

    solkat = gpd.read_file(r"C:\Users\hocrau00\Downloads\solkat_gdf_in_topo_sub.geojson")
    solkat.loc[solkat['EGID'] == egid, ['df_uid', 'EGID', 'STROMERTRAG','FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', ]]
    solkat.loc[solkat['EGID'] == egid, ['FLAECHE',]] / 10

    solkat_raw = pl.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_debugApr26\solkat.parquet")
    solkat_raw.filter(pl.col('DF_UID') == '10225986').select(pl.col(['FLAECHE', ])).item() / 10


    rows_list = []
    for egid in  [
        '390391', 
        '390392', 
        '390394', 
        '390398', 
        '390400', 
        '390402', 
        '390404', 

        # '390416' , 
        # '390426', '390403', 
        # '390405', '390417',
        # '390427'
        ]:
        row = {
            'scen': 'SUB',
            'egid': egid,
            'genh1' : topo_sub[egid]['gwr_info']['genh1'],
            'genh2' : topo_sub[egid]['gwr_info']['genh2'],
            'gwaerzh1' : topo_sub[egid]['gwr_info']['gwaerzh1'],
            'gwaerzh2' : topo_sub[egid]['gwr_info']['gwaerzh2'],
            'heating_system' : topo_sub[egid]['gwr_info']['heating_system'],
            'garea': topo_sub[egid]['gwr_info']['garea'],
        }
        rows_list.append(row)
        row_befFeb26 = {
            'scen': 'SUB_befFeb26',
            'egid': egid,
            'genh1' : topo_sub_befFeb26[egid]['gwr_info']['genh1'],
            'genh2' : topo_sub_befFeb26[egid]['gwr_info']['genh2'],
            'gwaerzh1' : topo_sub_befFeb26[egid]['gwr_info']['gwaerzh1'],
            'gwaerzh2' : topo_sub_befFeb26[egid]['gwr_info']['gwaerzh2'],
            'heating_system' : topo_sub_befFeb26[egid]['gwr_info']['heating_system'],
            'garea': topo_sub_befFeb26[egid]['gwr_info']['garea'],
        }
        rows_list.append(row_befFeb26)

    neighborhood = pl.DataFrame(rows_list)


    rows_check_gwaerzh7400_list = []
    for k, v in topo_sub.items():
        gwaerzh1 = v['gwr_info']['gwaerzh1']
        # gwaerzh2 = v['gwr_info']['gwaerzh2']
        gwaerzh2 = False
        if ( (gwaerzh1 == '7400') or (gwaerzh2 == '7400') ):
            row = {
                'scen': 'SUB',
                'egid': k,
                'gwaerzh1' : gwaerzh1,
                'gwaerzh2' : gwaerzh2,
            }
            rows_check_gwaerzh7400_list.append(row)
    check_gwaerzh7400_df = pl.DataFrame(rows_check_gwaerzh7400_list)



# ------------------------------------------------------------------------------------------------------

if False:
    scen_list = [
        'debug_4nodes_max__preprep_before_Feb26', 
        'debug_4nodes_max_wGBAUJminmax', 
    ]

    scen = scen_list[1]
    topo_df_comp_rows = []
    for scen in scen_list: 
        path_to_scen = os.path.join(os.getcwd(), 'data', 'pvalloc', scen )

        topo = json.load(open(os.path.join(path_to_scen, 'topo_egid.json'), 'r'))

        topo_df_rows = []
        for k, v in topo.items():
            for k_solkat, v_solkat in v['solkat_partitions'].items():
                row = {
                    'scen':             scen, 
                    'EGID':             k,
                    'BFS_NUMMER':       v['gwr_info']['bfs'],
                    'GKLAS':            v['gwr_info']['gklas'],
                    'GBAUJ':            v['gwr_info']['gbauj'],
                    'GAREA':            v['gwr_info']['garea'],
                    'dfuid':            k_solkat,
                    'FLAECHE':          v_solkat['FLAECHE'],
                    'STROMERTRAG':      v_solkat['STROMERTRAG'],
                }
                topo_df_rows.append(row)
                topo_df_comp_rows.append(row)

        # topo_df = pl.DataFrame(topo_df_rows)

    topo_comp_df = pl.DataFrame(topo_df_comp_rows)

    topo_max = topo_comp_df.filter(pl.col('scen') == 'debug_4nodes_max__preprep_before_Feb26')
    topo_wgbauj = topo_comp_df.filter(pl.col('scen') == 'debug_4nodes_max_wGBAUJminmax')

                                

    # FIND DIFFERENCES 
    agg = (
        topo_comp_df
        .group_by(['scen', 'EGID'])
        .agg([
            pl.col('FLAECHE').sum().alias('FLAECHE_sum'),
            pl.col('STROMERTRAG').sum().alias('STROMERTRAG_sum'),
            pl.col('dfuid').count().alias('n_partitions')
        ])
    )
    pivot = agg.pivot(
        values=['FLAECHE_sum', 'STROMERTRAG_sum', 'n_partitions'],
        index='EGID',
        columns='scen'
    )
    pivot = pivot.with_columns([
        (pl.col('FLAECHE_sum_debug_4nodes_max_wGBAUJminmax') -
        pl.col('FLAECHE_sum_debug_4nodes_max__preprep_before_Feb26')
        ).alias('delta_FLAECHE'),

        (pl.col('STROMERTRAG_sum_debug_4nodes_max_wGBAUJminmax') -
        pl.col('STROMERTRAG_sum_debug_4nodes_max__preprep_before_Feb26')
        ).alias('delta_STROMERTRAG'),
    ])
    tol = 1e-6
    egids_null_or_small = (
        pivot
        .filter(
            pl.col('delta_FLAECHE').is_null() |
            (pl.col('delta_FLAECHE').abs() > tol)
        )
        .select('EGID')
    )

    # EXPORT EGIDS WITH DIFFERENCES or missing in a df
    egids_diff__null_or_small = pivot.filter(pl.col('EGID').is_in(egids_null_or_small['EGID']))
    egids_diff__null_or_small.write_csv(os.path.join(os.getcwd(), 'data', 'pvalloc', 'egids_diff__null_or_small.csv'))
    egids_diff__null_or_small.write_excel(os.path.join(os.getcwd(), 'data', 'pvalloc', 'egids_diff__null_or_small.xlsx'))


    only_in_1_df = (
        topo_max.select('EGID').unique()
        .join(topo_wgbauj.select('EGID').unique(), on='EGID', how='anti')
    )
    only_in_2_df = (
        topo_wgbauj.select('EGID').unique()
        .join(topo_max.select('EGID').unique(), on='EGID', how='anti')
    )
    only_once = list(only_in_1_df['EGID']) + list(only_in_2_df['EGID']) 
    egids_diff__missing = topo_comp_df.filter(pl.col('EGID').is_in(only_once))
    egids_diff__missing.write_csv(os.path.join(os.getcwd(), 'data', 'pvalloc', 'egids_diff__missing.csv'))
    egids_diff__missing.write_excel(os.path.join(os.getcwd(), 'data', 'pvalloc', 'egids_diff__missing.xlsx'))






# ------------------------------------------------------------------------------------------------------
if False:
        # check scens for individual numbers
        scen_list = [
            'DEV2_pvalloc_16nbfs_RUR_max', 
            'DEV_pvalloc_16nbfs_RUR_max_prepFeb26', 
        ]
        egids_to_check = [
            # '2129223',
            '190630208',
        ]
        df_uids = [
            # 10855180, 
            10853811, 
            10853812, 
        ]
        for scen in scen_list: 
            for egid in egids_to_check:
                print(egid)
                solkat = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'input_split_data_geometry', 'solkat_pq.parquet'))
                solkat.dtypes
                solkat_dfuids = solkat.loc[solkat['DF_UID'].isin(df_uids), ['DF_UID', 'GWR_EGID', 'SB_UUID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', ]]
                solkat_dfuids
                divided_flaeche_str = ''

                for denom in [5, 4, 3, 2, 1]:
                    f'denom: {denom}; flaeche {solkat_dfuids["FLAECHE"].sum() / denom}'
                    print(f'denom: {denom}; flaeche {solkat_dfuids["FLAECHE"].sum() / denom}')

                solkat_scen_new = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'preprep', 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26', 'solkat.parquet'))
                solkat_scen_dfuids_new = solkat_scen_new.loc[solkat_scen_new['DF_UID'].isin([str(int(dfuid)) for dfuid in df_uids]), ['DF_UID', 'EGID', 'SB_UUID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', ]]
                solkat_scen_dfuids_new
                gwr_new = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'preprep', 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26', 'gwr.parquet'))
                gwr_new.loc[gwr_new['EGID'].isin(['388682', '388683','388684', '388681']), ['EGID', 'GSTAT', 'GKLAS', 'GBAUJ', ]]


                solkat_scen_old = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'preprep', 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26', 'solkat.parquet'))
                solkat_scen_dfuids_old = solkat_scen_old.loc[solkat_scen_old['DF_UID'].isin([str(int(dfuid)) for dfuid in df_uids]), ['DF_UID', 'EGID', 'SB_UUID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', 'BFS_NUMMER', ]]
                solkat_scen_dfuids_old
                gwr_old = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'preprep', 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26', 'gwr.parquet'))
                gwr_old.loc[gwr_old['EGID'].isin(['388682', '388683','388684', '388681']), ['EGID', 'GSTAT', 'GKLAS', 'GBAUJ', ]]


# ------------------------------------------------------------------------------------------------------
# check SUB_max results with new and old preprep
if False:
    solkat_pl               = pl.read_parquet(  r"C:\Models\OptimalPV_RH\data\input_split_data_geometry\solkat_pq.parquet")

    npv_df                  = pl.read_parquet( r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max\zMC1\pred_npv_inst_by_M\pred_inst_df_1.parquet")
    npv_df_OLD              = pl.read_parquet( r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max_OLDpreprep\zMC1\pred_npv_inst_by_M\pred_inst_df_1.parquet")

    topo                    = json.load(open(  r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max\topo_egid.json"))
    topo_OLD                = json.load(open(  r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max_OLDpreprep\topo_egid.json"))

    solkat_gdf_in_topo      = gpd.read_file(   r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max\topo_spatial_data\solkat_gdf_in_topo.geojson")
    solkat_gdf_in_topo_OLD  = gpd.read_file(   r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_10nbfs_SUB_max_OLDpreprep\topo_spatial_data\solkat_gdf_in_topo.geojson")

    gwr                     = pl.read_parquet( r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26\gwr.parquet")
    gwr_OLD                 = pl.read_parquet( r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr.parquet")

    gwr_all_building_df     = pl.read_parquet( r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26\gwr_all_building_df.parquet")
    gwr_all_building_df_OLD = pl.read_parquet( r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr_all_building_df.parquet")

    solkat_gdf              = gpd.read_file(   r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26\solkat_gdf.geojson")
    solkat_gdf_OLD          = gpd.read_file(   r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\solkat_gdf.geojson")


    print(f"\n\n{'NEW':<30.30} {'':>6} | OLD                                       "  )
    print(f"{'npv_df shape':<30.30} {npv_df.shape[0]:>6} | {'npv_df_OLD shape':<30.30} {npv_df_OLD.shape[0]:>6}  | "  )
    # print(f"{'topo:':<30.30} {len(topo.keys()):>6} | {'topo_OLD:':<30.30} {len(topo_OLD.keys()):>6}  | {len(topo_OLD.keys())/len(topo.keys()):.2%} "  )
    # print(f"{'solkat_in_topo shape':<30.30} {solkat_gdf_in_topo.shape[0]:>6} | {'solkat_in_topo_OLD shape':<30.30} {solkat_gdf_in_topo_OLD.shape[0]:>6}  | {solkat_gdf_in_topo_OLD.shape[0]/solkat_gdf_in_topo.shape[0]:.2%} "  )
    print(f"{'gwr shape':<30.30} {gwr.shape[0]:>6} | {'gwr_OLD shape':<30.30} {gwr_OLD.shape[0]:>6}  | {gwr_OLD.shape[0]/gwr.shape[0]:.2%} "  )
    print(f"{'gwr_all_building_df shape':<30.30} {gwr_all_building_df.shape[0]:>6} | {'gwr_all_building_df_OLD shape':<30.30} {gwr_all_building_df_OLD.shape[0]:>6}  | {gwr_all_building_df_OLD.shape[0]/gwr_all_building_df.shape[0]:.2%} "  )
    print(f"{'solkat shape':<30.30} {solkat_gdf.shape[0]:>6} | {'solkat_OLD shape':<30.30} {solkat_gdf_OLD.shape[0]:>6}  | {solkat_gdf_OLD.shape[0]/solkat_gdf.shape[0]:.2%} "  )
    print(f"{'solkat unique':<30.30} {solkat_gdf['EGID'].nunique():>6} | {'solkat_OLD unique':<30.30} {solkat_gdf_OLD['EGID'].nunique():>6}  | {solkat_gdf_OLD['EGID'].nunique()/solkat_gdf['EGID'].nunique():.2%} "  )

    # single case analysis
    egid = '390437'
    sb_uuid = solkat_gdf.loc[solkat_gdf['EGID'] == egid, 'SB_UUID'].values[0]

    solkat_gdf.columns
    # solkat_gdf.loc[solkat_gdf['EGID'] == egid, ['DF_UID', 'DF_NUMMER', 'SB_UUID', 'SB_OBJEKTART', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'GSTRAHLUNG','STROMERTRAG',]]
    solkat_gdf.loc[solkat_gdf['EGID'] == egid, ['EGID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'GSTRAHLUNG','STROMERTRAG',]]
    solkat_gdf_in_topo.loc[solkat_gdf_in_topo['EGID'] == egid, ['EGID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'GSTRAHLUNG','STROMERTRAG',]]

    solkat_gdf_in_topo.loc[solkat_gdf_in_topo['SB_UUID'] == sb_uuid, ['EGID', 'SB_UUID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'GSTRAHLUNG','STROMERTRAG',]]
    solkat_pl.filter(pl.col('SB_UUID') == sb_uuid).select([ 'SB_UUID', 'KLASSE', 'FLAECHE', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'GSTRAHLUNG','STROMERTRAG',]).to_pandas()


    solkat_gdf.shape




# ------------------------------------------------------------------------------------------------------
# find max sample region within DSO network
if False: 
    preprep_path = r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI"

    dsonodes_df         = pl.read_parquet(f'{preprep_path}/dsonodes_df.parquet')
    Map_egid_dsonode    = pl.read_parquet(f'{preprep_path}/Map_egid_dsonode.parquet')
    gwr_all_building_df = pl.read_parquet(f'{preprep_path}/gwr_all_building_df.parquet')

    gm_shp              = gpd.read_file(f'{preprep_path}/gm_shp_gdf.geojson')

    merge = Map_egid_dsonode.join(gwr_all_building_df, left_on='EGID', right_on='EGID', how='inner')    
    ALLDSO_bfs_list = [int(bfs) for bfs in list(merge['GGDENR'].unique()) ]

    RUR_bfs_list =[
        # RURAL
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
    ]
    SUB_bfs_list = [
        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
    ]
    # v5 -> T0_prediction: 2024
    LRG_bfs_list = [
        # RURAL 
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
        # SUBURBAN
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
        # URBAN
        2773, 2769, 2770,
        ]
    XLRG_bfs_list = [
        # RURAL 
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
        # SUBURBAN
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
        2471, 2481, 2775, 2764, 2771, 
        2763, 2473, 2475, 2474, 2472, 
        2478, 2830, 2766, 2767, 2774, 
        # URBAN
        2773, 2769, 2770,
        2762, 2765, 
        ]

    sample_lists = {
        'RUR': RUR_bfs_list,
        'SUB': SUB_bfs_list,
        'LRG': LRG_bfs_list,
        'XLRG': XLRG_bfs_list,
        'ALLDSO': ALLDSO_bfs_list,
    }

    for k, v in sample_lists.items():
        model_sample = gm_shp.loc[gm_shp['BFS_NUMMER'].isin(v), ]
        model_sample.to_file(f'{preprep_path}/model_sample_2_{k}.geojson', driver='GeoJSON')
        print(f'exported model_sample_{k}.geojson')


# ------------------------------------------------------------------------------------------------------

if False: 
    bfs_sample = [                                                    2773, 2769, 2770,                                     # URBAN: Reinach, Münchenstein, Muttenz
                                                        2767, 2771, 2775, 2764,                               # SEMI-URBAN: Bottmingen, Oberwil, Therwil, Biel-Benken
                                                        # 2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
                                                        2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren
    ]
    bfs_str = [str(bfs) for bfs in bfs_sample ]

    solkat = pl.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_22to23_extSolkatEGID_aggrfarms\solkat.parquet")
    gwr_all_buildings = pl.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_22to23_extSolkatEGID_aggrfarms\gwr_all_building_df.parquet")

    solkat.shape
    solkat.columns
    solkat.dtypes
    gwr_all_buildings.shape

    solkat_gwr = solkat.join(gwr_all_buildings, on='EGID', how='left')
    solkat_gwr = solkat_gwr.with_columns([
        pl.col('GBAUJ').replace('', '0').cast(pl.Int64).alias('GBAUJ'),
    ])


    solkat_gwr = solkat_gwr.filter(
        (pl.col('BFS_NUMMER').is_in(bfs_str)) & 
        (pl.col('GSTAT').is_in(['1001', '1002', '1003', '1004'])) &
        (pl.col('GKLAS').is_in(['1110', '1121', '1122',])) &
        (pl.col('GBAUJ') < 2021) &
        (pl.col('GBAUJ') > 1950) 
    )


    solkat_gwr = solkat_gwr.with_columns([
        pl.col('AUSRICHTUNG').abs().alias('AUSRICHTUNG_abs'),
    ])

    solkat_gwr.sort(by=['NEIGUNG', 'AUSRICHTUNG_abs', ], descending=[False, False, ]).write_csv(r"C:\Models\OptimalPV_RH\data\solkat_bfs_sample_FILTERED.csv")
    # solkat_rur.sort(by=['NEIGUNG', 'AUSRICHTUNG_abs', ], descending=[False, False, ]).write_excel(r"C:\Models\OptimalPV_RH\solkat_bfs_sample_FILTERED.xlsx")

# ------------------------------------------------------------------------------------------------------
# aggregation stats for GWR

if False:
    # kt selection
    kt_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,]

    wd_path = 'C:/Models/OptimalPV_RH'
    data_path     = f'{wd_path}/data'

    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')
    bfs_numbers = gm_shp.loc[gm_shp['KANTONSNUM'].isin(kt_numbers), 'BFS_NUMMER'].astype(int).tolist()


    # get ALL BUILDING data
    query_columns = [
        'EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE',
        'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ',
        'GANZWHG','GWAERZH1', 'GWAERZH2', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1',
        'GEBF', 'GAREA'
        ]
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in bfs_numbers])

    conn = sqlite3.connect(f'{data_path}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
    # cur.execute(f'SELECT * FROM building WHERE GGDENR IN ({query_bfs_numbers})')
    sqlrows = cur.fetchall()
    conn.close()

    gwr_df = pl.DataFrame(sqlrows, schema=query_columns, orient="row")

    # transformations
    gwr_df.columns
    gwr_df.dtypes
    gwr_df = gwr_df.with_columns([
        pl.when(pl.col('GEBF') == "").then(pl.lit('0')).otherwise(pl.col('GEBF')).cast(pl.Float64).alias('GEBF'),
        pl.when(pl.col('GAREA') == "").then(pl.lit('0')).otherwise(pl.col('GAREA')).cast(pl.Float64).alias('GAREA'),
    ])

    gwr_df = gwr_df.with_columns([
        pl.lit(0).cast(pl.Int64).alias('HP1_tf'),
        pl.lit(0).cast(pl.Int64).alias('HP2_tf'),
        pl.lit(0).cast(pl.Int64).alias('HPjoint_tf')
    ])
    gwr_df = gwr_df.with_columns([
        pl.when((pl.col('GWAERZH1') == '7410') | (pl.col('GWAERZH1') == '7411'))
        .then(1)
        .otherwise(0)
        .alias('HP1_tf'),

        pl.when((pl.col('GWAERZH2') == '7410') | (pl.col('GWAERZH2') == '7411'))
        .then(1)
        .otherwise(0)
        .alias('HP2_tf'),

        pl.when(
            (pl.col('GWAERZH1') == '7410') |
            (pl.col('GWAERZH1') == '7411') |
            (pl.col('GWAERZH2') == '7410') |
            (pl.col('GWAERZH2') == '7411')
        )
        .then(1)
        .otherwise(0)
        .alias('HPjoint_tf'),
    ])

    print(f'finisched gwr import - {time.time()}')


    # gwr_agg = gwr_df.groupby(['GSTAT', 'GKLAS', 'GBAUJ', 'HPjoint_TF' ]).agg(
    #     {
    #         'EGID': 'count',
    #         'GEBF': 'sum',
    #         'GAREA': 'sum',
    #         'HP1_tf': 'sum',
    #         'HP2_tf': 'sum',
    #     }
    # ).copy()
    # gwr_agg.rename(columns={'EGID': 'n_EGIDs', 'GEBF': 'GEBF_sum', 'GAREA': 'GAREA_sum', 'HP1_tf': 'n_HP1', 'HP2_tf': 'n_HP2', }, inplace=True)
    # gwr_agg.to_excel(r"C:\Models\Future_Markets\0_gwr_agg_allCH.xlsx")



    gwr_agg_gbauj = gwr_df.group_by(['GBAUJ', ]).agg(
        [
            pl.count('EGID').alias('n_EGIDs'),
            pl.sum('GEBF').alias('GEBF_sum'),
            pl.sum('GAREA').alias('GAREA_sum'),
            pl.sum('HP1_tf').alias('n_HP1'),
            pl.sum('HP2_tf').alias('n_HP2'),
        ])
    gwr_agg_gbauj.write_csv(r"C:\Models\Future_Markets\0_gwr_agg_allCH_GBAUJ.csv")


    gwr_gstat_selec = gwr_df.filter(pl.col('GSTAT').is_in([
        # '1001', '1002', '1003', 
        '1004', 
        ])).clone()
    gwr_agg_gklas = gwr_gstat_selec.group_by(['GKLAS', 'GSTAT']).agg(
        [
            pl.count('EGID').alias('n_EGIDs'),
            pl.sum('GEBF').alias('GEBF_sum'),
            pl.sum('GAREA').alias('GAREA_sum'),
            pl.sum('HP1_tf').alias('n_HP1'),
            pl.sum('HP2_tf').alias('n_HP2'),
        ])
    # gwr_agg_gklas.to_excel(r"C:\Models\Future_Markets\0_gwr_agg_allCH_GSTAT.xlsx")
    gwr_agg_gklas.write_csv(r"C:\Models\Future_Markets\0_gwr_agg_allCH_GSTAT.csv")
    print(f'exported <0_gwr_agg_allCH_GSTAT.xlsx> - {time.time()}')


    # grid import
    grid_df = pd.read_excel(r"Q:\_shared\Projekt - Optimal PV Expantionspaths\Daten_Primeo_x_UniBasel_V2.0.xlsx", )
    grid_df = pl.DataFrame(grid_df)
    grid_df = grid_df.with_columns([
        pl.col('EGID').cast(pl.Utf8),
    ])
    grid_df = grid_df.unique(subset=['EGID'], keep='first')

    grid_join_df = grid_df.join(gwr_df, on = 'EGID', how='left')
    print(f'finished grid import and merge with gwr - {time.time()}')
    grid_join_df.write_csv(r"C:\Models\Future_Markets\0_grid_join_df.csv")


    # grid_agg = grid_df.groupby(['GSTAT', 'GKLAS', 'GBAUJ', 'HPjoint_TF' ]).agg(
    #     {
    #         'EGID': 'count',
    #         'GEBF': 'sum',
    #         'GAREA': 'sum',
    #         'HP1_tf': 'sum',
    #         'HP2_tf': 'sum',
    #         'ID_Trafostation': 'nunique',
    #         'Trafoleistung_kVA': 'sum',
    #     }
    # ).copy()
    # grid_agg.rename(columns={'EGID': 'n_EGIDs', 'GEBF': 'GEBF_sum', 'GAREA': 'GAREA_sum', 'HP1_tf': 'n_HP1', 'HP2_tf': 'n_HP2', 'ID_Travostation': 'n_Trafos', 'Trafleistung_kVA': 'Trafleistung_kVA_sum' }, inplace=True)
    # grid_agg.to_excel(r"C:\Models\Future_Markets\0_time.timemeo.xlsx")

    grid_join_gstat_df = grid_df.join(gwr_gstat_selec, on = 'EGID', how='left', )
    # grid_join_gstat_df = grid_df.join(gwr_df, on = 'EGID', how='left', )
    grid_groupby_trafoID = grid_join_gstat_df.group_by([ 'ID_Trafostation']).agg(
        [   
            pl.count('EGID').alias('n_EGIDs'),
            pl.count('GKLAS').alias('n_GKLAS'),
            pl.count('GSTAT').alias('n_GSTAT'),

            pl.sum('GEBF').alias('GEBF_sum'),
            pl.sum('GAREA').alias('GAREA_sum'),
            pl.sum('HP1_tf').alias('n_HP1'),
            pl.sum('HP2_tf').alias('n_HP2'),
            pl.sum('HPjoint_tf').alias('n_HPjoint'),
            pl.n_unique('ID_Trafostation').alias('nunique_Trafos'),
            pl.sum('Trafoleistung_kVA').alias('Trafoleistung_kVA_sum'),
        ])

    grid_groupby_trafoID.write_csv(r"C:\Models\Future_Markets\0_grid_groupby_trafoID.csv")
    print(f'exported <0_grid_agg_gklas.csv> - {time.time()}')




# ------------------------------------------------------------------------------------------------------

# CONVERT parquet to csv
if False:

    pq_path = r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_mini_aggr_RUR_max_2_old_vers\zMC1\npv_df.parquet"
    file_name = pq_path.split('\\')[-1].split('.parquet')[0]
    csv_path = "\\".join(pq_path.split('\\')[0:-1])

    pq_path.split('\\')[0:-1]

    df  = pd.read_parquet(pq_path)
    # df = df.loc[df['EGID'].isin(['400415',])]
    # df = df.head(8760 * 40)
    df.to_csv(f'{csv_path}/{file_name}.csv')
    print(f'exported {file_name}.csv')


# ------------------------------------------------------------------------------------------------------

if False: 
    # import geopandas as gpd
    import pandas as pd
    import numpy as np
    from shapely.geometry import Point, MultiPoint
    import plotly.express as px
    import json

    # Step 1: Create sample GeoDataFrame with random points in central Europe
    np.random.seed(42)
    lons = np.random.uniform(5.0, 10.0, 10)
    lats = np.random.uniform(46.0, 49.0, 10)
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

    gdf = gpd.GeoDataFrame({
        'id': range(1, 11),
        'name': [f"Point {i}" for i in range(1, 11)],
    }, geometry=points, crs='EPSG:4326')

    # Step 2: Create convex hull polygon
    multipoint = MultiPoint(gdf.geometry.tolist())
    hull_polygon = multipoint.convex_hull

    hull_gdf = gpd.GeoDataFrame(geometry=[hull_polygon], crs='EPSG:4326')

    # Step 3: Convert to GeoJSON for plotly
    geojson_hull = json.loads(hull_gdf.to_json())

    # Step 4: Plot hull using px.choropleth_mapbox
    fig = px.choropleth_mapbox(
        hull_gdf,
        geojson=geojson_hull,
        locations=hull_gdf.index.astype(str),
        color_discrete_sequence=["lightblue"],
        center=dict(lat=gdf.geometry.y.mean(), lon=gdf.geometry.x.mean()),
        zoom=6,
        opacity=0.4,
        mapbox_style="carto-positron"
    )

    # Optional: Add the original points for reference
    import plotly.graph_objects as go
    fig.add_trace(go.Scattermapbox(
        lat=gdf.geometry.y,
        lon=gdf.geometry.x,
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Points'
    ))

    # Compute the median longitude
    median_lon = gdf.geometry.x.median()

    # Assign groups based on longitude
    gdf['group'] = np.where(gdf.geometry.x <= median_lon, 'A', 'B')


    # Define color mapping
    group_colors = {'A': 'green', 'B': 'orange'}

    from shapely.geometry import MultiPoint

    # Create two GeoDataFrames by group
    gdf_A = gdf[gdf['group'] == 'A']
    gdf_B = gdf[gdf['group'] == 'B']

    # Generate convex hulls for each group
    hull_A = MultiPoint(gdf_A.geometry.tolist()).convex_hull
    hull_B = MultiPoint(gdf_B.geometry.tolist()).convex_hull

    # Create GeoDataFrames for each hull
    hull_A_gdf = gpd.GeoDataFrame(geometry=[hull_A], crs='EPSG:4326')
    hull_B_gdf = gpd.GeoDataFrame(geometry=[hull_B], crs='EPSG:4326')

    import json

    hull_A_geojson = json.loads(hull_A_gdf.to_json())
    hull_B_geojson = json.loads(hull_B_gdf.to_json())

    # Extract trace from px figure
    hull_A_trace = px.choropleth_mapbox(
        hull_A_gdf,
        geojson=hull_A_geojson,
        locations=hull_A_gdf.index.astype(str),
        color_discrete_sequence=["rgba(0, 255, 0, 0.3)"],
    ).data[0]
    hull_A_trace.name = "Hull A"
    fig.add_trace(hull_A_trace)

    # Same for Hull B
    hull_B_trace = px.choropleth_mapbox(
        hull_B_gdf,
        geojson=hull_B_geojson,
        locations=hull_B_gdf.index.astype(str),
        color_discrete_sequence=["rgba(255, 165, 0, 0.3)"],
    ).data[0]
    hull_B_trace.name = "Hull B"
    fig.add_trace(hull_B_trace)

    fig.show()





# ------------------------------------------------------------------------------------------------------
if False: 
    wd_path = 'C:/Models/OptimalPV_RH'
    data_path     = f'{wd_path}/data'
    # data_path_def = f'{wd_path}_data'
    scen = "pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd"

    subdf_selected_list = []
    pq_paths = glob.glob(f'{data_path}/pvalloc/{scen}/topo_time_subdf/*.parquet')
    path = pq_paths[0]
    for path in pq_paths:
        subdf = pd.read_parquet(path)
        # subdf = subdf.loc[
        subdf['df_uid']

    pq_path = r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd\topo_time_subdf\topo_subdf_0to399.parquet"
    file_name = pq_path.split('\\')[-1].split('.parquet')[0]
    csv_path = "\\".join(pq_path.split('\\')[0:-1])

    pq_path.split('\\')[0:-1]

    df  = pd.read_parquet(pq_path)
    df = df.head(8760 * 40)
    df.to_csv(f'{csv_path}/{file_name}.csv')
    print(f'exported {file_name}.csv')

# ------------------------------------------------------------------------------------------------------
if False:
    # pv_all_gdf_raw = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
    # pv_all_gdf = gpd.read_file(f'{data_path}/input_split_data_geometry/pv_bsblso_geo.geojson')
    def get_bfsnr_name_tuple_list(bfs_number_list=None):
            
        BFS_NUMMER_BL_tuple = [
            (2761, 'Aesch (BL)'),
            (2762, 'Allschwil'),
            (2841, 'Anwil'),
            (2881, 'Arboldswil'),
            (2821, 'Arisdorf'),
            (2763, 'Arlesheim'),
            (2822, 'Augst'),
            (2842, 'Bennwil'),
            (2764, 'Biel-Benken'),
            (2765, 'Binningen'),
            (2766, 'Birsfelden'),
            (2781, 'Blauen'),
            (2842, 'Böckten'),
            (2767, 'Bottmingen'),
            (2883, 'Bretzwil'),
            (2782, 'Brislach'),
            (2823, 'Bubendorf'),
            (2843, 'Buckten'),
            (2783, 'Burg im Leimental'),
            (2844, 'Buus'),
            (2884, 'Diegten'),
            (2845, 'Diepflingen'),
            (2784, 'Dittingen'),
            (2785, 'Duggingen'),
            (2885, 'Eptingen'),
            (2768, 'Ettingen'),
            (2824, 'Frenkendorf'),
            (2825, 'Füllinsdorf'),
            (2846, 'Gelterkinden'),
            (2826, 'Giebenach'),
            (2786, 'Grellingen'),
            (2847, 'Häfelfingen'),
            (2848, 'Hemmiken'),
            (2827, 'Hersberg'),
            (2886, 'Hölstein'),
            (2849, 'Itingen'),
            (2850, 'Känerkinden'),
            (2851, 'Kilchberg (BL)'),
            (2887, 'Lampenberg'),
            (2888, 'Langenbruck'),
            (2852, 'Läufelfingen'),
            (2787, 'Laufen'),
            (2828, 'Lausen'),
            (2889, 'Lauwil'),
            (2890, 'Liedertswil'),
            (2788, 'Liesberg'),
            (2829, 'Liestal'),
            (2830, 'Lupsingen'),
            (2853, 'Maisprach'),
            (2769, 'Münchenstein'),
            (2770, 'Muttenz'),
            (2789, 'Nenzlingen'),
            (2891, 'Niederdorf'),
            (2854, 'Nusshof'),
            (2892, 'Oberdorf (BL)'),
            (2771, 'Oberwil (BL)'),
            (2855, 'Oltingen'),
            (2856, 'Ormalingen'),
            (2772, 'Pfeffingen'),
            (2831, 'Pratteln'),
            (2832, 'Ramlinsburg'),
            (2893, 'Reigoldswil'),
            (2773, 'Reinach (BL)'),
            (2857, 'Rickenbach (BL)'),
            (2790, 'Roggenburg'),
            (2791, 'Röschenz'),
            (2858, 'Rothenfluh'),
            (2859, 'Rümlingen'),
            (2860, 'Rünenberg'),
            (2774, 'Schönenbuch'),
            (2833, 'Seltisberg'),
            (2861, 'Sissach'),
            (2862, 'Tecknau'),
            (2863, 'Tenniken'),
            (2775, 'Therwil'),
            (2864, 'Thürnen'),
            (2894, 'Titterten'),
            (2792, 'Wahlen'),
            (2895, 'Waldenburg'),
            (2865, 'Wenslingen'),
            (2866, 'Wintersingen'),
            (2867, 'Wittinsburg'),
            (2868, 'Zeglingen'),
            (2834, 'Ziefen'),
            (2869, 'Zunzgen'),
            (2793, 'Zwingen'),
        ]
        BFS_NUMMER_AG_tuple = [
            (2421, 'Aedermannsdorf'),
            (2511, 'Aeschi (SO)'),
            (2541, 'Balm bei Günsberg'),
            (2422, 'Balsthal'),
            (2611, 'Bärschwil'),
            (2471, 'Bättwil'),
            (2612, 'Beinwil (SO)'),
            (2542, 'Bellach'),
            (2543, 'Bettlach'),
            (2513, 'Biberist'),
            (2445, 'Biezwil'),
            (2514, 'Bolken'),
            (2571, 'Boningen'),
            (2613, 'Breitenbach'),
            (2465, 'Buchegg'),
            (2472, 'Büren (SO)'),
            (2614, 'Büsserach'),
            (2572, 'Däniken'),
            (2516, 'Deitingen'),
            (2517, 'Derendingen'),
            (2473, 'Dornach'),
            (2535, 'Drei Höfe'),
            (2573, 'Dulliken'),
            (2401, 'Egerkingen'),
            (2574, 'Eppenberg-Wöschnau'),
            (2503, 'Erlinsbach (SO)'),
            (2615, 'Erschwil'),
            (2518, 'Etziken'),
            (2616, 'Fehren'),
            (2544, 'Feldbrunnen-St. Niklaus'),
            (2545, 'Flumenthal'),
            (2575, 'Fulenbach'),
            (2474, 'Gempen'),
            (2519, 'Gerlafingen'),
            (2546, 'Grenchen'),
            (2576, 'Gretzenbach'),
            (2617, 'Grindel'),
            (2547, 'Günsberg'),
            (2578, 'Gunzgen'),
            (2579, 'Hägendorf'),
            (2520, 'Halten'),
            (2402, 'Härkingen'),
            (2491, 'Hauenstein-Ifenthal'),
            (2424, 'Herbetswil'),
            (2618, 'Himmelried'),
            (2475, 'Hochwald'),
            (2476, 'Hofstetten-Flüh'),
            (2425, 'Holderbank (SO)'),
            (2523, 'Horriwil'),
            (2523, 'Horriwil'),
            (2548, 'Hubersdorf'),
            (2524, 'Hüniken'),
            (2549, 'Kammersrohr'),
            (2580, 'Kappel (SO)'),
            (2403, 'Kestenholz'),
            (2492, 'Kienberg'),
            (2619, 'Kleinlützel'),
            (2525, 'Kriegstetten'),
            (2550, 'Langendorf'),
            (2426, 'Laupersdorf'),
            (2526, 'Lohn-Ammannsegg'),
            (2551, 'Lommiswil'),
            (2493, 'Lostorf'),
            (2464, 'Lüsslingen-Nennigkofen'),
            (2527, 'Luterbach'),
            (2455, 'Lüterkofen-Ichertswil'),
            (2427, 'Matzendorf'),
            (2620, 'Meltingen'),
            (2457, 'Messen'),
            (2477, 'Metzerlen-Mariastein'),
            (2428, 'Mümliswil-Ramiswil'),
            (2404, 'Neuendorf'),
            (2405, 'Niederbuchsiten'),
            (2495, 'Niedergösgen'),
            (2478, 'Nuglar-St. Pantaleon'),
            (2621, 'Nunningen'),
            (2406, 'Oberbuchsiten'),
            (2553, 'Oberdorf (SO)'),
            (2528, 'Obergerlafingen'),
            (2497, 'Obergösgen'),
            (2529, 'Oekingen'),
            (2407, 'Oensingen'),
            (2581, 'Olten'),
            (2530, 'Recherswil'),
            (2582, 'Rickenbach (SO)'),
            (2554, 'Riedholz'),
            (2479, 'Rodersdorf'),
            (2555, 'Rüttenen'),
            (2461, 'Schnottwil'),
            (2583, 'Schönenwerd'),
            (2480, 'Seewen'),
            (2556, 'Selzach'),
            (2601, 'Solothurn'),
            (2584, 'Starrkirch-Wil'),
            (2499, 'Stüsslingen'),
            (2532, 'Subingen'),
            (2500, 'Trimbach'),
            (2463, 'Unterramsern'),
            (2585, 'Walterswil (SO)'),
            (2586, 'Wangen bei Olten'),
            (2430, 'Welschenrohr-Gänsbrunnen'),
            (2501, 'Winznau'),
            (2502, 'Wisen (SO)'),
            (2481, 'Witterswil'),
            (2408, 'Wolfwil'),
            (2534, 'Zuchwil'),
            (2622, 'Zullwil')
        ]
        
        BFS_all_tuple = BFS_NUMMER_BL_tuple + BFS_NUMMER_AG_tuple
        if isinstance(bfs_number_list, list):
            bfsnr_name_tuple_list = [x for x in BFS_all_tuple if x[0] in bfs_number_list]
        elif bfs_number_list == None:
            bfsnr_name_tuple_list = BFS_all_tuple

        return bfsnr_name_tuple_list

    def flatten_geometry(geom):
        if geom.has_z:
            if geom.geom_type == 'Polygon':
                exterior = [(x, y) for x, y, z in geom.exterior.coords]
                interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                return Polygon(exterior, interiors)
            elif geom.geom_type == 'MultiPolygon':
                return MultiPolygon([flatten_geometry(poly) for poly in geom.geoms])
        return geom

    pv_df = pd.read_parquet(f'{data_path}/input_split_data_geometry/pv_pq.parquet')

    gwr_bsblso_pq = pd.read_parquet(f'{data_path}/input_split_data_geometry/gwr_bsblso_pq.parquet')
    Map_egid_dsonode = pd.read_parquet(f'{data_path}/preprep/preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates/Map_egid_dsonode.parquet')

    gwr_in_primeo = gwr_bsblso_pq.loc[gwr_bsblso_pq['EGID'].isin(Map_egid_dsonode['EGID'].unique())]
    bfs_in_primeo = gwr_in_primeo['GGDENR'].unique()


    pv_df.dtypes
    Map_egid_dsonode.dtypes
    pv_in_primeo = copy.deepcopy(pv_df.loc[pv_df['BFS_NUMMER'].isin(bfs_in_primeo)])

    pv_in_primeo.rename(columns={'BeginningOfOperation': 'BeginOp', }, inplace=True)

    pv_in_primeo['BeginOp'] = pd.to_datetime(pv_in_primeo['BeginOp'], format='%Y-%m-%d')
    pv_in_primeo['BeginOp_year'] = pv_in_primeo['BeginOp'].dt.to_period('Y')
    pv_in_primeo = pv_in_primeo.groupby(['BeginOp_year', 'BFS_NUMMER'])['TotalPower'].sum().reset_index().copy()
    pv_in_primeo['BeginOp_year'] = pv_in_primeo['BeginOp_year'].dt.to_timestamp()

    fig = go.Figure()
    for b in pv_in_primeo['BFS_NUMMER'].unique():
        subdf = pv_in_primeo.loc[pv_in_primeo['BFS_NUMMER'] == b]
        b_name = get_bfsnr_name_tuple_list([int(b),])[0][1]
        fig.add_trace(go.Scatter(x=subdf['BeginOp_year'], y=subdf['TotalPower'], mode='lines+markers', name=f'{b}_{b_name}'))

    fig.update_layout(title='PV Total Power per year (all BFS in DSO grid)', xaxis_title='Year', yaxis_title='Total Power')
    fig.show()
    fig.write_html(f'{data_path}/pvinstCap_in_primeo_BFS.html')



    # add gemeinde mapplot
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')
    gm_shp = gm_shp.to_crs('EPSG:4326')
    gm_shp['geometry'] = gm_shp['geometry'].apply(flatten_geometry)

    gm_shp['BFS_NUMMER'] = gm_shp['BFS_NUMMER'].astype(str)
    date_cols = [col for col in gm_shp.columns if (gm_shp[col].dtype == 'datetime64[ns]') or (gm_shp[col].dtype == 'datetime64[ms]')]
    gm_shp.drop(columns=date_cols, inplace=True)
    gm_shp = gm_shp.loc[gm_shp['BFS_NUMMER'].isin(bfs_in_primeo)]

    pv_mrg = pv_in_primeo.merge(gm_shp, how='left', on='BFS_NUMMER')


    t0_row = []
    for bfs in pv_mrg['BFS_NUMMER'].unique():
        subdf = pv_mrg.loc[pv_mrg['BFS_NUMMER'] == bfs]
        subdf_t0 = subdf.loc[subdf['BeginOp_year'] == subdf['BeginOp_year'].min()]
        t0_row.append(subdf_t0)

    pv_t0_gdf = pd.concat(t0_row)
    pv_mrg_gdf = gpd.GeoDataFrame(pv_t0_gdf, geometry=pv_t0_gdf['geometry'], crs=gm_shp.crs)
    pv_mrg_gdf['BeginOp_year'] = pv_mrg_gdf['BeginOp_year'].astype(str)

    pv_mrg_gdf['hover_text'] = pv_mrg_gdf.apply(lambda row: f'BFS: {row["BFS_NUMMER"]}, {get_bfsnr_name_tuple_list([int(row["BFS_NUMMER"]),])[0][1]}<br>t0_TotalPower: {row["TotalPower"]} kWp', axis=1)

    # pv_mrg_gdf = pv_mrg_gdf.to_crs('EPSG:4326')
    geojson = json.loads(pv_mrg_gdf.to_json())


    map = px.choropleth_mapbox()
    for year in pv_mrg_gdf['BeginOp_year'].unique():
        subdf = pv_mrg_gdf.loc[pv_mrg_gdf['BeginOp_year'] == year]
        map.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=subdf['BFS_NUMMER'],
                z=subdf['TotalPower'],
                hoverinfo='text',
                hovertemplate=subdf['hover_text'],
                marker_opacity=0.5,
                marker_line_width=0,
                name=str(year),
            )
        )
    map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=7,
        mapbox_center={"lat": 47.5, "lon": 7.5},
        title_text="PV Total Power per year (all BFS in DSO grid)",
        title_x=0.5,
    )
    map.show()




# instcomp_year_df['BeginOp'] = pd.to_datetime(instcomp_year_df['BeginOp'], format='%Y-%m-%d')
# instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp'].dt.to_period('Y')
# instcomp_year_df = instcomp_year_df.groupby(['BeginOp_year',])['TotalPower'].sum().reset_index().copy()
# instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp_year'].dt.to_timestamp()
# instcomp_year_df['Cumm_TotalPower'] = instcomp_year_df['TotalPower'].cumsum()
# instcomp_year_df['growth_cumm_TotalPower'] = instcomp_year_df['Cumm_TotalPower'].diff() / instcomp_year_df['Cumm_TotalPower'].shift(1) 
# instcomp_year_df[['Cumm_TotalPower', 'growth_cumm_TotalPower']] 




# ------------------------------------------------------------------------------------------------------
if False:
    egid = '410298'

    topo_subdf_pq_list = glob.glob(f'{data_path}/output/{scen}/topo_time_subdf/*.parquet')
    df_list= []
    for f in topo_subdf_pq_list:
        subdf = pd.read_parquet(f)
        if egid in subdf['EGID'].unique():
            #topo_time_subdf
            subdf = subdf.loc[subdf['EGID'] == egid]
            df_list.append(subdf)

            f_name = f.split(f'{data_path}/output/{scen}/topo_time_subdf\\')[1].split('.parquet')[0]
            print(f'EGID: {f_name} -> export csv')

    df = pd.concat(df_list)
    df.to_csv(f'{data_path}/output/{scen}/topo_time_subdf/{f_name}.csv')

    # solkat_month
    solkat_month = pd.read_parquet(f'{data_path}/output/{scen}/solkat_month.parquet')
    df.columns
    dfuid_ls = df['df_uid'].unique()
    solkat_month.loc[solkat_month['DF_UID'].isin(dfuid_ls)].to_csv(f'{data_path}/output/{scen}/topo_time_subdf/solkat_month_{egid}.csv') 


# ------------------------------------------------------------------------------------------------------
# print directory scheme to txt file
if False:
    from pathlib import Path

    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '


    def tree(dir_path: Path, prefix: str='', exclude_list = None):
        """A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        if exclude_list is None:
            exclude_list = []

        contents = [p for p in dir_path.iterdir() if p.name not in exclude_list and not p.name.startswith('.')]
        # contents each get pointers that are ├── with a final └── :
        pointers = ['├── '] * (len(contents) - 1) + ['└── ']
        for pointer, path in zip(pointers, contents):
            yield prefix + pointer + path.name
            if path.is_dir():  # extend the prefix and recurse:
                extension = '│   ' if pointer == '├── ' else '    '
                # i.e. space because last, └── , above so no more |
                yield from tree(path, prefix=prefix+extension, exclude_list=exclude_list)

    # Print the directory tree excluding specified directories and those starting with "."
    txt_header = f'** Directory structure for OptimalPV_RH **\n date: {pd.Timestamp.now()}\n\n'

    with open(f'{wd_path}/OptimalPV_RH_directory_structure.txt', 'w') as f:
        f.write(txt_header)
        for line in tree(Path('C:/Models/OptimalPV_RH'), exclude_list=['archiv_no_longer_used']):
            print(line)
            f.write(line + '\n')





# ------------------------------------------------------------------------------------------------------
"""
solkat_preprep_wo_missingEGID_gdf = gpd.read_file(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes\solkat_gdf.geojson")
solkat_preprep_incl_missingEGID_gdf = gpd.read_file(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\solkat_gdf.geojson")
gwr = pd.read_parquet(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\gwr.parquet")
# solkat = pd.read_parquet(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\solkat.parquet")


cols_to_check = ['391293', '391294', '391295', '391296', '391297', ]
# cols_to_check = ['391291', '391290']
subdf_preprep_incl_missingEGID = solkat_preprep_incl_missingEGID_gdf[solkat_preprep_incl_missingEGID_gdf['EGID'].isin(cols_to_check)]
subdf_preprep_wo_missingEGID = solkat_preprep_wo_missingEGID_gdf[solkat_preprep_wo_missingEGID_gdf['EGID'].isin(cols_to_check)]

subdf_preprep_incl_missingEGID.loc[:, ['EGID', 'geometry', 'FLAECHE', 'DF_UID']]
subdf_preprep_wo_missingEGID.loc[:, ['EGID', 'geometry', 'FLAECHE', 'DF_UID']]

subdf_preprep_wo_missingEGID.buffer(-0.5, resolution=16)

isin_ls = ['245054165', ' 245054165', '245054165 ', ' 245054165 ']
gwr.loc[gwr['EGID'].isin(isin_ls)]['EGID'].unique()

solkat_preprep_incl_missingEGID_gdf.loc[solkat_preprep_incl_missingEGID_gdf['EGID'].isin(isin_ls)]['EGID'].unique()
"""

# ------------------------------------------------------------------------------------------------------
"""
gpd.list_layers(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg')
solkat_month = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='SOLKAT_CH_DACH_MONAT', rows = 100000)
month = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='MONAT', rows = 1000)

solkat_month.to_csv(f'{wd_path}/solkat_month.csv')
solkat_month.dtypes
type(solkat_month)
solkat_month.head(20)
month.columns
                             
"""
# ------------------------------------------------------------------------------------------------------
"""
topo = json.load(open(f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'r'))

topo[list(topo.keys())[0]].get('pv_inst').get('info_source')

egid_ls, alloc_algorithm_ls = [], []
for k,v in topo.items():
    egid_ls.append(k)
    alloc_algorithm_ls.append(v.get('pv_inst').get('info_source'))

df = pd.DataFrame({'EGID': egid_ls, 'info_source': alloc_algorithm_ls})
df['info_source'].value_counts()
egids = df.loc[df['info_source'] == 'alloc_algorithm', 'EGID'].to_list()
egids 

# -----------------------
subdf_t0['EGID'].isin(egids).sum()
subdf['EGID'].isin(egids).sum()

aggsubdf_combo['EGID'].isin(egids).sum()

npv_df['EGID'].isin(egids).sum()
"""

# ------------------------------------------------------------------------------------------------------
"""

topo = json.load(open(f'{data_path}/output/{scen}/topo_egid.json', 'r'))

# topo characteristics
topo[list(topo.keys())[0]].keys()
topo[list(topo.keys())[0]]['pv_inst']


all_topo_m = glob.glob(f'{data_path}/output/pvalloc_smallBL_SLCTN_npv_weighted/interim_predictions/topo*.json')

for f in all_topo_m: 

    topo = json.load(open(f, 'r'))
    print(f'\ncounts for {f.split("topo_")[-1].split(".json")[0]}')

    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    for k,v in topo.items():
        # print(k)
        egid_list.append(k)
        gklas_list.append(v['gwr_info']['gklas'])
        inst_tf_list.append(v['pv_inst']['inst_TF'])
        inst_info_list.append(v['pv_inst']['info_source'])
        if 'xtf_id' in v['pv_inst']:
            inst_id_list.append(v['pv_inst']['xtf_id'])
        else:   
            inst_id_list.append('')
        beginop_list.append(v['pv_inst']['BeginOp'])
        power_list.append(v['pv_inst']['TotalPower'])
    # for ls in [egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list]:
    #     print(len(ls))

    topo_df = pd.DataFrame({'egid': egid_list, 'gklas': gklas_list, 'inst_tf': inst_tf_list, 'inst_info': inst_info_list, 
                            'inst_id': inst_id_list, 'beginop': beginop_list, 'power': power_list})
    
"""

if False:
    topo = json.load(open(f'{data_path}/output\pvalloc_smallBL_1y_npv_weighted/topo_egid.json', 'r'))
    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    for k,v in topo.items():
        # print(k)
        egid_list.append(k)
        gklas_list.append(v['gwr_info']['gklas'])
        inst_tf_list.append(v['pv_inst']['inst_TF'])
        inst_info_list.append(v['pv_inst']['info_source'])
        if 'xtf_id' in v['pv_inst']:
            inst_id_list.append(v['pv_inst']['xtf_id'])
        else:   
            inst_id_list.append('')
        beginop_list.append(v['pv_inst']['BeginOp'])
        power_list.append(v['pv_inst']['TotalPower'])

    topo_df = pd.DataFrame({'egid': egid_list, 'gklas': gklas_list, 'inst_tf': inst_tf_list, 'inst_info': inst_info_list,
                            'inst_id': inst_id_list, 'beginop': beginop_list, 'power': power_list})

    # topo_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_df.parquet')

    topo_df.to_csv(f'{wd_path}/topo3_df.csv')
# ------------------------------------------------------------------------------------------------------
# Theoretical change in NPV distribution over multiple allocation algorithms
if False:
        
    import statistics as stats
    import matplotlib.pyplot as plt
    import plotly.figure_factory as ff

    def rand_skew_norm(fAlpha, fLocation, fScale):
        sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

        afRN = np.random.randn(2)
        u0 = afRN[0]
        v = afRN[1]
        u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

        if u0 >= 0:
            return u1*fScale + fLocation 
        return (-u1)*fScale + fLocation 

    def randn_skew(N, skew=0.0):
        return [rand_skew_norm(skew, 0, 1) for x in range(N)]

    n_sample = 10**5
    df_before = pd.DataFrame({'skew3': randn_skew(n_sample, 3), 'skew0': randn_skew(n_sample, 0)})
    df_before['stand'] = (df_before['skew3'] / df_before['skew3'].max())
    df_before['id'] = df_before.index

    df = df_before.copy()
    df_pick_list = []
    draws = 1000
    print('\n\nstart loop')
    for i in range(1, draws+1):
        if (i) % (draws/4)==0:
            print(f'{i/ draws * 100}% done')
        rand_num = np.random.uniform(0, 1)
        df['stand'] = (df['skew3'] / df['skew3'].max())
        df['diff_stand_rand'] = abs(df['stand'] - rand_num)
        df_pick  = df[df['diff_stand_rand'] == min(df['diff_stand_rand'])].copy()

        if df_pick.shape[0] > 1:
            print('more than one row picked')
            rand_row = np.random.randint(0, df_pick.shape[0])
            df_pick = df_pick.iloc[rand_row]
        # adjust df
        df_pick_list.append(df_pick)
        df = df.drop(df_pick.index)
    df_picked = pd.concat(df_pick_list)

    # hist_data = [df_before['skew0'],df['skew0'], df_before['stand'], df['stand'],]
    # labels = ['skew0_before', 'skew0_after', 'stand_before', 'stand_after']

    hist_data = [df_before['skew3'],df['skew3'], df_picked['skew3'], df_before['stand'], df['stand'], ]
    labels = ['skew3_before', 'skew3_after', 'skew3_picked', 'stand_before', 'stand_after']
    df_picked['skew3'].var()
    df_picked['stand'].var()

    df_before.shape, df.shape, df_picked.shape
    print('create fig')
    fig = ff.create_distplot(hist_data, labels, bin_size=0.005)
    fig.show()
    print('end loop')


# ------------------------------------------------------------------------------------------------------






