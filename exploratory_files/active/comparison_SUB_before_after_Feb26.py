import sys
import os
import pandas as pd
import numpy as np
import json
import winsound
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

