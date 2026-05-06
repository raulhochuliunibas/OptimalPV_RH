import sys
import os
import pandas as pd
import numpy as np
import json
import glob
import polars as pl
# import xlsxwriter

import shutil
import plotly.graph_objects as go
import plotly.express as px


def update_facet_titles(fig, prefix='preprep_scen='):
    for annotation in fig.layout.annotations:
        if annotation.text.startswith(prefix):
            annotation.text = annotation.text.replace(prefix, '')

    return fig


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


pvalloc_preprep_scen_tupls = [
    # ('debug_5nbfs_NOgbauj__preprep_NOgbauj',                     'split_mini_geodata__5nbfs_NOgbauj'),
    # ('debug_5nbfs_NOgbauj__preprep_Wgbauj',                      'split_mini_geodata__5nbfs_Wgbauj'),
    # ('debug_5nbfs_Wgbauj__preprep_Wgbauj',                       'split_mini_geodata__5nbfs_Wgbauj'),
    
    # ('pvalloc_29nbfs_LRG2_max',                  'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI__before_Feb26'),
    # ('pvalloc_29nbfs_LRG2_max_epzb0_50',         'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI__before_Feb26'),

    # ('DEV2_pvalloc_29nbfs_LRG2_max',             'preprep_BLSO_15to24_extSolkatEGID__May26'),
    # # ('DEV2_pvalloc_29nbfs_LRG2_max_epzb1',       'preprep_BLSO_15to24_extSolkatEGID__May26'),
    # ('DEV2_pvalloc_29nbfs_LRG2_max_epzb0_5',     'preprep_BLSO_15to24_extSolkatEGID__May26'),
    # ('DEV2_pvalloc_29nbfs_LRG2_max_hist_constr', 'preprep_BLSO_15to24_extSolkatEGID__May26'),
    
    ('DEV2_pvalloc_16nbfs_RUR_max',             'preprep_BLSO_15to24_extSolkatEGID__May26'),
    ('x_DEV2_pvalloc_10nbfs_SUB_max_11_old_vers',             'preprep_BLSO_15to24_extSolkatEGID__May26'),

    # ('debug_5nbfs__preprep_Wgbauj__EGID_gwradded_T',      'split_mini_geodata__5nbfs_Wgbauj__EGID_gwradded_T'),
    
    ]

data_extracted_to_comp_dir = True


# copy relevant files to seperate dir --------------------------------------------------
run_on_scicore = True if 'scicore' in os.getcwd() else False
extract_dir_path = os.path.join(os.getcwd(), 'data', 'comparison_GBAUJ_preprep_pvalloc')
if run_on_scicore:
    os.makedirs(extract_dir_path, exist_ok=True)
    for scen, preprep_scen in pvalloc_preprep_scen_tupls:
        # topos
        topo_src_path = os.path.join(os.getcwd(), 'data', 'pvalloc', scen, 'zMC1', 'topo_egid.json')
        shutil.copy(topo_src_path, os.path.join(extract_dir_path, f'topo_egid_{scen}.json') )

        # gwr all
        gwr_all_src_path = os.path.join(os.getcwd(), 'data', 'preprep', preprep_scen, 'gwr_all_building_df.parquet')
        shutil.copy(gwr_all_src_path, os.path.join(extract_dir_path, f'gwr_all_building_df_{preprep_scen}.parquet') )

        # gwr select
        gwr_select_src_path = os.path.join(os.getcwd(), 'data', 'preprep', preprep_scen, 'gwr.parquet') 
        shutil.copy(gwr_select_src_path, os.path.join(extract_dir_path, f'gwr_{preprep_scen}.parquet') )

        





scen_list           =  pd.Series([scen          for scen, preprep_scen in pvalloc_preprep_scen_tupls] ).unique().tolist()
preprep_scen_list   =  pd.Series([preprep_scen  for scen, preprep_scen in pvalloc_preprep_scen_tupls] ).unique().tolist()

# analize pvalloc TOPO data files --------------------------------------------------
gridnode_agg_df = pl.DataFrame()
for scen in scen_list:

    if data_extracted_to_comp_dir:
        # topo = json.load(open(os.path.join(extract_dir_path, f'topo_egid_{scen}.json'), 'r'))
        
        file_path = os.path.join(extract_dir_path, f'topo_egid_{scen}.json')
        print(f"Scen: {scen}; File size: {os.path.getsize(file_path)} bytes")
        try:
            topo = json.load(open(os.path.join(extract_dir_path, f'topo_egid_{scen}.json'), 'r'))
            
        except json.JSONDecodeError as e:
            print(f"JSON error at line {e.lineno}, column {e.colno}: {e.msg}")
            print(f"Character position: {e.pos}")
    else:
        topo = json.load(open(os.path.join('data', 'pvalloc', scen, 'zMC1', 'topo_egid.json'), 'r'))
    
    topo_df_rows = []
    for k, v in topo.items():
        # for k, v in topo_lrg.items():
        gbauj = v['gwr_info']['gbauj']
        gbauj_tf = True if gbauj != '' else False

        row = {
            'scen':         scen ,
            'EGID':         k , 
            'grid_node':     v['grid_node'] , 
            'GBAUJ' :       v['gwr_info']['gbauj'] ,
            'has_GBAUJ' :   gbauj_tf ,
            'no_GBAUJ':     not gbauj_tf, 
            'n_dfuid':      len(v['solkat_partitions']) , 
        }
        topo_df_rows.append(row)

    topo_df = pl.DataFrame(topo_df_rows)
    sum(topo_df['no_GBAUJ'])

    scen_gridnode_df = topo_df.group_by( ['scen', 'grid_node'] ).agg([
        pl.col('EGID').count().alias('n_EGID'), 
        pl.col('n_dfuid').sum().alias('n_sum_dfuid'), 
        pl.col('has_GBAUJ').sum().alias('has_GBAUJ'), 
        pl.col('no_GBAUJ').sum().alias('no_GBAUJ'), 
    ])
    gridnode_agg_df = pl.concat([gridnode_agg_df, scen_gridnode_df])


# plot pvalloc TOPO -----------------------------
melted_pvalloc_df = gridnode_agg_df.unpivot(
    on=['has_GBAUJ', 'no_GBAUJ'],
    index=['scen', 'grid_node'],
    variable_name='gbauj_status',
    value_name='count',
)
fig = px.bar(
    melted_pvalloc_df,
    x='grid_node',
    y='count',
    color='gbauj_status',
    barmode='stack',
    facet_col='scen',
    title='topo_egids (pvalloc): has_GBAUJ vs no_GBAUJ per grid_node by scenario',
)
fig = update_facet_titles(fig, prefix='scen=')
fig.show() if not run_on_scicore else fig.write_html(os.path.join(extract_dir_path, 'pvalloc_topo_GBAUJ_comparison.html'))
del fig


# analize GWR ALL preprep data files --------------------------------------------------
gwr_all_bfs_agg_df = pl.DataFrame()
for preprep_scen in preprep_scen_list:
    if data_extracted_to_comp_dir:
        gwr_all_building_df = pl.read_parquet(os.path.join(extract_dir_path, f'gwr_all_building_df_{preprep_scen}.parquet') )
    else:
        gwr_all_building_df = pl.read_parquet(os.path.join('data', 'preprep', preprep_scen, 'gwr_all_building_df.parquet'))

    gwr_all_building_df = gwr_all_building_df.with_columns(
        pl.lit(preprep_scen).alias('preprep_scen'), 
        pl.when(pl.col('GBAUJ') == '')
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias('no_GBAUJ'),
        pl.when(pl.col('GBAUJ') != '')
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias('has_GBAUJ')
        )
    

    gwr_bfs_all = gwr_all_building_df.group_by( ['preprep_scen', 'GGDENR']).agg([
        pl.col('EGID').count().alias('n_EGID'), 
        pl.col('has_GBAUJ').sum().alias('has_GBAUJ'), 
        pl.col('no_GBAUJ').sum().alias('no_GBAUJ'), 
    ])
    gwr_all_bfs_agg_df = pl.concat([gwr_all_bfs_agg_df, gwr_bfs_all])

melted_preprep_all_df = gwr_all_bfs_agg_df.unpivot(
    on=['has_GBAUJ', 'no_GBAUJ'],
    index=['preprep_scen', 'GGDENR'],
    variable_name='gbauj_status',
    value_name='count',
)
fig = px.bar(
    melted_preprep_all_df,
    x='GGDENR',
    y='count',
    color='gbauj_status',
    barmode='stack',
    facet_col='preprep_scen',
    title='GWR all (preprep): has_GBAUJ vs no_GBAUJ per grid_node by scenario',
)
fig = update_facet_titles(fig)
fig.show() if not run_on_scicore else fig.write_html(os.path.join(extract_dir_path, 'GWR_all_preprep_GBAUJ_comparison.html'))
del fig



# analize GWR SELECT preprep data files --------------------------------------------------
gwr_select_bfs_agg_df = pl.DataFrame()
for preprep_scen in preprep_scen_list:
    if data_extracted_to_comp_dir:
        gwr = pl.read_parquet(os.path.join(extract_dir_path, f'gwr_{preprep_scen}.parquet') )
    else:
        gwr = pl.read_parquet(os.path.join('data', 'preprep', preprep_scen, 'gwr.parquet'))

    gwr = gwr.with_columns(
        pl.lit(preprep_scen).alias('preprep_scen'), 
        pl.when(pl.col('GBAUJ') == 0)
        # pl.when(pl.col('GBAUJ').is_null())
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias('no_GBAUJ'),
        pl.when(pl.col('GBAUJ') != 0)
        # pl.when(pl.col('GBAUJ').is_not_null()) # != '')
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias('has_GBAUJ')
        )
    # gwr = gwr.with_columns(
    #     pl.lit(preprep_scen).alias('preprep_scen'),
    #     # pl.col('GBAUJ').replace('', None).alias('GBAUJ'),
    # ).with_columns(
    #     pl.col('GBAUJ').is_null().cast(pl.Int8).alias('no_GBAUJ'),
    #     pl.col('GBAUJ').is_not_null().cast(pl.Int8).alias('has_GBAUJ'),
    # )


    gwr_bfs = gwr.group_by( ['preprep_scen', 'GGDENR']).agg([
        pl.col('EGID').count().alias('n_EGID'), 
        pl.col('has_GBAUJ').sum().alias('has_GBAUJ'), 
        pl.col('no_GBAUJ').sum().alias('no_GBAUJ'), 
    ])
    gwr_select_bfs_agg_df = pl.concat([gwr_select_bfs_agg_df, gwr_bfs])
    

melted_preprep_select_df = gwr_select_bfs_agg_df.unpivot(
    on=['has_GBAUJ', 'no_GBAUJ'],
    index=['preprep_scen', 'GGDENR'],
    variable_name='gbauj_status',
    value_name='count',
)
fig = px.bar(
    melted_preprep_select_df,
    x='GGDENR',
    y='count',
    color='gbauj_status',
    barmode='stack',
    facet_col='preprep_scen',
    title='GWR SELECT (preprep): has_GBAUJ vs no_GBAUJ per grid_node by scenario',
)
fig = update_facet_titles(fig,)
fig.show() if not run_on_scicore else fig.write_html(os.path.join(extract_dir_path, 'GWR_select_preprep_GBAUJ_comparison.html'))    

print('end')
    













# ------------------------------------------------------------------------------------------------------
# parquet to csv
if False:
    path_list = [
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

        # r"C:\Users\hocrau00\Downloads\DEV2_pvalloc_16nbfs_RUR_gridoptim_max\zMC1_OptimExpa\npv_df.parquet",
        # r"C:\Users\hocrau00\Downloads\DEV2_pvalloc_16nbfs_RUR_gridoptim_max\zMC1_OptimExpa\pred_npv_inst_by_M\pred_inst_df_2.parquet",
        # r"C:\Users\hocrau00\Downloads\DEV2_pvalloc_16nbfs_RUR_gridoptim_max\zMC1_OptimExpa\pred_npv_inst_by_M\pred_inst_df_1.parquet",
        # r"C:\Users\hocrau00\Downloads\DEV2_pvalloc_16nbfs_RUR_max\zMC1\pred_npv_inst_by_M\pred_inst_df_2.parquet",
        # r"C:\Users\hocrau00\Downloads\DEV2_pvalloc_16nbfs_RUR_max\zMC1\pred_npv_inst_by_M\pred_inst_df_1.parquet" ,       
        
        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_2nbf_10y_compare3_max\zMC1\npv_df.parquet",
        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_2nbf_10y_compare3_max\zMC1\pred_inst_df.parquet",
        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_2nbf_10y_compare3_max__preprep_before_Feb26\zMC1\npv_df.parquet",
        # r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_2nbf_10y_compare3_max__preprep_before_Feb26\zMC1\pred_inst_df.parquet",
        ]
    
    for pq_path in path_list:
        scen_name = pq_path.split('\\')[-3]
        file_name = pq_path.split('\\')[-1].split('.parquet')[0]
        # csv_path = "\\".join(pq_path.split('\\')[0:-1])
        csv_path = 'C:\Models\OptimalPV_RH\data\pvalloc'

        df  = pd.read_parquet(pq_path)
        if any( [tag in pq_path for tag in ['OLDpreprep', 'COPYpreprep_used_untilFeb26',] ] ):
            export_path = f'{csv_path}\{scen_name}_{file_name}_OLDpreprep.csv'
        else:
            export_path = f'{csv_path}\{scen_name}_{file_name}.csv'

        # df.to_csv(export_path)
        df.to_excel(export_path.replace('.csv', '.xlsx'), index=False)


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




