import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import itertools
import glob
import itertools
import shutil
import fnmatch
import polars as pl
import copy
import random

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from plotly.colors import qualitative



scen_dir_path = r"C:\Models\OptimalPV_RH\data\pvalloc\DEV_pvalloc_46nbfs_30y_max"

data_dir_path = r"C:\Models\OptimalPV_RH\data"
preprep_dir_path = os.path.join(data_dir_path, 'preprep', 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI')
mc_subdir_path = os.path.join(scen_dir_path, 'zMC1')


gridnode_df_paths = glob.glob(f'{mc_subdir_path}/pred_gridprem_node_by_M/gridnode_df_*.parquet')
trange_prediction = pd.read_parquet(f'{mc_subdir_path}/trange_prediction.parquet')
gridnode_df_main    = pl.read_parquet(f'{mc_subdir_path}/gridnode_df.parquet')
constrcapa = pd.read_parquet(f'{mc_subdir_path}/constrcapa.parquet')


fig_agg = go.Figure()

