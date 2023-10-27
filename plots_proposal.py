import os as os
import functions
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
import winsound
import plotly 
import plotly.express as px

from functions import chapter_to_logfile, checkpoint_to_logfile
from datetime import datetime
from shapely.ops import unary_union

import warnings


# pre run settings -----------------------------------------------------------------------------------------------
script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server


# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------


# pre setup + working directory ----------------------------------------------------------------------------------

# create log file for checkpoint comments
timer = datetime.now()
with open(f'plots_proposal_log.txt', 'w') as log_file:
        log_file.write(f' \n')
chapter_to_logfile('started running main_file.py')

# set working directory
if script_run_on_server == 0:
     winsound.Beep(840,  100)
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)

# import data -----------------------------------------------------------------------------------------------------

# all electricity production plants
elec_prod_path = f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen'
files_names = [x for x in os.listdir(elec_prod_path) if x.endswith(".shp")]
files_names = list(map(lambda i: i[:-4], files_names))
files_names
# import data frames
gdf = [gpd.read_file(f'{elec_prod_path}', layer = x) for x in files_names]  
elec_prod = pd.concat(gdf, ignore_index=True)

# rename faulty column name
cols = elec_prod.columns.tolist()
cols[0] = 'random_id'
elec_prod.columns = cols

elec_prod['BeginningO'] = pd.to_datetime(elec_prod['BeginningO'], format='%Y-%m-%d')

# count number of plants per category
elec_prod_ts_count = elec_prod.groupby([elec_prod['BeginningO'].dt.to_period("M"), 'SubCategor']).size().reset_index(name='Counts')

elec_prod_ts_count['BeginningO'] = elec_prod_ts_count['BeginningO'].dt.to_timestamp()

# plot
fig = px.line(elec_prod_ts_count, x='BeginningO', y='Counts', color='SubCategor', title='Monthly Counts of SubCategor Entries')
fig.show()

type(elec_prod_ts_count)
type(pv_ts_count)
pv_ts_count["SubCategor"].value_counts()


os.listdir(f'{wd_path}_data/input/ch.bfe.elektrizitaetsproduktionsanlagen')













