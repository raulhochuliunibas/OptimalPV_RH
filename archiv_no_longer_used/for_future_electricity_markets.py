import os as os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px

from datetime import datetime
from shapely.ops import unary_union

# still uncertain if this is needed
import warnings



# Setup + Import --------------------------------------------------------------------------------------------------

script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server


# pre setup + working directory ----------------------------------------------------------------------------------

if script_run_on_server == 0:
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)


# import data -----------------------------------------------------------------------------------------------------

pv = gpd.read_file(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
pv.head(10)
pv.dtypes

# figure ----------------------------------------------------------------------------------------------------------

date_range = pd.date_range(start = '2021-01-01', end = '2023-12-31', freq = 'D')
date_bool = pv['BeginningO'].isin(date_range)
power_bool = pv['TotalPower'] < 2000

power_bool.value_counts()
fig_df = pv.loc[date_bool & power_bool, ['TotalPower']].copy()

fig = px.histogram(fig_df, x = 'TotalPower', nbins = 1000, 
                   marginal="box", 
                   hover_data = fig_df.columns)
fig.show()

pv.loc[pv['TotalPower'] < 2000, 'TotalPower'].describe()
pv.loc[pv['TotalPower'] < 1500, 'TotalPower'].describe()