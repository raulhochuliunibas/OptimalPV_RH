import pandas as pd
import numpy as np
import os as os
import geopandas as gpd
import xarray as xr
import datashader as ds 
import contextily as ctx
import matplotlib as mpl
import fiona
import pyogrio

from shapely import geometry

wd_path = "C:/Models/OptimalPV"
data_path = "C:/Models/data"
os.chdir(wd_path)   
os.listdir()
os.listdir(f'{data_path}')

ch_kt = 0
gpd.read_file(f'{data_path}/data/swissboundaries3d_2023-01_2056_5728.shp')












print("****************************** \nfinished running main_file.py \n******************************")
