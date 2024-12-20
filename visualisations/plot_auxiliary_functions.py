import os as os
import sys

import os as os
import pandas as pd
import geopandas as gpd
import numpy as np
import json 
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import copy
import glob
import matplotlib.pyplot as plt
import winsound
import itertools
import shutil
import scipy.stats as stats

from datetime import datetime
from pprint import pformat
from shapely.geometry import Polygon, MultiPolygon
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


# ------------------------------------------------------------------------------------------------------
# PLOT-AUXILIARY FUNCTIONS
# ------------------------------------------------------------------------------------------------------

def add_scen_name_to_plot(fig_func, scen, pvalloc_scen):
    # add scenario name
    fig_func.add_annotation(
        text=f'Scen: {scen}, (start T0: {pvalloc_scen["T0_prediction"].split(" ")[0]}, {pvalloc_scen["months_prediction"]} prediction months)',
        xref="paper", yref="paper",
        x=0.5, y=1.05, showarrow=False,
        font=dict(size=12)
    )
    return fig_func

# universal func for plot T0 tick -----
def add_T0_tick_to_plot(fig, T0_prediction, df, df_col):
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=T0_prediction,
            y0=0,
            x1=T0_prediction,
            y1= df[df_col].max(),  # Dynamic height
            line=dict(color="black", width=1, dash="dot"),
        )
    )
    fig.add_annotation(
        x=  T0_prediction,
        y= df[df_col].max(),
        text="T0 Prediction",
        showarrow=False,
        yshift=10
    )
    return fig

# universial func to set default plot zoom -----
def set_default_fig_zoom_year(fig, zoom_window, df, datecol):
    start_zoom = pd.to_datetime(f'{zoom_window[0]}-01-01')
    max_date = df[datecol].max() + pd.DateOffset(years=1)
    if pd.to_datetime(f'{zoom_window[1]}-01-01') > max_date:
        end_zoom = max_date
    else:
        end_zoom = pd.to_datetime(f'{zoom_window[1]}-01-01')
    fig.update_layout(
        xaxis = dict(range=[start_zoom, end_zoom])
    )
    return fig 

def set_default_fig_zoom_hour(fig, zoom_window):
    start_zoom, end_zoom = zoom_window[0], zoom_window[1]
    fig.update_layout(
        xaxis_range=[start_zoom, end_zoom])
    return fig

# Function to flatten geometries to 2D (ignoring Z-dimension) -----
def flatten_geometry(geom):
    if geom.has_z:
        if geom.geom_type == 'Polygon':
            exterior = [(x, y) for x, y, z in geom.exterior.coords]
            interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
            return Polygon(exterior, interiors)
        elif geom.geom_type == 'MultiPolygon':
            return MultiPolygon([flatten_geometry(poly) for poly in geom.geoms])
    return geom