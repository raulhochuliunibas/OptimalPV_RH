import os as os
import sys

import os as os
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import copy
import glob
import matplotlib.pyplot as plt
import scipy.stats as stats

from datetime import datetime

sys.path.append('..')
from auxiliary_functions import *


# ------------------------------------------------------------------------------------------------------
# ANALYSIS: PVALLOCATION from the PAST until TODAY
# ------------------------------------------------------------------------------------------------------

def prediction_accuracy(pvalloc_scen_list, postprocess_analysis_settings, wd_path, data_path, log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]

    