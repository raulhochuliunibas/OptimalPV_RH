
import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import glob2
import datetime as datetime
import shutil
import json
from dataclasses import dataclass, field, asdict
from typing_extensions import List, Dict

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
import itertools
import plotly.graph_objects as go


import time
tstamp = time.localtime().tm_min
# tstamp = '2 40407_0705h'

file = f'{os.getcwd()}/src/test_pckgs_venv {tstamp}.txt'
if os.path.exists(file):
    os.remove(file)

with open(file, 'w') as f:
    f.write('test file\n')

