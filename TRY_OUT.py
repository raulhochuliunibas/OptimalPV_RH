import os as os
import pandas as pd
import geopandas as gpd
import glob
import shutil
import winsound
import functions
import datetime

from auxiliary_functions import chapter_to_logfile, checkpoint_to_logfile
from data_aggregation import *
from datetime import datetime

# SETTIGNS --------------------------------------------------------------------
script_run_on_server = 0
# recreate_parquet_files = 1



