import os as os
from datetime import datetime


def crs2wsg84(gdf_fnc):
    """
    Function to convert a geodataframe to wsg84
    """
    wgs84_crs = gdf_fnc.crs.to_string().split(" +up")[0]
    gdf_fnc = gdf_fnc.to_crs(wgs84_crs)
    return gdf_fnc

# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------


def checkpoint_to_logfile(str):
    """
    Function to write a checkpoint to the logfile
    """
    check = f'* {str}: {datetime.now()}'
    print(check)
    with open(f'log_file.txt', 'a') as log_file:
        log_file.write(f"{check}\n")

def chapter_to_logfile(str):
    """
    Function to write a chapter to the logfile
    """
    check = f'\n\n****************************************\n {str} \n start at:{datetime.now()} \n****************************************\n\n'
    print(check)
    with open(f'log_file.txt', 'a') as log_file:
        log_file.write(f'{check}\n')

