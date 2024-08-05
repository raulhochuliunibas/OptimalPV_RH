# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024


# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------

# SETUP --------

# IMPORT API DATA --------
# B - Get ENTSOE API to work properly

# still to import: 
# A - TS for electricity demand
# A - TS for meteo sunshine => these two TS should ideally come from the same source
# B- TS electricity market price (maybe not relevant)

# Z - heat demand data (not relevant if I have demand TS)


# IMPORT LOCAL DATA + SPATIAL MAPPINGS --------
# B - remove unneeded mapping functions (e.g. create_spatial_mappings, solkat_spatial_toparquet, etc.).

# EXTEND WITH TIME FIXED DATA --------
# A - remove unnecessary CUMSUM function for PV installation cost, find a way to export and store cost function params


