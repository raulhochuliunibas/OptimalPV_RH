# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024


# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------

# SETUP --------

# IMPORT API DATA --------
# TODO: B - Get ENTSOE API to work properly

# still to import: 
# OK A - TS for electricity demand
#    > import and transformation to df from dict even sensible? => No, i just retransfrom the df to a dict again. keep it simple :)
#    ok >> group agg_demand_df by upper and lower 50 percentilie for with/out HP and with/out PV. 
#    ok >> then export this demand series and 
#    ok >> attach type to EGID of GWR

# (OK) A - TS for meteo sunshine => these two TS should ideally come from the same source
#      > Not from the same source as demand TS -> ok, doesn't matter according also to Frank
#      > For now just imported the shortwave radiation data (visible sunlight, amongst other columns) for Basel SBB (?)
#      > only. Will add more weather stations later

# TODO: B- TS electricity market price (maybe not relevant)

# TODO: Z - heat demand data (not relevant if I have demand TS)


# IMPORT LOCAL DATA + SPATIAL MAPPINGS --------
# OK B - remove unneeded mapping functions (e.g. create_spatial_mappings, solkat_spatial_toparquet, etc.).

# EXTEND WITH TIME FIXED DATA --------
# OK A - remove unnecessary CUMSUM function for PV installation cost, find a way to export and store cost function params


# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------


# INITIALIZATION --------

# - func to import and transform all prepreped data
#   > return all relevant dfs
# - func to create pv topo dict
#   > return dict
# BOOKMARK - The Map_solkategid_pv appears to be empty, so I have to recheck data aggregation for that part of the code

# - func to create the feedin tarif TS (empty)
#   > return TS
# - func to attach all roof partition combos to dict
#   > return dict

# - define Construction capacity per month

# ALLOCATION
# - loop for month

#   - func to update pv topo dict
#    . 
#   > return dict   

#   - func to update feedin premium
#   > return premium TS
