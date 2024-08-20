# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024


# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------
if True: 
    # SETUP --------
    

    # IMPORT API DATA --------
    # OK B - Get ENTSOE API to work properly

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

    # TODO: Z - remove heat demand related code (not relevant if I have demand TS)


    # IMPORT LOCAL DATA + SPATIAL MAPPINGS --------
    # OK B - remove unneeded mapping functions (e.g. create_spatial_mappings, solkat_spatial_toparquet, etc.).
    # TODO: IMPORTANT: EGID in solkat is a faulty data point - at least for the small developing data set (nrows = ca. 1000 - 5000). 
    #       That's why a second attempt was started to aggregate roof partitions on the sbuuid and map those then to the EGID numbers. 
    #       Problem there - the amount of missmatches was strikingly large (+- 70 vs 30%) that SB_UUID aggregation does not seem to 
    #       be warrented. Also  in developing Map_solkatdfuid_egid, makes small overcounts (1 partition is attributed to two houses). 
    # --------> Check the BSBL Case thoroughly for the EGID mapping if not multiple EGIDS are attributed to one "house" (building) which then 
    #           would mean double counting


    # EXTEND WITH TIME FIXED DATA --------
    # OK A - remove unnecessary CUMSUM function for PV installation cost, find a way to export and store cost function params
    print('')

# TODO: Check EGIDs in solkat for BSBL for not to have double counting.

# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------


# INITIALIZATION --------

# OK - func to import and transform all prepreped data
#   > return all relevant dfs
# OK - func to create pv topo dict
#   > return dict
# OK - func to attach all roof partition combos to dict
#   > return dict
# OK - filter all input data again by alloc settings such that the allocation algorithm is applicable also on larger preped data sets

# B - func to create the feedin tarif TS (empty)
#   > return TS


# A - define Construction capacity per month

# ALLOCATION
# - loop for month

#   - func to update pv topo dict
#    . 
#   > return dict   

#   - func to update feedin premium
#   > return premium TS

# TO-DOs:
# TODO: Adjust GBAUJ to other variable
# TODO: Include WNART to only consider residential buildings for "primary living"
# TODO: check in QGIS if GKLAS == 1273 are really also denkmalgeschützt buildings or just monuments


# -----------------------------------------------------------------------------
# data / model extensions 
# -----------------------------------------------------------------------------
# MA Lupien, Thormeyer et al 2020; also use protected / heritage zones to rule out PVs in that area
# MA Lupien, EV stations per Gemeinde => possible to map stations to DSO nodes?
# MA Lupien, "Exploitable solar resource" per gmeinde?
