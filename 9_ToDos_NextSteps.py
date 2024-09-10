<<<<<<< HEAD
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024


# ==============================================================================
#  BURNER LIST
# ==============================================================================


# - geschenk fly bestellen

# read pv allocation from valverde



# ==============================================================================


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
    # TODO: B - IMPORTANT: EGID in solkat is a faulty data point - at least for the small developing data set (nrows = ca. 1000 - 5000). 
    #           That's why a second attempt was started to aggregate roof partitions on the sbuuid and map those then to the EGID numbers. 
    #           Problem there - the amount of missmatches was strikingly large (+- 70 vs 30%) that SB_UUID aggregation does not seem to 
    #           be warrented. Also  in developing Map_solkatdfuid_egid, makes small overcounts (1 partition is attributed to two houses). 
    # ------------> Check the BSBL Case thoroughly for the EGID mapping if not multiple EGIDS are attributed to one "house" (building) which then 
    #               would mean double counting
    # TODO: remove GWR that are not in solkat?
    # TODO: Check EGIDs in solkat for BSBL for not to have double counting.



    # EXTEND WITH TIME FIXED DATA --------
    # OK A - remove unnecessary CUMSUM function for PV installation cost, find a way to export and store cost function params
    print('')

# TODO: A- Start a summary statistic logfile that just logs how many buildings etc are acutally in sample of the whole data frame!
# TODO: rerun prepre with 2018 to 2022 => GWR should only include building landscape of 2022 (latest puplication of solkat, Jan 2023)
# TODO: B - find the Swiss building 3D data set and rebuild the kw / m2 production potential yourself for accurate NPV calculations

# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------


# INITIALIZATION --------

# OK - func to import and transform all prepreped data
#   > return all relevant dfs
# OK - func to create pv topo dict
# TODO: A - Remove all GWR EGID that are not part of the solkat data set
#               > somehow export these EGIDs to a list and display which house types are not in solkat data set
# TODO: A - if multiple pvtarif 
#               > which gmd are affected?
#               > take average? / random allocation to 1 or the other nrElcom provider?
# TODO: A - 
#   > return dict

# BRANCH 1: PV ALLOCATION --------
if True: 
    # OK - func to attach all roof partition combos to dict
    #   > return dict
    # OK - filter all input data again by alloc settings such that the allocation algorithm is applicable also on larger preped data sets

    # OK B - func to create the feedin tarif TS (empty)
    #   > return TS

    # OK A - define Construction capacity per month

    # ALLOCATION
    # - loop for month
    # OK D - find a less error prone way to find interim runs for the topology and Mapping files in the pvalloc folders (renamed versions)
    #   
    # TODO: A - func to update pv topo dict
    #   > return dict   

    # TODO: B - func to update feedin premium
    #   > return premium TS
    print('')

# BRANCH 2: PV ALLOCATION --------
if True: 
    # OK - func to get all roof partitions to df WITHOUT combos
    # TODO: A - compute production (and self consumption) before monthly prediction iteration
    #       > merge hour of radiation to partition df. 
    #       > multiply with roof FLAECHE for production

    # TODO: A - compute monetary values for production and self consumption
    #       > then groupby EGID and roof partition to get economic gain and spending per partition, per house
    #       > build all possible combinations of partitions, attach economic gain / spend to them and estimate investment cost
    #       > then calculate NPV for all possible combinations of partitions
    print('')



# TO-DOs:
# TODO: A - Adjust code to change prediction step size (e.g. quarter, not just every month!)

# TODO: Adjust GBAUJ to other variable
# TODO: Include WNART to only consider residential buildings for "primary living"
# TODO: check in QGIS if GKLAS == 1273 are really also denkmalgeschützt buildings or just monuments


# -----------------------------------------------------------------------------
# data / model extensions 
# -----------------------------------------------------------------------------
# MA Lupien, Thormeyer et al 2020; also use protected / heritage zones to rule out PVs in that area
# MA Lupien, EV stations per Gemeinde => possible to map stations to DSO nodes?
# MA Lupien, "Exploitable solar resource" per gmeinde?
=======
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
# TODO: some egid in gwr.parquet are double up to 10 times in the column, check why!

# -----------------------------------------------------------------------------
# pv_allocation__MASTER.py 
# -----------------------------------------------------------------------------


# INITIALIZATION --------

# OK - func to import and transform all prepreped data
#   > return all relevant dfs
# OK - func to create pv topo dict
#   OK A - add Ausrichtung and Neigung to the topo for later uses
#   OK A - find a functional way to adjust production to Ausrichtung (Neigung)
#   OK A - add a fictive node to kantons 11, 12, 13 egids (just a plain csv, then merge it with empty gridprem_ts) 
#   > return dict
# OK - func to attach all roof partition combos to dict
#   > return dict
# OK - filter all input data again by alloc settings such that the allocation algorithm is applicable also on larger preped data sets


#===============================
# # BOOKMARK: 
# - There is an issue with buildings that have a HUGE number of partitions. 
#  > make either a function exporting all solkat with more than x rows for 1 egid
#  > OR make a function after topo_df generation that matches the topo df to the geo data and exports it to a shape file
#  > PLUS attach also exisiting pv installations to the topoology! so it is possible to see in topo_df which buildings have a pv installation already
#===============================

# OK:  A - define Construction capacity per month

# ALLOCATION

# LOOP for MONTE CARLO
#   - #TODO:  - create copies of all the data objects that are needed "fresh" for each iteration. oterhwise iter2
#               will be influenced by iter1 (will start where iter1 ended)
#   - #TODO:OK A - create a directory to store all the monte carlo iterations

#        - LOOP for MONTH
#          - update npv for all non pv buildings
#            OK A - get all topo_time_subdf from many seperated sub_topo_dfs.parquet by EGID
#            > return npv_df
#       
#          - func to update feedin premium
#            OK A - get all installations from topo.dict
#            OK A - calculate production HOYs and find where voltage level meets tier
#            OK A - update gridprem_ts
#            > return premium TS
#       
#          - pick installation
#            OK A - create selection mechanism NPV based as function
#            OK A - create random selection mechanism as function (for comparision)
#       
#          - add picked installation to topo.dict 
#       
#          - export built installations for later visualizations

#   - #TODO:OK A - copy-paste the MC iteration to MC dir
#   - #TODO:OK B - select only the relevant output files to store in MC dir


# -----------------------------------------------------------------------------
# visualizations_MASTER.py 
# -----------------------------------------------------------------------------

# Create plot functions for the following:

# - individual scenarios
#   TODO: A - Amount of installed capacity per node (avg incl std bands)
#   TODO: A - Amount of installed capacity per bfs (avg incl std bands)
#   TODO: ? - Map of covered regions (bfs) with installed capacity
#   TODO: ? - Avg building characteristics (amount of houses in each class etc., with x roof partitions etc.)


# - aggregated scenarios
#   TODO: A - Avg Amount of installed capacity in months (avg incl std bands)
#   TODO: A - Avg Production HOY per scenario (avg incl std bands)
#   TODO: A - Final avg gridprem ts per scenario (avg incl std bands)

# 









#   - pick building for PV



# TO-DOs:
# TODO: Adjust GBAUJ to other variable
# TODO: Include WNART to only consider residential buildings for "primary living"
# TODO: check in QGIS if GKLAS == 1273 are really also denkmalgeschützt buildings or just monuments
# TODO: inst_power in allocation algorithm NOT weighted by angle tilt efficiency; But in updating of grid premium
        # which one makes more sense?
# TODO: gridprem update disregards selfconsumption rate; calculates 100% feedin, which would be possible. 



#   - repick building for PV
# -----------------------------------------------------------------------------
# data / model extensions 
# -----------------------------------------------------------------------------
# MA Lupien, Thormeyer et al 2020; also use protected / heritage zones to rule out PVs in that area
# MA Lupien, EV stations per Gemeinde => possible to map stations to DSO nodes?
# MA Lupien, "Exploitable solar resource" per gmeinde?
>>>>>>> 08030786609740b4e3b4e89216f4c7b4fb98015b
