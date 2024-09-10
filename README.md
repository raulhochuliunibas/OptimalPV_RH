# OptimalPV 
Preamble:
> Author: raul.hochuli@unibas.ch
> Version/Date: July 2024
> Abstract: ...

# Code Structure
The code in a partitioned way. Each step of the aggregation / modelling process has a *MASTER* file, which runs a number of functions, all stored in subfiles that are called sequentially. 

## data_aggregation_MASTER.py
objective: 

### api_electricity_prices.py
### api_pvtarif.py
### installation_cost.py
### preprepare_data.py
### sql_gwr.py

## pv_allocation_MASTER.py


# Further Steps

1. Achieve cumulative building cost per partition. 
    - create an extrapolation function for installations in installtion_cost.py which is applicable to houses by EGID
    - then use groupby and .cumsum to cumulative sum over all houses (ATTENTION: ensure that cumsum sums up the areas in a decreasing order for suitability => best suitable partitions first, then add later less suitable partitions. )
    - attach total and pkW cost to the data frame
    
## Functionality
>stil before pv_allocation
 - create a py file that creates all assumptions, cost etc. 

> prepare all data computations

> cost computation
- compute cost per roof partition - CHECK
    - "downwoard" computation -> compute NPV for best partition, second best and best partition, etc.
    - include ratio of self consumption

- (compute elec demand by heating squrare)
 

> subset selection


> initiate topology
- create dict with gwr id and year
    - subsetable by bfs number and building type

- define thresholds for installations per year

- assign all partitions to dict
- assign production of installed pv to dict
- (assign grid connection to dict)

> calculate NPV
- select free gwrs 
- calculate NPV by partition (possible with switch to only consider KLASSE 3+)
- select best NPV

## Data Extension
- Include facades
- Remove MSTRAHLUNG from Cummulative summation => unnecessary and not true anyway (summed up an average)


## Nice to have for more utility later
- make data import to prepreped data only for the selected number of municipalites, no just all of Switzerland
- a single dict, py file for all model assumptions (numeric and boolean)

- create a GWR>GM and SOLKAT>GM mapping for the spatial data
-  Add many more variables for GWR extraction (heating, living area etc.), WAREA not found in SQL data base
- change code such that prepred_data is on the same directory level than output
-  ADJUST ALL MAPPINGS so that the data type is a string, not an int
- Map_egroof_sbroof carries an unnecessary index in the export file. remove that in the preppred_data function



## 