# OptimalPV 

## Preamble:
**Author**: raul.hochuli@unibas.ch 
**Latest update**: of README: Feb 2025 
**Abstract**: Due to the global push for distributed photovoltaic (PV) systems, grid capacity will be a limiting factor for further expansion in the near future. Additional installations can cause a negative externality by hindering other production in the same grid circuit. 
This calls for a detailed modeling approach that captures the negative impact of a PV installation on neighbor's production capability. 
Using spatial data at the individual house level for ca. 40 thousand single-family houses, I propose a stochastic Monte Carlo method to model theoretical PV expansion pathways until 2050 from a social planer's perspective considering optimal grid ussage and a economic rational individual's perspective. 
I use a local grid operators data on low distribution house-grid connections and transformer capacities. This is the cornerstone for further research on a dynamic feed-in permit that is designed to mitigate grid congestion without requiring intense regulator monitoring.  
Find the entire research proposal in ([Proposal_OptimalPV_RH](docs/Proposal_OptimalPV_RH.pdf)) here for more elaboration on the sceintific contribution. Note that since the proposal acceptance in December 2023, notable changes have been made to the propose model structure and data usage. 

The following description file gives a short description over the entire code used in my personal PhD research proposal. 


## Code Structure and Process
For a better understanding, review the visualization of the model structure in the [readme_OptimalPV_graph.png](docs/readme_OptimalPV_graph.png). The two files:
- [execution_scenarios.py](execution_scenarios.py) (defining all scenario settings (just deviations from default settings) for data aggregation and pv allocation model runs)
- [execution_RUNFILE.py](execution_RUNFILE.py) (the main file, running (selected parts of the model) with selected scenarios and visualization settings)

- **MASTER files:**The MASTER functions use docstrings to explain in a few steps what the called subfiles and functions are doing. Because they represent a larger process with many steps, MASTER functions do not return any objects but store various datafiles in an interm folder directory, which is then again accessed by the next MASTER function. This allows for development of only small specific parts in the entire model and later parallelization of the Monte Carlo iterations. 
- **Setting dictionaries:** Each MASTER file has a corresponding settings directory containing relevant information for the specific part of the model. These setting directories have default values, stored in the *default_settings.py* (e.g. [default_settings.py](pv_allocation/default_settings.py)) file of every code subfolder. For increased readabiligy, the run scenarios for the data aggregation and pv allocation step are defined in *execution_scenarios.py*. There only the changes to the default values are written down to increase visibility. The remaining default values for each scenario are later added with a function in [execution_RUNFILE.py](execution_RUNFILE.py).



For a detailed account of what the individual subfiles and functions are doing, read the [OptimalPV_RH_ModelDescription.pdf](docs/OptimalPV_RH_ModelDescription.pdf) which tries to stay up to date with all functions and their purpose. This model description will be transferred step by step to an individual README file in each subfolder. 


