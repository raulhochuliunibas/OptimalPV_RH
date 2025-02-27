# import scenario_default_settings as scen_def_sett
from scenario_default_settings import DataAggDefaultSettings, DataAgg_MainSpecs, DataAgg_GWRSelectionSpecs, DataAgg_SolkatSelectionSpecs, DataAgg_DemandSpecs


class DataAggExecutionScenarios:
    def __init__(self, run_on_server):
        self.run_on_server = run_on_server
        self.all_scenarios = {
            'preprep_BLBSSO_18to23_1and2homes_API_reimport': DataAggDefaultSettings(
                main_specs=DataAgg_MainSpecs(
                    name_dir_export='preprep_BLBSSO_18to23_1and2homes_API_reimport',
                    script_run_on_server=run_on_server,
                    kt_numbers=[13, 12, 11],
                    year_range=[2018, 2023],
                    split_data_geometry_AND_slow_api=True,
                ),
                gwr_selection_specs=DataAgg_GWRSelectionSpecs(
                    GKLAS=['1110', '1121', '1276']
                ),
                solkat_selection_specs=DataAgg_SolkatSelectionSpecs(
                    cols_adjust_for_missEGIDs_to_solkat=['FLAECHE', 'STROMERTRAG'],
                    match_missing_EGIDs_to_solkat_TF=True,
                    extend_dfuid_for_missing_EGIDs_to_be_unique=False,
                ),
                demand_specs=DataAgg_DemandSpecs(
                    input_data_source="NETFLEX"
                )
            ),
            'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates': DataAggDefaultSettings(
                main_specs=DataAgg_MainSpecs(
                    name_dir_export='preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
                    script_run_on_server=run_on_server,
                    kt_numbers=[13],
                    year_range=[2022, 2023],
                    split_data_geometry_AND_slow_api=False,
                ),
                gwr_selection_specs=DataAgg_GWRSelectionSpecs(
                    GKLAS=['1110', '1121']
                ),
                solkat_selection_specs=DataAgg_SolkatSelectionSpecs(
                    cols_adjust_for_missEGIDs_to_solkat=['FLAECHE', 'STROMERTRAG'],
                    match_missing_EGIDs_to_solkat_TF=True,
                    extend_dfuid_for_missing_EGIDs_to_be_unique=False,
                ),
                demand_specs=DataAgg_DemandSpecs(
                    input_data_source="NETFLEX"
                )
            ),
            'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates': DataAggDefaultSettings(
                main_specs=DataAgg_MainSpecs(
                    name_dir_export='preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
                    script_run_on_server=run_on_server,
                    kt_numbers=[13, 11],
                    year_range=[2022, 2023],
                    split_data_geometry_AND_slow_api=False,
                ),
                gwr_selection_specs=DataAgg_GWRSelectionSpecs(
                    GKLAS=['1110', '1121']
                ),
                solkat_selection_specs=DataAgg_SolkatSelectionSpecs(
                    cols_adjust_for_missEGIDs_to_solkat=['FLAECHE', 'STROMERTRAG'],
                    match_missing_EGIDs_to_solkat_TF=True,
                    extend_dfuid_for_missing_EGIDs_to_be_unique=False,
                ),
                demand_specs=DataAgg_DemandSpecs(
                    input_data_source="NETFLEX"
                )
            ),
            'preprep_BLBSSO_22to23_extSolkatEGID_DFUIDduplicates': DataAggDefaultSettings(
                main_specs=DataAgg_MainSpecs(
                    name_dir_export='preprep_BLBSSO_22to23_extSolkatEGID_DFUIDduplicates',
                    script_run_on_server=run_on_server,
                    kt_numbers=[13, 12, 11],
                    year_range=[2022, 2023],
                    split_data_geometry_AND_slow_api=False,
                ),
                gwr_selection_specs=DataAgg_GWRSelectionSpecs(
                    GKLAS=['1110', '1121']
                ),
                solkat_selection_specs=DataAgg_SolkatSelectionSpecs(
                    cols_adjust_for_missEGIDs_to_solkat=['FLAECHE', 'STROMERTRAG'],
                    match_missing_EGIDs_to_solkat_TF=True,
                    extend_dfuid_for_missing_EGIDs_to_be_unique=False,
                ),
                demand_specs=DataAgg_DemandSpecs(
                    input_data_source="NETFLEX"
                )
            ),
        }

    def get_scenarios(self, scen_group_names):
        scen_group_dir = {}
        for scen_name in scen_group_names:
            if scen_name in self.all_scenarios:
                scen_group_dir[scen_name] = self.all_scenarios[scen_name]
            else:
                print(f'Scenario <{scen_name}> not found in data aggregation scenarios')
        return scen_group_dir