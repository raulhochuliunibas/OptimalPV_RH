from calibration_class import Calibration_Settings, Calibration

if __name__ == '__main__':
    preprep_list = [
        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs',
            name_calib_subscen='reg2_all_CH_bfs',
            kt_numbers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,], 
            n_rows_import= None,
            rerun_import_and_preprp_data_TF = True,
            export_gwr_ALL_building_gdf_TF = False
        ), 
    ]
            
    for sett in preprep_list:
        calib_class = Calibration(sett)
        calib_class.concatenate_prerep_data()

        calib_class.estimdf2_regression_instsize()
