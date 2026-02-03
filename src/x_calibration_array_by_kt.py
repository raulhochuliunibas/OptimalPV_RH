import sys
from calibration_class import Calibration_Settings, Calibration

if __name__ == '__main__':
    if len(sys.argv) > 1:
        kt_number = int(sys.argv[1])
        preprep_list = [
            Calibration_Settings(
                name_dir_export='calib_all_CH',
                name_preprep_subsen=f'kt{kt_number}',
                kt_numbers=[kt_number,], 
                n_rows_import= None,
                rerun_import_and_preprp_data_TF = True,
                export_gwr_ALL_building_gdf_TF = False
            ), 
        ]

    else:
        preprep_list = [
            Calibration_Settings(), 
        ]
            
    for sett in preprep_list:
        calib_class = Calibration(sett)
        calib_class.import_and_preprep_data()
