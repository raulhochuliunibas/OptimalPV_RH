import sys
from calibration_class import Calibration_Settings, Calibration

if __name__ == '__main__':
    preprep_list = [
        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs',
            # name_preprep_subsen=f'kt{kt_number}',
            # kt_numbers=[kt_number,], 
            # name_preprep_subsen=f'bfs{bfs_number}',
            # bfs_numbers=[bfs_number,], 
            n_rows_import= None,
            rerun_import_and_preprp_data_TF = True,
            export_gwr_ALL_building_gdf_TF = False
        ), 
    ]
            
    for sett in preprep_list:
        calib_class = Calibration(sett)
        calib_class.concatenate_prerep_data()
