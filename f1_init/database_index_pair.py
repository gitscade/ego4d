"""
Makes pair of indices for test spatial scenes
"""
import sys
import os
import database_init as database_init
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
from util import util_constants






if __name__ == "__main__":
    # read all test spatial files with json.load(file) as list
    data_path = util_constants.PATH_DATA
    TEST_SPATIAL_ANNOTATION_PATH_MANUAL = data_path + 'spatial_all/manual'
    TEST_SPATIAL_ANNOTATION_PATH_SEMI = data_path + 'spatial_all/semi'    
    spatial_videos_list = database_init.make_spatial_json_video_list(TEST_SPATIAL_ANNOTATION_PATH_MANUAL, TEST_SPATIAL_ANNOTATION_PATH_SEMI)
    len(spatial_video_list)
    # 


    # For each scene in list, compare similarity with other scenes


    # Plot similarity index to other scenes


    # Pick out similarity.

