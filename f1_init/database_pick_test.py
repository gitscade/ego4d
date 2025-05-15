import sys
import os
import database_init as database_init
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
from util import util_constants



# -----------------------
# Path & API & Model
# -----------------------
data_path = util_constants.PATH_DATA
GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'

SPATIAL_ANNOTATION_PATH1 = data_path + 'spatial/manual'
SPATIAL_ANNOTATION_PATH2 = data_path + 'spatial/semiauto' 

video_list1 = database_init.make_spatial_json_video_list_singlepath(SPATIAL_ANNOTATION_PATH1)
video_list2 = database_init.make_spatial_json_video_list_singlepath(SPATIAL_ANNOTATION_PATH2)
uid_list1 = database_init.make_spatial_json_video_uid_list(video_list1)
uid_list2 = database_init.make_spatial_json_video_uid_list(video_list2)
print(f"len {len(uid_list1)}\n uid_list: {uid_list1}")
print(f"len {len(uid_list2)}\n uid_list: {uid_list2}")



import os
import shutil

path1 = "/path/to/source"
path2 = "/path/to/destination"
filenames = ["file1.txt", "file2.jpg", "data.csv"]  # Your list of files

# Make sure destination path exists
os.makedirs(path2, exist_ok=True)

# Copy matching files
for fname in filenames:
    src_file = os.path.join(path1, fname)
    dst_file = os.path.join(path2, fname)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {fname}")
    else:
        print(f"File not found in source: {fname}")