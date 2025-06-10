
import sys
import json
import os
import re 
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import f1_init.agent_init as agent_init
import f1_init.constants_init as constants_init

source_folder = constants_init.PATH_AUGMENTATION_v8_source
target_folder = constants_init.PATH_AUGMENTATION_v8_1200
source_spatial_json_list, target_spatial_json_list = agent_init.get_paired_spatial_json_list_v8(source_folder, target_folder)

print(len(source_spatial_json_list))
print(len(target_spatial_json_list))

# def sort_3part_filenames(filenames):
#     """
#     Sorts a list of filenames based on 'id', 'secondorder', and 'lastorder' numerically.
#     """
#     def parse_filename(filename):
#         # Remove the .json extension
#         base_name = filename.rsplit('.', 1)[0]
#         parts = base_name.split('_')
        
#         # Ensure there are always three parts
#         if len(parts) != 3:
#             # Handle cases where the format might be slightly off, or just skip/log them
#             return filename, -1, -1 # Return original filename and dummy sort keys

#         id_part = parts[0]
#         second_order_part = parts[1]
#         last_order_part = parts[2]

#         # Convert second_order to float for numerical sorting
#         try:
#             second_order_num = float(second_order_part)
#         except ValueError:
#             second_order_num = -1  # Assign a low value for non-numeric second_order

#         # Extract number from last_order_part (e.g., '0th' -> 0, '1th' -> 1)
#         last_order_match = re.match(r'(\d+)(th|st|nd|rd)', last_order_part)
#         if last_order_match:
#             last_order_num = int(last_order_match.group(1))
#         else:
#             last_order_num = -1  # Assign a low value for non-numeric last_order

#         return id_part, second_order_num, last_order_num

#     # Sort the filenames using the custom key
#     # First by id (lexicographically), then by second_order (numerically), then by last_order (numerically)
#     return sorted(filenames, key=parse_filename)

# # Example Usage:
# folderpath = 'your_folder_path_here' # Replace with your actual folder path

# # Simulate filenames if you don't have a physical folder
# # In a real scenario, you would use:
# # filenames = os.listdir(folderpath)
# filenames = [
#     "0ae6293e-eda5-44f7-b56e-e8f27fcde953_0.2_0th.json",
#     "0ae6293e-eda5-44f7-b56e-e8f27fcde953_0.2_1th.json",
#     "0ae6293e-eda5-44f7-b56e-e8f27fcde953_0.2_3th.json",
#     "0ae6293e-eda5-44f7-b56e-e8f27fcde953_0.4_2th.json",
#     "01ce4fd6-197a-4792-8778-775b03780369_0.6_8th.json",
#     "01ce4fd6-197a-4792-8778-775b03780369_0_7th.json",
#     "a1b2c3d4-e5f6-7890-1234-567890abcdef_1.0_10th.json", # New example
#     "a1b2c3d4-e5f6-7890-1234-567890abcdef_0.5_5th.json",  # New example
#     "0ae6293e-eda5-44f7-b56e-e8f27fcde953_1.0_0th.json", # Another example for secondorder sorting
# ]

# sorted_filenames = sort_3part_filenames(filenames)

# for filename in sorted_filenames:
#     print(filename)
