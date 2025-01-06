from enum import Enum
import copy 

# path
annotation_data_path = 'annotations/'
common_data_path = 'original_data/'
ego4d_json_path = common_data_path + "ego4d.json"
goalstep_train_path = common_data_path + "goalstep_train.json"
goalstep_test_path = common_data_path + "goalstep_test_unannotated.json"
goalstep_val_path = common_data_path + "goalstep_val.json"

class Component(str, Enum):
    number = "number"
    videos = "videos"
    duration_sec = "duration_sec"
    level = "level"
    video_uid = "video_uid"
    duration = 'duration'
    scenarios = 'scenarios'
    goal_category = "goal_category"
    goal_description = "goal_description"
    spatial_context = "spatial_context"
    segments = "segments"
    context = "context"
    split = "split"
    
context_format = {
    "player": [
        
    ],
    "change": [
        
    ]
}

spatial_format = {
    "room1": [
        
    ],
    "room2": [
        
    ]
}

segment_format_lev2 = {
    Component.level: 2,
    Component.number: 1,
    Component.context: copy.deepcopy(context_format),
    Component.segments: [] # segment_format_lev3
}

segment_format_lev3 = {
    Component.level: 3,
    Component.number: 1,
    Component.context: copy.deepcopy(context_format),
}

video_format = {
    Component.level: 1,
    Component.number: 0,
    Component.video_uid: "",
    Component.duration: 0,
    Component.scenarios: "",
    Component.split: "",
    Component.goal_category:"",
    Component.goal_description:"",
    Component.spatial_context: copy.deepcopy(spatial_format),
    Component.segments:[] # segment_format_lev2
}

annotation_format = {
    Component.videos: [] # video_format
}