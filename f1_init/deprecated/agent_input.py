# Agent input is copied in f1_init.agent_init->delete when needed


import json
# TODO: def function that reads one annotation file and selects level 2 actions in formatted forms. The function should return 
"""
- read from "root/data/ego4d_annotation/demo/input"
- choose video index to use as input data
- extract goalstep lv2 segments from video
- extract spatial context from video
- parse goalstep segments
- parse spatial context segments
"""

def extract_all_goalstep_segments(data):
    """
    func: 
    input: data: json loaded data
    output: segments: [v1seg1, v1seg2 ... v1segn, v2seg1, ... , v2segm, .....]
    """
    segments = []
    def recurse_segments(segment, parent_id=None, level=1):
        segment_id = segment.get("number")

        
        text = json.dumps(segment.get("context"))
        metadata = {
            "level": level,
            "segment_id": segment_id,
            "parent_id": parent_id,
            "video_uid": segment.get("video_uid"),
        }
        segments.append({"text": text, "metadata": metadata})

        # Process child segments recursively 
        for child_segment in segment.get("segments", []):
            recurse_segments(child_segment, parent_id=segment_id, level=level+1)

    for video in data["videos"]:
        recurse_segments(video, parent_id=None, level=1)    
    return segments            


def extract_lower_goalstep_segments(video):
    """
    func: return lev2 & lev3 segments for a single video
    input: video: one video element of json loaded file
    output: lv2segments: ['Kneads the dough with the mixer.', 'Pours the flour into the mixer', 'Organize the table', ...]
    output: lv3segments: ['weigh the dough', 'weigh the dough', 'weigh the dough', 'move scale to tabletop', ...]
    """
    lv2segments = []
    lv3segments = []

    #print(video)
    for i, level2_segment in enumerate(video.get("segments", [])):    
        lv2segments.append(level2_segment["step_description"])

        for j, level3_segment in enumerate(level2_segment.get("segments", [])):
            lv3segments.append(level3_segment["step_description"])

    #print(lv2segments)
    #print(lv3segments)
    return lv2segments, lv3segments


def extract_spatial_context(video):
    """
    func: extract spatial_context section from the video dictionary
    input: video: video from which to extract spatial context
    output: spatial_context = {'room1': [{'entity': {'type': 'avatar', 'name': 'player', 'status': 'sit'}, 'relation':...
    """
    return video["spatial_context"]




# if __name__=="__main__":

#     video = {"segments":[
#         {"level":2,
#         "context":{"content1":"xxx"},
#         "segments": 
#         [
#             {"level":3,
#             "context":{"content1-1":"xxx"},
#             },
#             {
#             "level":3,
#             "context":{"content1-2":"xxx"},
#             }
#         ]
#         },
#         {
#         "level":2,
#         "context":{"content2":"xxx"},
#         "segments": []
#         }
#     ]}

#     #print(video["segments"][0])
#     extract_lower_goalstep_segments(video)