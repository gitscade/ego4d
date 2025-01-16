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
def read_data(path):
    data = json.load(open(path))
    return data


def extract_goalstep_segments(data):
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

def extract_lev2_goalstep_segments(video):
    """
    func: return lev2 segments for a single video
    input: video: one video element of json loaded file
    output: lv2segments: [v1seg1lv2, v1seg2lv2,...]
    """
    lv2segments = []


    return lv2segments


def extract_spatial_context(video):
    """
    func: extract spatial_context section from the video dictionary
    input: video: video from which to extract spatial context
    output: spatial_context = []
    """
    spatial_context = []
    
    return spatial_context
