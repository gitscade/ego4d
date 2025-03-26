import json
import ast

input = """
{"query": "Give me a sequence of actions to fulfill the target_activity inside the environment of target_scene_graph", "target_activity":"clean kitchen", "target_scene_graph": {'room1': [{'type': 'avatar', 'name': 'player', 'status': 'stand'}, {'entity': {'entity': {'type': 'item', 'id': 1, 'name': 'gas stove', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 2, 'name': 'pot', 'status': 'default'}}, 'relation': 'has', 'target': {'type': 'item', 'id': 3, 'name': 'soup', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 5, 'name': 'spoon', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 6, 'name': 'container', 'status': 'default'}}, {'type': 'item', 'id': 7, 'name': 'sink', 'status': 'default'}, {'entity': {'type': 'item', 'id': 8, 'name': 'sink cabinet', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 9, 'name': 'peeler', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 10, 'name': 'potato', 'status': 'unpeeled'}}, {'entity': {'type': 'item', 'id': 8, 'name': 'sink cabinet', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 11, 'name': 'knife', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 12, 'name': 'salt', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 8, 'name': 'sink cabinet', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 13, 'name': 'spice container', 'status': 'default'}}, {'type': 'item', 'id': 14, 'name': 'cabinet', 'status': 'default'}, {'type': 'item', 'id': 15, 'name': 'trash can', 'status': 'default'}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 17, 'name': 'potato', 'status': 'unpeeled'}}, {'type': 'item', 'id': 18, 'name': 'refrigerator', 'status': 'default'}, {'type': 'item', 'id': 19, 'name': 'counter top', 'status': 'default'}]}, "lessons": [{"lesson": "Clean kitchen area"},{"lesson": "Clean up the kitchen area"},{"lesson": "clean kitchen utensils"}]}

"""

# Convert string to a proper Python dictionary
input_dict = ast.literal_eval(input.strip())  # Safely parse single-quoted structure

# Convert back to valid JSON (with double quotes)
valid_json = json.dumps(input_dict, indent=4)  # Pretty-print with double quotes

# print(valid_json)
# print(f"this is the input: {valid_json}")
params = json.loads(valid_json)
# print(params)


query = params.get("query")
print(query)