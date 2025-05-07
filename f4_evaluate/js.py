import json
import ast

messages = (['Stir the soup with the spatula', 'Add water to the soup', 'Stir the soup with the spatula', 'Cook the soup with the potato', 'Put the spice container on the shelf', 'Trash the potato peel', 'Wash the utensils', 'Stir the soup with the spatula', 'Put the potatoes in the fridge', 'Stir the soup with the spatula', 'Clean the counter top', 'Stir the soup with the spatula'], ['Fetch water in a container', 'add water to soup', 'peel outer skin of potato', 'stir soup', 'cut potato into soup', 'stir soup ', 'cut potato into soup', 'stir soup', 'add salt to soup', 'stir soup', 'cover pot with lid '])



def convert_single_to_double_quotes(messages):
    # Loop through each element in the list
    for message in messages:
        # Check if 'content' key exists in the message dictionary
        if 'content' in message:
            # Replace single quotes with double quotes in the 'content' string
            message['content'] = message['content'].replace("'", '"')
    return messages


messages = convert_single_to_double_quotes(messages)
# print(messages)


def convert_single_to_double_quotes_in_tuple(data):
    # Convert the entire tuple to a string with double quotes
    return str(data).replace("'", '"')

messages = convert_single_to_double_quotes_in_tuple(messages)
# print(messages)



spatial = {'room1': [{'type': 'avatar', 'name': 'player', 'status': 'stand'}, {'entity': {'entity': {'type': 'item', 'id': 1, 'name': 'gas stove', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 2, 'name': 'pot', 'status': 'default'}}, 'relation': 'has', 'target': {'type': 'item', 'id': 3, 'name': 'soup', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 5, 'name': 'spoon', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 6, 'name': 'container', 'status': 'default'}}, {'type': 'item', 'id': 7, 'name': 'sink', 'status': 'default'}, {'entity': {'type': 'item', 'id': 8, 'name': 'sink cabinet', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 9, 'name': 'peeler', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 10, 'name': 'potato', 'status': 'unpeeled'}}, {'entity': {'type': 'item', 'id': 8, 'name': 'sink cabinet', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 11, 'name': 'knife', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 12, 'name': 'salt', 'status': 'default'}}, {'entity': {'type': 'item', 'id': 8, 'name': 'sink cabinet', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 13, 'name': 'spice container', 'status': 'default'}}, {'type': 'item', 'id': 14, 'name': 'cabinet', 'status': 'default'}, {'type': 'item', 'id': 15, 'name': 'trash can', 'status': 'default'}, {'entity': {'type': 'item', 'id': 4, 'name': 'table', 'status': 'default'}, 'relation': 'has', 'target': {'type': 'item', 'id': 17, 'name': 'potato', 'status': 'unpeeled'}}, {'type': 'item', 'id': 18, 'name': 'refrigerator', 'status': 'default'}, {'type': 'item', 'id': 19, 'name': 'counter top', 'status': 'default'}]}


spatial = json.dumps(spatial)
# print(spatial)


# messages=[
#     json.dumps({
#     "role": "system", 
#     "content": "Return a serial of noun or words which is a specific and deep taxonomical description of source activity. For example, consider that ""cook steak"" is input activity, and sequence is about cooking steak. Construct a taxonomy of steak, that reflects the given action sequence. The taxonomy should take a form of: (cook)(steak)(meat)(salted)() and so on, with bracket enclosing noun description. The level of taxonomy should be at maxinum 5 levels deep."
#     }), 
#     json.dumps({ "role": "user", "content": "ddd""})
#         ]



query = """{"query": "Categorically describe source activity in a very specific way.", "source_action_sequence": "(["Stir the soup with the spatula", "Add water to the soup", "Stir the soup with the spatula", "Cook the soup with the potato", "Put the spice container on the shelf", "Trash the potato peel", "Wash the utensils", "Stir the soup with the spatula", "Put the potatoes in the fridge", "Stir the soup with the spatula", "Clean the counter top", "Stir the soup with the spatula"], ["Fetch water in a container", "add water to soup", "peel outer skin of potato", "stir soup", "cut potato into soup", "stir soup ", "cut potato into soup", "stir soup", "add salt to soup", "stir soup", "cover pot with lid "])", "source_scene_graph": "{"room1": [{"type": "avatar", "name": "player", "status": "stand"}, {"entity": {"entity": {"type": "item", "id": 1, "name": "gas stove", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 2, "name": "pot", "status": "default"}}, "relation": "has", "target": {"type": "item", "id": 3, "name": "soup", "status": "default"}}, {"entity": {"type": "item", "id": 4, "name": "table", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 5, "name": "spoon", "status": "default"}}, {"entity": {"type": "item", "id": 4, "name": "table", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 6, "name": "container", "status": "default"}}, {"type": "item", "id": 7, "name": "sink", "status": "default"}, {"entity": {"type": "item", "id": 8, "name": "sink cabinet", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 9, "name": "peeler", "status": "default"}}, {"entity": {"type": "item", "id": 4, "name": "table", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 10, "name": "potato", "status": "unpeeled"}}, {"entity": {"type": "item", "id": 8, "name": "sink cabinet", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 11, "name": "knife", "status": "default"}}, {"entity": {"type": "item", "id": 4, "name": "table", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 12, "name": "salt", "status": "default"}}, {"entity": {"type": "item", "id": 8, "name": "sink cabinet", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 13, "name": "spice container", "status": "default"}}, {"type": "item", "id": 14, "name": "cabinet", "status": "default"}, {"type": "item", "id": 15, "name": "trash can", "status": "default"}, {"entity": {"type": "item", "id": 4, "name": "table", "status": "default"}, "relation": "has", "target": {"type": "item", "id": 17, "name": "potato", "status": "unpeeled"}}, {"type": "item", "id": 18, "name": "refrigerator", "status": "default"}, {"type": "item", "id": 19, "name": "counter top", "status": "default"}]}" }"""


import re
def convert_to_valid_json(input_string):
    # Replace single quotes with double quotes
    input_string = input_string.replace("'", '"')
    
    # Ensure that quotes inside key names are double quotes
    input_string = re.sub(r'(?<=\w)\"(?=\w)', '', input_string)  # Remove escaped quotes if present
    
    # Ensure proper escaping for quotes inside the string
    input_string = input_string.replace('\"', '"')
    
    # Try to load and then reformat the string as valid JSON
    try:
        json_obj = json.loads(input_string)
        return json.dumps(json_obj, indent=4)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    

def convert_to_valid_json2(input_string):
    # Step 1: Fix the source_action_sequence by replacing internal single quotes with double quotes
    # This assumes the input string is trying to look like a JSON-compatible structure
    input_string = input_string.replace("'", '"')

    # Step 2: Fix the part where the source_action_sequence contains nested quotes.
    # Handle nested JSON-like strings in `source_action_sequence` and `source_scene_graph`
    input_string = re.sub(r'\"(\[.*?\])\"', r'\1', input_string)

    # Step 3: Escape problematic double quotes inside strings
    input_string = input_string.replace('\\"', '"')

    # Step 4: Try to load it into JSON format
    try:
        json_obj = json.loads(input_string)
        return json.dumps(json_obj, indent=4)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None



