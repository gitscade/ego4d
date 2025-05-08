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

query="""{"query": "Give me an phrase describing the activity of source_action_sequence.", "source_action_sequence": [["Cook tortilla soup", "Transfer tortilla wrap on the fridge", "Drink soft drink", "Mix spices into the soup", "Arrange items on the counter", "Add sweetcorn to the soup", "Drink the soft drink", "Rinse a peeler", "Heat a skillet", "Arrange items in the cabinet", "Cook minced meat"], ["Adds Oil to the Soup", "Stirs Soup", "Adds White pepper to Soup", "Adds Parsley", "Adds Chili Flakes to Soup", "Adds Spice to the soup", "Pours Sweetcorn into sieve", "Washes Sweetcorn in water", "Adds Sweetcorn into the Soup", "Stirs Soup", "Unpacks Minced meat", "Stirs minced meat"]], "source_scene_graph": [{"object_id": 3, "object_name": "soup", "init_status": {"status": "uncooked", "container": null}}, {"object_id": 14, "object_name": "counter top", "init_status": {"status": "default", "container": null}}, {"object_id": 4, "object_name": "oil", "init_status": {"status": "default", "container": 14}}, {"object_id": 5, "object_name": "stirrer", "init_status": {"status": "default", "container": 14}}, {"object_id": 6, "object_name": "fridge", "init_status": {"status": "default", "container": null}}, {"object_id": 7, "object_name": "tortilla", "init_status": {"status": "wrapped", "container": 6}}, {"object_id": 8, "object_name": "table", "init_status": {"status": "default", "container": null}}, {"object_id": 9, "object_name": "a bottle of water", "init_status": {"status": "contain water", "container": null}}, {"object_id": 19, "object_name": "stovetop", "init_status": {"status": "default", "container": null}}, {"object_id": 10, "object_name": "pepper spice", "init_status": {"status": "default", "container": 19}}, {"object_id": 11, "object_name": "herbs", "init_status": {"status": "default", "container": 19}}, {"object_id": 12, "object_name": "chilli flakes", "init_status": {"status": "default", "container": 19}}, {"object_id": 13, "object_name": "spice", "init_status": {"status": "default", "container": 19}}, {"object_id": 15, "object_name": "sweet corn", "init_status": {"status": "default", "container": 19}}, {"object_id": 16, "object_name": "sieve", "init_status": {"status": "default", "container": null}}, {"object_id": 17, "object_name": "sink", "init_status": {"status": "default", "container": null}}, {"object_id": 18, "object_name": "peeler", "init_status": {"status": "unwashed", "container": 17}}, {"object_id": 20, "object_name": "skillet", "init_status": {"status": "default", "container": 19}}, {"object_id": 21, "object_name": "player", "init_status": {"status": null}}, {"object_id": 22, "object_name": "minced meat", "init_status": {"status": "unpacked", "container": null}}, {"object_id": 23, "object_name": "knife", "init_status": {"status": "default", "container": null}}] }"""

query = """{"query": "Give me an phrase describing the activity of source_action_sequence.", "source_action_sequence": "Adds Oil to the Soup, Stirs Soup, Adds White pepper to Soup, Adds Parsley, Adds Chili Flakes to Soup, Adds Spice to the soup, Pours Sweetcorn into sieve, Washes Sweetcorn in water, Adds Sweetcorn into the Soup, Stirs Soup, Unpacks Minced meat, Stirs minced meat", "source_scene_graph": "[{'object_id': 3, 'object_name': 'soup', 'init_status': {'status': 'uncooked', 'container': None}}, {'object_id': 14, 'object_name': 'counter top', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 4, 'object_name': 'oil', 'init_status': {'status': 'default', 'container': 14}}, {'object_id': 5, 'object_name': 'stirrer', 'init_status': {'status': 'default', 'container': 14}}, {'object_id': 6, 'object_name': 'fridge', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 7, 'object_name': 'tortilla', 'init_status': {'status': 'wrapped', 'container': 6}}, {'object_id': 8, 'object_name': 'table', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 9, 'object_name': 'a bottle of water', 'init_status': {'status': 'contain water', 'container': None}}, {'object_id': 19, 'object_name': 'stovetop', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 10, 'object_name': 'pepper spice', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 11, 'object_name': 'herbs', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 12, 'object_name': 'chilli flakes', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 13, 'object_name': 'spice', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 15, 'object_name': 'sweet corn', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 16, 'object_name': 'sieve', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 17, 'object_name': 'sink', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 18, 'object_name': 'peeler', 'init_status': {'status': 'unwashed', 'container': 17}}, {'object_id': 20, 'object_name': 'skillet', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 21, 'object_name': 'player', 'init_status': {'status': None}}, {'object_id': 22, 'object_name': 'minced meat', 'init_status': {'status': 'unpacked', 'container': None}}, {'object_id': 23, 'object_name': 'knife', 'init_status': {'status': 'default', 'container': None}}]" }"""

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

query = json.loads(query)
QUERY=query["query"]
sequence=query["source_action_sequence"]
graph = query["source_scene_graph"]



graph ="""[{'object_id': 3, 'object_name': 'soup', 'init_status': {'status': 'uncooked', 'container': None}}, {'object_id': 14, 'object_name': 'counter top', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 4, 'object_name': 'oil', 'init_status': {'status': 'default', 'container': 14}}, {'object_id': 5, 'object_name': 'stirrer', 'init_status': {'status': 'default', 'container': 14}}, {'object_id': 6, 'object_name': 'fridge', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 7, 'object_name': 'tortilla', 'init_status': {'status': 'wrapped', 'container': 6}}, {'object_id': 8, 'object_name': 'table', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 9, 'object_name': 'a bottle of water', 'init_status': {'status': 'contain water', 'container': None}}, {'object_id': 19, 'object_name': 'stovetop', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 10, 'object_name': 'pepper spice', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 11, 'object_name': 'herbs', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 12, 'object_name': 'chilli flakes', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 13, 'object_name': 'spice', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 15, 'object_name': 'sweet corn', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 16, 'object_name': 'sieve', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 17, 'object_name': 'sink', 'init_status': {'status': 'default', 'container': None}}, {'object_id': 18, 'object_name': 'peeler', 'init_status': {'status': 'unwashed', 'container': 17}}, {'object_id': 20, 'object_name': 'skillet', 'init_status': {'status': 'default', 'container': 19}}, {'object_id': 21, 'object_name': 'player', 'init_status': {'status': None}}, {'object_id': 22, 'object_name': 'minced meat', 'init_status': {'status': 'unpacked', 'container': None}}, {'object_id': 23, 'object_name': 'knife', 'init_status': {'status': 'default', 'container': None}}]"""
graph = graph.replace("'", '"')
graph = json.dumps(graph)

# print(json.load(QUERY))
print({QUERY})
print(graph)


# #json dump to make this into json format
# print(json.dumps(sequence))

# # This only five double quote with NONE
# print(graph)

# # when dumped, escape character appears
# print(json.dumps(graph))






#"We could not parse the JSON body of your request. 

# q = json.loads(query.strip())
# # q = ast.literal_eval(query.strip())  # convert to python dict
# valid_json = json.dumps(1, indent=4)  # read as JSON(wth "")
# input_json = json.loads(valid_json)