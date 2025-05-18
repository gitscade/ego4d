"""
common utility functions
"""
import argparse
import json
import os
# Argparse
def GetArgParser():
    return argparse.ArgumentParser(description="Argument or Popup")


def convert_single_to_double_quotes(messages):
    # Loop through each element in the list
    for message in messages:
        # Check if 'content' key exists in the message dictionary
        if 'content' in message:
            # Replace single quotes with double quotes in the 'content' string
            message['content'] = message['content'].replace("'", '"')
    return messages

def convert_single_to_double_quotes_in_tuple(data):
    """
    use this conversion for goalstep sequence...
    """
    # Convert the entire tuple to a string with double quotes
    return str(data).replace("'", '"')

def jsondump_agent_response(input):
    parsed_output = json.loads(input)

    # # Step 2: Dump it to a file
    # with open("output.json", "w") as f:
    #     json.dump(parsed_output, f, indent=2)

    parsed_output = json.dumps(parsed_output)
    return parsed_output

def remove_files(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
                count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    return count

# Save / Read


# BERT, BLUE, CLIP embeddings

# additional distances