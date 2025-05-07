"""
common utility functions
"""
import argparse
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



# Save / Read


# BERT, BLUE, CLIP embeddings

# additional distances