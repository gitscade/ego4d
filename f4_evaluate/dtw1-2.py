# %matplotlib qt
import sys
import os
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import mplcursors
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from bert_score import score
import pickle

#soft dtw
import torch
from tslearn.metrics import soft_dtw

sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
from util import util_constants
from util import util_funcs
import f4_evaluate.evaluate_scene as evaluate_scene
import f1_init.database_init as database_init
import f1_init.agent_init as agent_init
import f1_init.constants_init as constants_init

#Computing similarity
import json
from sentence_transformers import SentenceTransformer
from bert_score import score
import numpy as np
from numpy.linalg import norm
import ast
import pandas as pd


#BASELINE FROM v8/ index range(300) from jch 
PATH_CURR_FOLDER = os.path.abspath('') 

path_b0_p1 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base0-part1-300.json'))
path_b0_p2 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base0-part2-300.json'))

path_b1_1_p1 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base1-1-part1-300.json'))
path_b1_1_p2 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base1-1-part2-300.json'))

path_b1_2_p1 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base1-2-part1-300.json'))
path_b1_2_p2 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base1-2-part2-300.json'))

path_b2_1_p1 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base2-1-part1-300.json'))
path_b2_1_p2 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base2-1-part2-300.json'))

path_b2_2_p1 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base2-2-part1-300.json'))
path_b2_2_p2 = os.path.abspath(os.path.join(PATH_CURR_FOLDER, 'result_v8/ispossible/base2-2-part2-300.json'))



def load_pickle_file(path):
    try:          
         with open(path, "rb") as f:
             data = pickle.load(f)
             return data
    except:
         return None

def load_json_file(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data
    except:
        return None

def load_json_utf8(path):
    with open(path, 'r', encoding='utf-8') as f:
        dict = json.load(path)
    return dict


def check_pickle_file(path):
    try:
        with open(path, "rb") as f:
            pickle.load(f)  # try loading to ensure file is not empty or corrupted
        return True
    except (EOFError, FileNotFoundError, PermissionError, IsADirectoryError, pickle.UnpicklingError):
        return False

def check_json_file(path):
    try:
        with open(path, "rb") as f:
            json.load(f)  # try loading to ensure file is not empty or corrupted
        return True
    except (EOFError, FileNotFoundError, PermissionError, IsADirectoryError, pickle.UnpicklingError):
        return False   

def normalize_to_string(value):
    '''
    used so to cope with value from a dictionary that happens to be inside a list bracket    
    '''
    # If it's a list and the first element is a string, return that
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
        return value[0]
    # If it's already a string, return it
    elif isinstance(value, str):
        return value
    # Fallback for anything else
    else:
        return str(value)
    

# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

path_eval_results = constants_init.PATH_ROOT + "f4_evaluate/result_v6/"
baseline_folders = ['output-rag-0602','output-norag-0602','output-1goalmediation-0602','output-1goalmediation-norag-0602']

# #---------------------------------------
# # read human annotated list
# #---------------------------------------
import json

data_base0_p1 = load_json_file(path_b0_p1)
data_base0_p2 = load_json_file(path_b0_p2)

data_base1_1_p1 = load_json_file(path_b1_1_p1)
data_base1_1_p2 = load_json_file(path_b1_1_p2)
data_base1_2_p1 = load_json_file(path_b1_2_p1)
data_base1_2_p2 = load_json_file(path_b1_2_p2)

data_base2_1_p1 = load_json_file(path_b2_1_p1)
data_base2_1_p2 = load_json_file(path_b2_1_p2)
data_base2_2_p1 = load_json_file(path_b2_2_p1)
data_base2_2_p2 = load_json_file(path_b2_2_p2)

#.sort_values(by='idx').reset_index(drop=True)
df_base0 = pd.concat([pd.DataFrame(data_base0_p1)  , pd.DataFrame(data_base0_p2)  ], ignore_index=True)
df_base1_1 = pd.concat([pd.DataFrame(data_base1_1_p1)  , pd.DataFrame(data_base1_1_p2)  ], ignore_index=True)
df_base1_2 = pd.concat([pd.DataFrame(data_base1_2_p1)  , pd.DataFrame(data_base1_2_p2)  ], ignore_index=True)
df_base2_1 = pd.concat([pd.DataFrame(data_base2_1_p1)  , pd.DataFrame(data_base2_1_p2)  ], ignore_index=True)
df_base2_2 = pd.concat([pd.DataFrame(data_base2_2_p1)  , pd.DataFrame(data_base2_2_p2)  ], ignore_index=True)
                    

#========================================
# seq (df object => list of double quoted strings) : read_df_seqence()
#========================================
def convert_df_sequence(sequence):
    '''
    converts sequences from df to iterable list of strings for similarity measurement later.

    args:
        sequence: object datatype from dataframe
    
    return
        list of actions, where each action is enclosed in double quotes
    '''
    if isinstance(sequence, list):
        return [str(x).strip().strip('"') for x in sequence]
    elif isinstance(sequence, str):
        # Fall back for stringified list (just in case)
        try:
            import ast
            parsed = ast.literal_eval(sequence)
            if isinstance(parsed, list):
                return [str(x).strip().strip('"') for x in parsed]
        except:
            pass
        # fallback to comma split
        return [s.strip().strip('"') for s in sequence.split(',')]
    else:
        raise ValueError(f"Unsupported type: {type(sequence)}")

#========================================
# tax (df object => list of two dictionaries) : json.loads()
#========================================
# source_tax = df_base2_rag.loc[4, 'source_taxonomy']
# target_tax = df_base2_rag.loc[4, 'target_taxonomy']
# source_tax = json.loads(source_tax)
# target_tax = json.loads(target_tax)
def apply_json_loads(tax):
    return json.loads(tax)


#========================================
# Apply to our dataframes(only once after dataset loading)
#========================================
def apply_seq_conversion_to_df(df:pd.DataFrame):
    df['source_action_sequence'] = df['source_action_sequence'].apply(convert_df_sequence)
    df['target_sequence'] = df['target_sequence'].apply(convert_df_sequence)

def apply_tax_conversion_to_df(df:pd.DataFrame):
    df['source_taxonomy'] = df['source_taxonomy'].apply(apply_json_loads)
    df['target_taxonomy'] = df['target_taxonomy'].apply(apply_json_loads)

apply_seq_conversion_to_df(df_base0)
apply_seq_conversion_to_df(df_base1_1)
apply_seq_conversion_to_df(df_base1_2)
apply_seq_conversion_to_df(df_base2_1)
apply_seq_conversion_to_df(df_base2_2)

apply_tax_conversion_to_df(df_base2_1)
apply_tax_conversion_to_df(df_base2_2)

print(df_base0['source_action_sequence'].dtype)
print(df_base0['target_sequence'].dtype)
print(df_base0['source_taxonomy'].dtype)
print(df_base0['target_taxonomy'].dtype)






#========================================
# check iterations and sizes of sequence and tax
#========================================
source_seq = df_base2_2.loc[4, 'source_action_sequence']
target_seq = df_base2_2.loc[4, 'target_sequence']

print(source_seq)
print(target_seq)

for item in source_seq:
    print(item)

for item in target_seq:
    print(item)

source_tax = df_base2_2.loc[4, 'source_taxonomy']
target_tax = df_base2_2.loc[4, 'target_taxonomy']

for item in source_tax:
    print(item)

for item in target_tax:
    print(item)

#========================================
# dtw between df columns
#========================================
import openai
import numpy as np
from fastdtw import fastdtw

openai.api_key = os.getenv("OPENAI_API_KEY")  

def embed_texts(text_list, model="text-embedding-3-large"):
    """Generate embeddings for a list of text strings."""
    response = openai.embeddings.create(
        model=model,
        input=text_list
    )
    return np.array([d.embedding for d in response.data])

def compute_dtw_distance(seq1, seq2):
    """Compute DTW distance between two sequences of text using embeddings."""
    if not isinstance(seq1, list) or not isinstance(seq2, list) or len(seq1) == 0 or len(seq2) == 0:
        return np.nan

    embeddings1 = embed_texts(seq1)
    embeddings2 = embed_texts(seq2)

    distance, path = fastdtw(embeddings1, embeddings2, dist=cosine)
    similarity = 1 - (distance / len(path))
    return similarity

def compute_dtw_column(df):
    """Compute DTW for each row of the DataFrame and store it in a new column."""
    dtw_results = []
    for i, row in df.iterrows():
        try:
            print(row['source_action_sequence'])
            print(row['target_sequence'])
            dtw_val = compute_dtw_distance(row['source_action_sequence'], row['target_sequence'])
            print(dtw_val)
        except Exception as e:
            print(f"Error on row {i}: {e}")
            dtw_val = np.nan
        dtw_results.append(dtw_val)
    df['dtw_similarity'] = dtw_results
    return df


apply_seq_conversion_to_df(df_base0)
apply_seq_conversion_to_df(df_base1_1)
apply_seq_conversion_to_df(df_base1_2)
apply_seq_conversion_to_df(df_base2_1)
apply_seq_conversion_to_df(df_base2_2)

# apply_tax_conversion_to_df(df_base2_1)
# apply_tax_conversion_to_df(df_base2_2)

print(df_base0['source_action_sequence'].dtype)
print(df_base0['target_sequence'].dtype)
print(df_base0['source_taxonomy'].dtype)
print(df_base0['target_taxonomy'].dtype)

# df_base0 = compute_dtw_column(df_base0)
# df_base1_1 = compute_dtw_column(df_base1_1)
df_base1_2 = compute_dtw_column(df_base1_2)
# df_base2_1 = compute_dtw_column(df_base2_1)
# df_base2_2 = compute_dtw_column(df_base2_2)

# df_base0.to_pickle("df_base0.pkl")
# df_base1_1.to_pickle("df_base1_1.pkl")
df_base1_2.to_pickle("df_base1_2.pkl")
# df_base2_1.to_pickle("df_base2_1.pkl")
# df_base2_2.to_pickle("df_base2_2.pkl")
