"""
Ablations
- by ablation of system module structure (dependence between sequence / taxonomy)
- by differing similarity of scenes
- by ablation of vectorstore usage
- by differning llms (for testing for domain knowledge)

Visualize data, results
- t-SNE for datasets?
"""
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

# sentence bert for two sequence
def perform_sbert(sequence1:str, sequence2:str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(sequence1, convert_to_tensor=True)
    embedding2 = model.encode(sequence2, convert_to_tensor=True)
    cos_sim = util.cos_sim(embedding1, embedding2)
    return cos_sim

# Compute graph similarity
def graph_to_triplets(scene_graph, id_to_name=None):
    id_to_name = {obj['object_id']: obj['object_name'] for obj in scene_graph}
    triplets = set()
    for obj in scene_graph:
        name = obj['object_name']
        status = obj['init_status'].get('status', None)
        container_id = obj['init_status'].get('container', None)
        container_name = id_to_name.get(container_id, None) if container_id else None
        triplets.add((name, status, container_name))
    return triplets

def jaccard_similarity(triplets1, triplets2):
    intersection = triplets1 & triplets2
    union = triplets1 | triplets2
    return len(intersection) / len(union) if union else 1.0


# scene similarity
# jaccard for HARD STRING MATCH
def jaccard_dict_similarity(dict1, dict2):
    set1 = set(dict1.items())
    set2 = set(dict2.items())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0

# soft_similarity for somewhat different but contextually similar strings
def soft_dict_similarity(dict1, dict2):
    total_sim = 0
    count = 0
    for key in dict1.keys() | dict2.keys():
        val1 = dict1.get(key, "")
        val2 = dict2.get(key, "")
        if val1 == val2:
            sim = 1.0
        else:
            sim = SequenceMatcher(None, val1, val2).ratio()
        total_sim += sim
        count += 1
    return total_sim / count if count else 1.0

