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

#===================================
# sequence evaluation
#===================================
def perform_sbert(sequence1:str, sequence2:str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(sequence1, convert_to_tensor=True)
    embedding2 = model.encode(sequence2, convert_to_tensor=True)
    cos_sim = util.cos_sim(embedding1, embedding2)
    return cos_sim

#===================================
# Compute graph similarity
#===================================
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


#TODO sentence-bert level similarity for two taxnomy

#===================================
# scene similarity
#===================================
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


#====================================
# scene similarity
#====================================
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_container_name(obj, graph):
    container_id = obj["init_status"].get("container")
    if container_id is None:
        return None
    for o in graph:
        if o["object_id"] == container_id:
            return o["object_name"]
    return None

def extract_edges(graph):
    return [(obj["object_name"], get_container_name(obj, graph))
            for obj in graph if obj["init_status"].get("container") is not None]

def compute_soft_matches(names1, names2, threshold=0.8):
    emb1 = model.encode(names1, convert_to_tensor=True)
    emb2 = model.encode(names2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb1, emb2)
    matches = []
    matched_1 = set()
    matched_2 = set()
    for i in range(len(names1)):
        for j in range(len(names2)):
            if cosine_scores[i][j] > threshold:
                matches.append((names1[i], names2[j], float(cosine_scores[i][j])))
                matched_1.add(i)
                matched_2.add(j)
    return matches, matched_1, matched_2

def compute_f1(len1, len2, matched_1, matched_2):
    precision = len(matched_1) / len1 if len1 else 0
    recall = len(matched_2) / len2 if len2 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def entity_level_similarity(graph1, graph2, threshold=0.8):
    names1 = [obj["object_name"] for obj in graph1]
    names2 = [obj["object_name"] for obj in graph2]
    matches, matched_1, matched_2 = compute_soft_matches(names1, names2, threshold)
    precision, recall, f1 = compute_f1(len(names1), len(names2), matched_1, matched_2)
    return {"precision": precision, "recall": recall, "f1_score": f1, "matches": matches}

def relationship_level_similarity(graph1, graph2, threshold=0.8):
    edges1 = extract_edges(graph1)
    edges2 = extract_edges(graph2)
    matches = []
    matched_1 = set()
    matched_2 = set()
    for i, (child1, parent1) in enumerate(edges1):
        for j, (child2, parent2) in enumerate(edges2):
            if parent1 is None or parent2 is None:
                continue
            child_sim = util.cos_sim(model.encode(child1), model.encode(child2)).item()
            parent_sim = util.cos_sim(model.encode(parent1), model.encode(parent2)).item()
            if child_sim > threshold and parent_sim > threshold:
                matches.append(((child1, parent1), (child2, parent2), child_sim, parent_sim))
                matched_1.add(i)
                matched_2.add(j)
    precision, recall, f1 = compute_f1(len(edges1), len(edges2), matched_1, matched_2)
    return {"precision": precision, "recall": recall, "f1_score": f1, "matches": matches}

def compare_scene_graphs(json_file_1, json_file_2, threshold=0.8):
    with open(json_file_1, 'r') as f:
        graph1 = json.load(f)

    with open(json_file_2, 'r') as f:
        graph2 = json.load(f)

    print("🔍 Entity-Level Similarity:")
    entity_result = entity_level_similarity(graph1, graph2, threshold)
    print(f"  Precision: {entity_result['precision']:.2f}")
    print(f"  Recall:    {entity_result['recall']:.2f}")
    print(f"  F1 Score:  {entity_result['f1_score']:.2f}")
    print(f"  Matches:   {entity_result['matches']}")

    print("\n🔗 Relationship-Level Similarity:")
    relation_result = relationship_level_similarity(graph1, graph2, threshold)
    print(f"  Precision: {relation_result['precision']:.2f}")
    print(f"  Recall:    {relation_result['recall']:.2f}")
    print(f"  F1 Score:  {relation_result['f1_score']:.2f}")
    print(f"  Matches:   {relation_result['matches']}")


