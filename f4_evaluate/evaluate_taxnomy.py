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
# taxonomy similarity
#===================================
# 1. jaccard for HARD STRING MATCH
def jaccard_dict_similarity(dict1, dict2):
    set1 = set(dict1.items())
    set2 = set(dict2.items())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0

# 2. soft_similarity for somewhat different but contextually similar strings. More like mechanical matcher. To Understand taxonomy and semantics, we need to use other stuff.
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


# 3. SOFT SIMILARITY USING EMBEDDINGS for semantics?
# 3. Embedding-based similarity (optional, if values are descriptive text)
# If values are long or descriptive (e.g., "add oil and stir over medium heat"), encode each value using Sentence-BERT and average the cosine similarity across keys.


