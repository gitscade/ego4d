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