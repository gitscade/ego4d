
# =============================================
# CODE Credited to
# brown-palm / AntGPT
# https://github.com/brown-palm/AntGPT/blob/main/GPT_models/Finetuning/edit_distance.py
# =============================================

import editdistance
import numpy as np
#expect numpy array as input
def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)
