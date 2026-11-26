"""Topic distributions, divergence and keyword comparisons."""
import numpy as np
import pandas as pd
from scipy.stats import entropy


def distributions(model, corpus):
    rows = [dict(model.get_document_topics(doc, minimum_probability=0)) for doc in corpus]
    return pd.DataFrame(rows, columns=range(model.num_topics)).fillna(0)


def dominant_topics(model, distribution):
    topics = distribution.idxmax(axis=1)
    return pd.DataFrame({'dominant_topic': topics, 'contribution': distribution.max(axis=1),
                         'keywords': [', '.join(w for w, _ in model.show_topic(int(k))) for k in topics]})


def kl_divergence(model):
    terms = model.get_topics()
    return np.array([[entropy(a, b) for b in terms] for a in terms])


def jaccard_distance(first, second):
    a = [set(w for w, _ in first.show_topic(k, topn=50)) for k in range(first.num_topics)]
    b = [set(w for w, _ in second.show_topic(k, topn=50)) for k in range(second.num_topics)]
    return np.array([[1 - len(x & y) / len(x | y) if x | y else 0 for y in b] for x in a])
