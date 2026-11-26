"""LDA training and topic-count experiments."""
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import pandas as pd


def build_corpus(tokens):
    dictionary = Dictionary(tokens)
    if not len(dictionary):
        raise ValueError('No tokens remain after preprocessing.')
    return dictionary, [dictionary.doc2bow(doc) for doc in tokens]


def train(dictionary, corpus, topics, passes=50):
    return LdaModel(corpus=corpus, id2word=dictionary, num_topics=topics,
                    random_state=100, update_every=1, chunksize=100,
                    passes=passes, alpha='auto', per_word_topics=True)


def metrics(model, dictionary, corpus, tokens):
    bound = float(model.log_perplexity(corpus))
    return {'topics': model.num_topics,
            'coherence': float(CoherenceModel(model=model, texts=tokens, dictionary=dictionary, coherence='c_v', processes=1).get_coherence()),
            'log_perplexity': bound, 'perplexity': 2 ** (-bound)}


def sweep(dictionary, corpus, tokens, counts, passes=50):
    return pd.DataFrame([metrics(train(dictionary, corpus, k, passes), dictionary, corpus, tokens) for k in counts])
