"""Corpus-specific preprocessing based on the original notebook."""
import re
from collections import Counter
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

EXTRA = set('http https co amp from dont else said subject re edu use not would say could _ be know good go get do done try many some nice thank think see rather easy easily lot lack make want seem run need even right line also may take come'.split())


def preprocess(texts, lemmatize=False):
    """Tokenise, stem, filter rare terms and construct bigrams/trigrams."""
    stemmer = SnowballStemmer('english')
    tokens = [[stemmer.stem(t) for t in tokenize(text, lower=True)] for text in texts]
    counts = Counter(t for doc in tokens for t in doc)
    excluded = set(stopwords.words('english')) | EXTRA
    excluded |= {t for t, n in counts.items() if n == 1 or len(t) == 1}
    cleaned = []
    for doc in tokens:
        row = []
        for token in doc:
            token = re.sub(r'[^\w\s]', '', token)
            token = re.sub(r'http\S+', '', token)
            token = ''.join(c for c in token if c.isalpha() or ord(c) < 128)
            if token and token not in excluded and not token.startswith('@'):
                row.append(token)
        cleaned.append(row)
    bigram = Phraser(Phrases(cleaned, min_count=5, threshold=100))
    trigram = Phraser(Phrases(bigram[cleaned], min_count=5, threshold=100))
    result = [list(trigram[bigram[doc]]) for doc in cleaned]
    if lemmatize:
        import spacy
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        result = [[t.lemma_ for t in doc if t.pos_ in {'NOUN', 'ADJ', 'VERB'}] for doc in nlp.pipe(' '.join(row) for row in result)]
    return result
