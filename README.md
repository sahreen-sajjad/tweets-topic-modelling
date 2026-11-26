# Tweets Topic Modelling

A comparative natural language processing project investigating whether grouping short social media documents improves topic coherence using Latent Dirichlet Allocation (LDA).

## Historical results

| Corpus | Topics | C_v coherence |
| --- | ---: | ---: |
| Individual tweets | 17 | 0.3621 |
| Grouped tweets | 16 | 0.4287 |

Grouped tweets achieved approximately 18.4% higher coherence in the original notebook. These are historical results, not newly reproduced benchmarks.

## Methodology

1. Load individual tweets and aggregate preassigned groups.
2. Tokenise, stem, remove stopwords and rare tokens, and construct bigrams and trigrams.
3. Train separate Gensim LDA models with a fixed seed and 50 passes.
4. Evaluate coherence, perplexity, KL divergence, Jaccard distance and dominant topic contributions.
5. Export topic keywords, distributions, word clouds and optional interactive pyLDAvis maps.

## Repository layout

```text
src/
  data.py
  preprocessing.py
  model.py
  evaluate.py
  visualization.py
notebooks/topic_modelling.ipynb
main.py
requirements.txt
```

## Usage

Python 3.10 or 3.11 recommended. Install dependencies and NLTK stopwords:

```sh
pip install -r requirements.txt
python -m nltk.downloader stopwords
python main.py --tweets data/tweets.csv --grouped data/groupedTweets.csv
```

The original CSV files are not included. `tweets.csv` requires a `text` column; `groupedTweets.csv` requires `group` and `text` columns. Group assignments are supplied externally; the original notebook does not implement single-pass clustering.

Use `--sweep` for the original coarse and fine topic-count searches, `--html` for interactive topic maps, and `--passes` to adjust training cost. Results are written to `results/`.

## Refactoring notes

The original notebook is preserved with its outputs and attributions, including course material credited to Professor Joemon Jose and Gensim plotting examples. The original lemmatisation calls did not assign their results back to the corpus; the default pipeline preserves that effective behaviour. To enable lemmatisation, install `python -m spacy download en_core_web_sm` and pass `--lemmatize`. Perplexity is now calculated using Gensim's documented base-2 conversion, rather than the original exponential of the negative bound. Jaccard distance is reported directly without Gensim's optional matrix normalisation. Missing texts and empty cleaned tokens are handled explicitly. The original exploratory notebook remains the reference for historical experiments. The refactored pipeline has not been executed against the original data.
