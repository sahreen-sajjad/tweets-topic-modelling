"""Run the comparative tweet topic modelling pipeline."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.data import load_tweets
from src.preprocessing import preprocess
from src.model import build_corpus, train, metrics, sweep
from src.evaluate import distributions, dominant_topics, kl_divergence, jaccard_distance
from src.visualization import heatmap, wordcloud, topic_map, sweep_plot


def run(path, grouped, topics, args):
    out = Path(args.output) / ('grouped' if grouped else 'individual')
    out.mkdir(parents=True, exist_ok=True)
    frame = load_tweets(path, grouped)
    tokens = preprocess(frame['text'], args.lemmatize)
    dictionary, corpus = build_corpus(tokens)
    model = train(dictionary, corpus, topics, args.passes)
    model.save(str(out / 'lda.model'))
    dictionary.save(str(out / 'dictionary'))
    scores = metrics(model, dictionary, corpus, tokens)
    dist = distributions(model, corpus)
    dominant = dominant_topics(model, dist)
    scores.update(documents=len(frame), strong_topics=int((dominant.contribution >= .5).sum()), weak_topics=int((dominant.contribution < .2).sum()))
    (out / 'metrics.json').write_text(json.dumps(scores, indent=2))
    pd.concat([frame, dominant], axis=1).to_csv(out / 'dominant_topics.csv', index=False)
    dist.to_csv(out / 'topic_distributions.csv', index=False)
    pd.DataFrame([{'topic': k, 'word': w, 'weight': float(p)} for k in range(topics) for w, p in model.show_topic(k, topn=10)]).to_csv(out / 'keywords.csv', index=False)
    heatmap(kl_divergence(model), out / 'kl_divergence.png', 'KL divergence')
    heatmap(jaccard_distance(model, model), out / 'jaccard.png', 'Jaccard distance')
    heatmap(dist, out / 'distributions.png', 'Document-topic distributions')
    wordcloud([text.lower().split() for text in frame.text], out / 'raw_wordcloud.png')
    wordcloud(tokens, out / 'processed_wordcloud.png')
    for k in range(topics):
        from wordcloud import WordCloud
        WordCloud(background_color='white').generate_from_frequencies(dict(model.show_topic(k))).to_file(str(out / f'topic_{k}.png'))
    if args.html:
        topic_map(model, corpus, dictionary, out / 'topics.html')
    if args.sweep:
        for label, counts in [('coarse', range(3, 41, 5)), ('fine', range(2, 21))]:
            results = sweep(dictionary, corpus, tokens, counts, args.passes)
            results.to_csv(out / f'sweep_{label}.csv', index=False)
            sweep_plot(results, out / f'sweep_{label}.png')
    print(out, scores)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tweets', default='data/tweets.csv')
    parser.add_argument('--grouped', default='data/groupedTweets.csv')
    parser.add_argument('--output', default='results')
    parser.add_argument('--passes', type=int, default=50)
    parser.add_argument('--individual-topics', type=int, default=17)
    parser.add_argument('--grouped-topics', type=int, default=16)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--lemmatize', action='store_true')
    args = parser.parse_args()
    first = run(args.tweets, False, args.individual_topics, args)
    second = run(args.grouped, True, args.grouped_topics, args)
    matrix = jaccard_distance(first, second)
    np.savetxt(Path(args.output) / 'cross_model_jaccard.csv', matrix, delimiter=',')
    heatmap(matrix, Path(args.output) / 'cross_model_jaccard.png', 'Cross-model Jaccard distance')


if __name__ == '__main__':
    main()
