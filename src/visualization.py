"""Save reproducible plots and interactive LDA visualisations."""
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def heatmap(matrix, path, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(matrix, aspect='auto')
    ax.set(title=title, xlabel='Topic', ylabel='Document or topic')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def wordcloud(tokens, path):
    counts = Counter(t for doc in tokens for t in doc)
    if counts:
        WordCloud(background_color='white', width=1000, height=500).generate_from_frequencies(dict(counts.most_common(100))).to_file(str(path))


def topic_map(model, corpus, dictionary, path):
    import pyLDAvis
    import pyLDAvis.gensim_models
    pyLDAvis.save_html(pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False, n_jobs=1), str(path))


def sweep_plot(frame, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['coherence', 'perplexity']):
        ax.plot(frame['topics'], frame[metric], marker='o')
        ax.set(xlabel='Number of topics', ylabel=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
