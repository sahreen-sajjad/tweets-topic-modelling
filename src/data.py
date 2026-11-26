"""Load individual tweets and aggregate preassigned groups."""
import pandas as pd


def load_tweets(path, grouped=False):
    frame = pd.read_csv(path)
    required = {'text', 'group'} if grouped else {'text'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    frame = frame.dropna(subset=list(required)).copy()
    frame['text'] = frame['text'].astype(str)
    if grouped:
        frame = frame.groupby('group')['text'].agg(' '.join).reset_index()
    return frame.reset_index(drop=True)
