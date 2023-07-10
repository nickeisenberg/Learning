import reddit_funcs as rfunc
import numpy as np
import keras
from keras import layers, Model, Input, Sequential

path = '/Users/nickeisenberg/GitRepos/'
path += 'DataSets_local/subreddit_texts/tech_vs_sport/'

# create the vocabulary
tech_proc = rfunc.TextProcessing()
tech_proc.get_words(
    f'{path}tech_coms.txt', ignore=['!!body!!'], from_file=True
)
tech_proc.get_vocab(no_words=10000)

sport_proc = rfunc.TextProcessing()
sport_proc.get_words(
    f'{path}sport_coms.txt', ignore=['!!body!!'], from_file=True
)
sport_proc.get_vocab(no_words=10000)

tech_proc.vocab
sport_proc.vocab

count = 0
for k in tech_proc.vocab:
    if k in sport_proc.vocab.keys():
        count += 1
count

