import reddit_funcs as rfunc
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/nickeisenberg/GitRepos/'
path += 'DataSets_local/subreddit_texts/tech_vs_sport/'

# create the vocabulary
tech_proc = rfunc.TextProcessing()
tech_proc.get_words(
    f'{path}tech_post.txt', ignore=['!!body!!'], from_file=True
)
tech_proc.get_vocab(no_words=10000)

sport_proc = rfunc.TextProcessing()
sport_proc.get_words(
    f'{path}sport_post.txt', ignore=['!!body!!'], from_file=True
)
sport_proc.get_vocab(no_words=10000)

# tech and sport words
tech_words = np.array([*tech_proc.vocab.keys()])
sport_words = np.array([*sport_proc.vocab.keys()])

# shared, total and unique
shared = np.intersect1d(tech_words, sport_words)
total = np.unique(np.hstack((tech_words, sport_words)))

total_key = {word: i for i, word in enumerate(total)}
shared_key = {word: total_key[word] for word in shared}

shared_count_tech = {}
tt = []
for word, i in shared_key.items():
    if word == '[UNK]':
        continue
    shared_count_tech[i] = tech_proc.word_counts[word]
    tt.append([i, tech_proc.word_counts[word]])

shared_count_sport = {}
ss = []
for word, i in shared_key.items():
    if word == '[UNK]':
        continue
    shared_count_sport[i] = sport_proc.word_counts[word]
    ss.append([i, sport_proc.word_counts[word]])

tt = np.array(tt)
ss = np.array(ss)

plt.scatter(tt[:,1], ss[:,1])
plt.show()
