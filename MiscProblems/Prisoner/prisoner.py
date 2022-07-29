import numpy as np
import random
import pandas as pd

total = 100
names = np.linspace(0, total - 1, total)
boxes = np.linspace(0, total - 1, total)
np.random.shuffle(boxes)

results = {}

for name in names:
    count = 1
    guess = np.random.choice(names)
    while count < 2 * total:
        if count > total / 2:
            results[f'Person {name}'] = 'fail'
            break
        elif boxes[int(guess)] == name:
            results[f'Person {name}'] = count
            break
        else:
            count += 1
            guess = boxes[int(guess)]

data = pd.DataFrame.from_dict(results, orient='index')
data.to_csv('data.csv')

