import numpy as np
import pandas as pd

total = 100
num_of_trials = 100
names = [*range(total)]
boxes = [*range(total)]

def prisoner_sim(total, names, boxes, num_of_trials):
    results = {}
    for name in names:
        results[f'Person {name + 1}'] = []
    for trial in range(1, num_of_trials + 1):
        np.random.shuffle(boxes)
        # print(f'Trial {trial}')
        # print(names)
        # print(boxes)
        for name in names:
           count = 1
           guess = name
           # print(f'First guess for Person {name + 1} in trial {trial} = {guess}')
           while count < total:
                if count > total / 2:
                    results[f'Person {name + 1}'].append('fail')
                    break
                elif boxes[guess] == name:
                    results[f'Person {name + 1}'].append(count)
                    break
                else:
                    count += 1
                    guess = boxes[guess]
    data = pd.DataFrame.from_dict(results, orient='index')
    data_column_values = []
    for i in range(1, num_of_trials + 1):
        data_column_values.append(f'Trial {i}') 
    data.columns = data_column_values
    return data

data = prisoner_sim(total, names, boxes, num_of_trials)
data.to_csv('data.csv')

success_num = 0

for col in data.columns:
    if 'fail' in data[col].to_list():
        continue
    else:
        success_num += 1

print(f'The success rate is {success_num / num_of_trials}')
