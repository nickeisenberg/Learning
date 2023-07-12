path = '/Users/nickeisenberg/GitRepos/'
path += 'DataSets_local/subreddit_texts/tech_vs_sport/'

string = '!!body!! starts_here'
string[8:]

with open(f'{path}tech.txt', 'r') as ops:
    lines = ops.readlines()
    for line in lines:
        if line.startswith('!!body!!'):
            line = line[8:].strip()
            with open(f'{path}tech_coms.txt', 'a') as opsc:
                _ = opsc.write(line)
                _ = opsc.write('\n')
        elif line.startswith('!!post!!'):
            line = line[8:].strip()
            with open(f'{path}tech_post.txt', 'a') as opsc:
                _ = opsc.write(line)
                _ = opsc.write('\n')

with open(f'{path}sport.txt', 'r') as ops:
    lines = ops.readlines()
    for line in lines:
        if line.startswith('!!body!!'):
            line = line[8:].strip()
            with open(f'{path}sport_coms.txt', 'a') as opsc:
                _ = opsc.write(line)
                _ = opsc.write('\n')
        elif line.startswith('!!post!!'):
            line = line[8:].strip()
            with open(f'{path}sport_post.txt', 'a') as opsc:
                _ = opsc.write(line)
                _ = opsc.write('\n')


