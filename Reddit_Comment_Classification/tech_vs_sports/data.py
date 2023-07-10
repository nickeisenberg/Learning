import reddit_funcs as rfunc
import praw

tech_subs = [
    'tech',
    'computing',
    'windows',
    'mac',
    'linux',
]

sport_subs = [
    'sports',
    'nba',
    'soccer',
    'baseball',
    'mma'
]

client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
user_agent = 'phython_act'

tech_praw = []
sport_praw = []
for sub0, sub1 in zip(tech_subs, sport_subs):
    
    praw_sub = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent).subreddit(sub0)
    tech_praw.append(praw_sub)

    praw_sub = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent).subreddit(sub1)
    sport_praw.append(praw_sub)

no_of_submission = 100
tech_top = {}
sport_top = {}
for i, (sub0, sub1) in enumerate(zip(tech_praw, sport_praw)):

    tech_top[sub0] = {}
    sport_top[sub1] = {}
    
    subms0 = list(sub0.top(time_filter='all', limit=no_of_submission))
    subms1 = list(sub1.top(time_filter='all', limit=no_of_submission))

    for j, (subm0, subm1) in enumerate(zip(subms0, subms1)):
        prog = f"sub:{i + 1} / {len(tech_praw)} post:{j + 1} / {len(subms0)}"
        print(prog)
        tech_top[sub0][subm0] = subm0.comments[: 50]
        sport_top[sub1][subm1] = subm1.comments[: 50]

path = '/Users/nickeisenberg/GitRepos/'
path += 'DataSets_local/subreddit_texts/tech_vs_sport/'

# store all the subs, posts and commments
for i, k in enumerate(tech_top.keys()):
    for j, kk in enumerate(tech_top[k].keys()):
        if isinstance(kk, praw.models.MoreComments):
            continue
        for com in tech_top[k][kk]:
            prog = f'post:{i + 1}/{len(tech_top.keys())}'
            prog += f'com:{j + 1}/{len(tech_top[k].keys())}'
            print(prog)
            try:
                with open(f'{path}tech.txt', 'a') as opf:
                    _ = opf.write(f'!!subreddit!! {k.display_name}\n')
                    _ = opf.write(f'!!post!! {kk.title}\n')
                    _ = opf.write(f'!!body!! {com.body}\n')
            except:
                print('error')
                sub = k
                post = kk
                coms = tech_top[k][kk]
                break

for i, k in enumerate(sport_top.keys()):
    for j, kk in enumerate(sport_top[k].keys()):
        if isinstance(kk, praw.models.MoreComments):
            continue
        for com in list(sport_top[k][kk]):
            prog = f'post:{i + 1}/{len(sport_top.keys())}'
            prog += f'com:{j + 1}/{len(sport_top[k].keys())}'
            print(prog)
            try:
                with open(f'{path}sport.txt', 'a') as opf:
                    _ = opf.write(f'!!subreddit!! {k.display_name}\n')
                    _ = opf.write(f'!!post!! {kk.title}\n')
                    _ = opf.write(f'!!body!! {com.body}\n')
            except:
                print('error')
                sub = k
                post = kk
                coms = sport_top[k][kk]
                break
#--------------------------------------------------


# store commments
for i, k in enumerate(tech_top.keys()):
    for j, kk in enumerate(tech_top[k].keys()):
        if isinstance(kk, praw.models.MoreComments):
            continue
        for com in tech_top[k][kk]:
            prog = f'post:{i + 1}/{len(tech_top.keys())}'
            prog += f'com:{j + 1}/{len(tech_top[k].keys())}'
            print(prog)
            try:
                with open(f'{path}tech_coms.txt', 'a') as opf:
                    _ = opf.write(f'{com.body}\n')
            except:
                print('error')
                sub = k
                post = kk
                coms = tech_top[k][kk]
                break

for i, k in enumerate(sport_top.keys()):
    for j, kk in enumerate(sport_top[k].keys()):
        if isinstance(kk, praw.models.MoreComments):
            continue
        for com in list(sport_top[k][kk]):
            prog = f'post:{i + 1}/{len(sport_top.keys())}'
            prog += f'com:{j + 1}/{len(sport_top[k].keys())}'
            print(prog)
            try:
                with open(f'{path}sport_coms.txt', 'a') as opf:
                    _ = opf.write(f'{com.body}\n')
            except:
                print('error')
                sub = k
                post = kk
                coms = sport_top[k][kk]
                break
#--------------------------------------------------

with open(f'{path}tech.txt', 'r') as read_file:
    lines = read_file.readlines()
    tech_count = 0
    for line in lines:
        if line.startswith("!!body!!"):
            tech_count+= 1

with open(f'{path}sport.txt', 'r') as read_file:
    lines = read_file.readlines()
    sport_count = 0
    for line in lines:
        if line.startswith("!!body!!"):
            sport_count+= 1
        
tech_count
sport_count
