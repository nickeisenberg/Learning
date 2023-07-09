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

tech_top_50 = {}
sport_top_50 = {}
for sub0, sub1 in zip(tech_praw, sport_praw):

    tech_top_50[sub0.display_name] = []
    sport_top_50[sub1.display_name] = []

    subms0 = list(sub0.top(time_filter='all', limit=50))
    subms1 = list(sub1.top(time_filter='all', limit=50))

    for subm0, subm1 in zip(subms0, subms1):
        tech_top_50[sub0.display_name].append(subm0)
        sport_top_50[sub1.display_name].append(subm1)












