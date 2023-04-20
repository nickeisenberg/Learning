import os
import pandas as pd
import praw
import yfinance as yf
from reddit_funcs import *

client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
user_agent = 'phython_act'

def get_the_comments(subreddit,
                     search,
                     client_id,
                     client_secret,
                     user_agent):

    praw_sub = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent).subreddit(subreddit)

    # get the daily discussion submissions
    subs = submission_getter(subreddit=praw_sub,
                             search=search,
                             no_of_submissions=100)
    
    # get the comments
    coms = comment_getter(submission_list=subs,
                          no_of_comments=100)

    return coms

def dataset_makers(com_dic, fn):
    with open(fn, 'w') as f:
        for s in com_dic.keys():
            f.write(f'!!post_title!!: {s.title}\n')
            for c in com_dic[s]:
                f.write(f'!!comment!!: {c.body}\n')
    return None

wsb = get_the_comments('wallstreetbets',
                       'Daily Discussion Thread',
                       client_id, client_secret, user_agent)

crypto = get_the_comments('CryptoCurrency',
                          'Daily General Discussion',
                          client_id, client_secret, user_agent)

comments = {'wsb': wsb, 'crypto': crypto}

for k in comments.keys():
    dataset_makers(comments[k], f'subreddit_texts/{k}.txt')
