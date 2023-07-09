import reddit_funcs as rfunc

client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
user_agent = 'phython_act'

wsb = rfunc.get_the_comments('wallstreetbets',
                       'Daily Discussion Thread',
                       client_id, client_secret, user_agent)

crypto = rfunc.get_the_comments('CryptoCurrency',
                          'Daily General Discussion',
                          client_id, client_secret, user_agent)

comments = {'wsb': wsb, 'crypto': crypto}

path = '/Users/nickeisenberg/GitRepos/DataSets_local/subreddit_texts/' 
for k in comments.keys():
    rfunc.dataset_makers_sc(comments[k], f'{path}{k}.txt')
