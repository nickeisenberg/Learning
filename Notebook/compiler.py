import pandas as pd

d_path = '/Users/nickeisenberg/GitRepos/Python_Misc/Notebook/DataSets/gme_df.csv'
d_path1 = '/Users/nickeisenberg/GitRepos/Python_Misc/Notebook/DataSets/gme_11_3_22.csv'

df = pd.read_csv(d_path)
df1 = pd.read_csv(d_path1)

df = pd.read_csv(d_path).rename(columns={df.columns[0] : 'Date'})
df1 = pd.read_csv(d_path).rename(columns={df1.columns[0] : 'Date'})

df_date = [d[:-6] for d in df['Date'].values]
df['Date'] = pd.Series(data=df_date)

print(df.head())
print('---')
date_ex = df_date[31]
print(date_ex)

print('--')
print(df.index[df['Date'] == date_ex])
print(df.index[df['Date'] == date_ex].tolist())

print(df.iloc[31])
