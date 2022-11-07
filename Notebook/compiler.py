import pandas as pd

d_path = '/Users/nickeisenberg/GitRepos/Python_Misc/Notebook/DataSets/gme_df.csv'
d_path = '/Users/nickeisenberg/GitRepos/Python_Misc/Notebook/DataSets/gme_11_3_22.csv'
df = pd.read_csv(d_path)

print(df.head())
dfhead = df.iloc[:5]
print(dfhead)
