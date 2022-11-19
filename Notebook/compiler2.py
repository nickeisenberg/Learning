import pandas as pd

dic = {'a' : [1, 2, 3],
       'b' : [ 2]}

print(dic)

print(dic.items())

for k, v in dic.items():
    print(k)
    print(v)

df = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in dic.items()]))

print(df)



