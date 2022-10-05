import pandas as pd
import numpy as np
from operator import itemgetter

# This program will take the WebAssign csv and clean it up so that
# it can easily be pushed to Canvas.
# The clean WebAssign csv will be called 'webassignCLEAN.csv'
# The program will compare the student list from canvas to Web assign and if
# a student is on WA but not Canvas, then the row will be deleted. If a student
# is on canvans but not web assign, then a row of zeros will be added.


# Enter the WebAssign and Canvas csv files
dfWA = pd.read_csv('finaluncleanWA.csv')
dfC = pd.read_csv('finaluncleanC.csv')
#

# Remove homework columns
coldrop = list()
itemrm = ['Homework', 'Student', 'Total', 'Username']

for col in dfWA.columns[1:]:
    for item in dfWA[col]:
        for rm in itemrm:
            if rm in str(item):
                if col not in coldrop:
                    coldrop.append(col)

# All of the columns to remove are stored in coldrop
# Now remove the columns
for col in coldrop:
    dfWA = dfWA.drop([col], axis=1)
#

# Fix the Webassign column titles
# Remove the '#' from quizes and change the reviews to 'Test x Review'
dfWA = dfWA.rename(columns = {dfWA.columns[0]:'Student'})
assign = dfWA.loc[dfWA['Student'] == 'Assignment Name']

assval = assign.values.tolist()[0][1:]

count = 0
for x in assval:
    x = x.strip()
    title = x.split()[2:]
# use .title() to turn REVIEW into Review
    if 'REVIEW' in title:
        title = ' '.join(title[2:]+['Review']).title()
    else:
        title = ' '.join(title)
    assval[count] = title.replace('#', '')
    count = count + 1

for i in range(len(assval)):
    dfWA = dfWA.rename(columns = {dfWA.columns[i+1]:assval[i]})
#

# Turn ND and NS into 0
ndns = ['ND', 'NS']
for col in dfWA.columns[1:]:
    dfWA[col] = dfWA[col].replace(ndns, 0)
#

# Remove rows 0-4 and 6,7 and 35,36 #
# This is specific only for Web Assign Csv files
ind = dfWA.index.values
rowrm = np.concatenate((ind[0:5], ind[6:8], ind[35:37]), axis=None)
dfWA = dfWA.drop(rowrm)
# Reset the index
dfWA = dfWA.reset_index(drop=True)
#

# Remove in inactive students
# Lower case the names of all the studetns.
# Do this because some there will be discrepencies from WA to Canvas
WAstu = list(dfWA[dfWA.columns[0]][1:])
lWAstu = list()
for stu in WAstu:
    low = stu.lower()
    lWAstu.append(low)

Cstu = list(dfC['Student'][1: len(dfC['Student'])])
lCstu = list()
for stu in Cstu:
    low = stu.lower()
    lCstu.append(low)

for stu in WAstu:
    low = stu.lower()
    if low not in lCstu:
        dfWA = dfWA.loc[dfWA[dfWA.columns[0]] != stu]
#Reset the index
dfWA = dfWA.reset_index(drop=True)
#

# Add Canvas students who are not on WebAssign
# Set all their scores to 0
length = len(dfWA.iloc[0].values.tolist())

# Function to insert row in the dataframe
# www.geeksforgeeks.org/insert-row-at-given-position-in-pandas-dataframe/
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0
    # End value of upper half
    end_upper = row_number
    # Start value of lower half
    start_lower = row_number
    # End value of lower half
    end_lower = df.shape[0]
    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]
    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]
    # Combine the two lists
    index_ = upper_half + lower_half
    # Update the index of the dataframe
    df.index = index_
    # Insert a row at the end
    df.loc[row_number] = row_value
    # Sort the index labels
    df = df.sort_index()
    # return the dataframe
    return df

count = 0
for stu in Cstu:
    rowadd = np.zeros(length-1).tolist()
    low = stu.lower()
    if low not in lWAstu:
        rowadd.insert(0, stu)
        dfWA = Insert_row(count+1 , dfWA, rowadd)
    count = count + 1
#

# Change the index values
dfWA = dfWA.set_index('Student')

# Fix capitoliztion of indexes
indval = dfWA.index.values.tolist()
count = 0
for ind in indval:
    indval[count] = ind.title()
    count = count + 1

dfWA.index = indval
dfWA.index.name = 'Student'
#

# Group each column. Quizes ... Test Reviews...
# Fix the 'Student' column
colorder = [dfWA.columns.values.tolist()[0]]

for name in dfWA.columns.values.tolist()[1:]:
    if 'Quiz' in name:
        colorder = colorder + [name]
for name in dfWA.columns[1:]:
    if 'Review' in name:
        colorder = colorder + [name]

dfWA = dfWA[colorder]
#

# Add a column for Test scores from canvas

tests = []
for i in range(4):
    tests.append(' '.join(['Test', str(i+1), '(']))

tests.append('Final (')

for name in dfC.columns.values.tolist(): 
    for nm in tests:
        if name.startswith(nm) == True:
            addition = pd.DataFrame(dfC[name])
            newname = ' '.join(name.split()[:len(name.split())-1])
            addition = addition.rename(columns = {addition.columns[0]:newname})
            addition.index = dfWA.index.values
            dfWA = pd.concat([dfWA, addition], axis=1,)
dfWA.index.name = 'Student'
#

# Add a column for the quiz average
qavg = np.zeros(len(dfWA['Quiz 1.8']))
qavgDF = pd.DataFrame(qavg)
qavgDF.index = dfWA.index.values
qavgDF = qavgDF.rename(columns = {qavgDF.columns[0]:'Quiz Average'})
dfWA = pd.concat([dfWA, qavgDF], axis=1) 

# Add up the total points for all quizes  
total = dfWA.loc[dfWA.index.values[0], 'Quiz 1.8'] 
qtotal = []
for colname in dfWA.columns.values:
    if 'Quiz' in colname and 'Average' not in colname:
        qtotal.append(float(dfWA.loc[dfWA.index.values[0], colname]))
qtotalsum = sum(qtotal)
dfWA.loc[dfWA.index.values[0], 'Quiz Average'] = qtotalsum

# calculate the average quiz grade for each student 
for stu in dfWA.index.values[1:]:
    stuq = []
    for colname in dfWA.columns.values:
        if 'Quiz' in colname and 'Average' not in colname:
            stuq.append(float(dfWA.loc[stu, colname]))
    stuqsum = sum(stuq)
    dfWA.loc[stu, 'Quiz Average'] = round(100 * (stuqsum / qtotalsum), 2)
#

# Add a column for the Test Review Average 
# Drop the lowest score 
travg = np.zeros(len(dfWA[dfWA.columns.values[0]]))
travgDF = pd.DataFrame(travg)
travgDF.index = dfWA.index.values
travgDF = travgDF.rename(columns = {travgDF.columns[0]:'Test Review Average'})
dfWA = pd.concat([dfWA, travgDF], axis=1)

# find the average. drop the lowest.
for stu in dfWA.index.values[1:]:
    trscore = []
    for colname in dfWA.columns.values:
        if 'Review' in colname and 'Average' not in colname:
            num = float(dfWA.loc[stu, colname])
            den = float(dfWA.loc['Totals', colname])
            trscore.append([num , den, round(100 * num / den, 2)])
    trscore = sorted(trscore, key=itemgetter(2))
    trscore.remove(trscore[0])
    pts = []
    totpts = []
    for pair in trscore:
        pts.append(pair[0])
        totpts.append(pair[1])
    trscoreavg = round(100 * sum(pts) / sum(totpts), 2)
    dfWA.loc[stu, colname] = trscoreavg
#

# make a column for test average without repacing the worst with the final
tavgraw = np.zeros(len(dfWA[dfWA.columns.values[0]]))
tavgrawDF = pd.DataFrame(tavgraw)
tavgrawDF.index = dfWA.index.values
tavgrawDF = tavgrawDF.rename(columns = {tavgrawDF.columns[0]:'Test Average (raw)'})
dfWA = pd.concat([dfWA, tavgrawDF], axis=1)

# find the average and drop the lowest
for stu in dfWA.index.values[1:]:
    tscore = []
    for colname in dfWA.columns.values:
        if ('Test' in colname and 'Review' not in colname) and \
                ('Test' in colname and 'Average' not in colname):
            num = float(dfWA.loc[stu, colname])
            den = float(dfWA.loc['Totals', colname])
            tscore.append([num, den, round(100 * num / den, 2)])
    pts = []
    totpts = []
    for pair in tscore:
        pts.append(pair[0])
        totpts.append(pair[1])
    avg = round(100 * sum(pts) / sum(totpts), 2)
    dfWA.loc[stu, 'Test Average (raw)'] = avg
    
#

# make a column for the test average
# the final will replace the lowest test score 
tavg = np.zeros(len(dfWA[dfWA.columns.values[0]]))
tavgDF = pd.DataFrame(tavg)
tavgDF.index = dfWA.index.values
tavgDF = tavgDF.rename(columns = {tavgDF.columns[0]:'Test Average (best)'})
dfWA = pd.concat([dfWA, tavgDF], axis=1)

# find the average and drop the lowest
for stu in dfWA.index.values[1:]:
    tscore = []
    for colname in dfWA.columns.values:
        if ('Test' in colname and 'Review' not in colname) and \
                ('Test' in colname and 'Average' not in colname):
            num = float(dfWA.loc[stu, colname])
            den = float(dfWA.loc['Totals', colname])
            tscore.append([num, den, round(100 * num / den, 2)])
    tscore.append([float(dfWA.loc[stu, 'Final']), float(dfWA.loc['Totals', 'Final']), \
            round(100 * float(dfWA.loc[stu, 'Final']) / float(dfWA.loc['Totals', 'Final']), 2)])
    tscore = sorted(tscore, key=itemgetter(2))   
    tscore.remove(tscore[0])
    pts = []
    totpts = []
    for pair in tscore:
        pts.append(pair[0])
        totpts.append(pair[1])
    avg = round(100 * sum(pts) / sum(totpts), 2)
    dfWA.loc[stu, 'Test Average (best)'] = avg
#            

# Final grade before Final 
bfinal = np.zeros(len(dfWA[dfWA.columns.values[0]]))
bfinalDF = pd.DataFrame(bfinal)
bfinalDF.index = dfWA.index.values
bfinalDF = bfinalDF.rename(columns = {bfinalDF.columns[0]: 'Grade Before Final'})
dfWA = pd.concat([dfWA, bfinalDF], axis=1)

for stu in dfWA.index.values[1:]:
    grade = round(
            .1 * float(dfWA.loc[stu, 'Quiz Average']) + 
            .1 * float(dfWA.loc[stu, 'Test Review Average']) + 
            .6 * float(dfWA.loc[stu, 'Test Average (raw)']), 2
            )
    dfWA.loc[stu, 'Grade Before Final'] = round(grade / .8, 2)


# calculate final grade 
fingd = np.zeros(len(dfWA[dfWA.columns.values[0]]))
fingdDF = pd.DataFrame(fingd)
fingdDF.index = dfWA.index.values
fingdDF = fingdDF.rename(columns = {fingdDF.columns[0]:'Final Grade'})
dfWA = pd.concat([dfWA, fingdDF], axis=1)

for stu in dfWA.index.values[1:]:
    grade = round(
            .1 * float(dfWA.loc[stu, 'Quiz Average']) +
            .1 * float(dfWA.loc[stu, 'Test Review Average']) +
            .6 * float(dfWA.loc[stu, 'Test Average (best)']) +
            .2 * float(dfWA.loc[stu, 'Final']), 2
            )
    dfWA.loc[stu, 'Final Grade'] = grade
#

# create the data frame for dr merchant.
# grade before final + final + grade after final
merchDF = pd.DataFrame(dfWA['Grade Before Final'])
merchDF = pd.concat([merchDF, dfWA['Final'], dfWA['Final Grade']], axis=1)
merchDF.index.name = 'Student'
merchDF.to_csv('math1120GL.csv')
#


dfWA.index.name = 'Student'
# Store dfWA to a csv file
dfWA.to_csv('clean.csv')
