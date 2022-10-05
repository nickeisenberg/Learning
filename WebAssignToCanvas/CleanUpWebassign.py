import pandas as pd
import numpy as np

# This program will take the WebAssign csv and clean it up so that
# it can easily be pushed to Canvas.
# The clean WebAssign csv will be called 'webassignCLEAN.csv'
# The program will compare the student list from canvas to Web assign and if
# a student is on WA but not Canvas, then the row will be deleted. If a student
# is on canvans but not web assign, then a row of zeros will be added.


# Enter the WebAssign and Canvas csv files
dfWA = pd.read_csv('WAuncleanfinal.csv')
dfC = pd.read_csv('Cuncleanfinal.csv')
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

## Add a column for Test scores from canvas
#
#tests = []
#for i in range(4):
#    tests.append(' '.join(['Test', str(i+1), '(']))
#
#tests.append('Final (')
#
#for name in dfC.columns.values.tolist(): 
#    for nm in tests:
#        if name.startswith(nm) == True:
#            addition = pd.DataFrame(dfC[name])
#            newname = ' '.join(name.split()[:len(name.split())-1])
#            addition = addition.rename(columns = {addition.columns[0]:newname})
#            addition.index = dfWA.index.values
#            dfWA = pd.concat([dfWA, addition], axis=1,)
##

# Store dfWA to a csv file
dfWA.to_csv('clean.csv')
