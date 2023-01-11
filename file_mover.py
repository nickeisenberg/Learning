import os
import shutil

path_0 = '/Users/nickeisenberg/GitRepos/Python_Notebook/Notebook/'
path_1 = '/Users/nickeisenberg/GitRepos/Python_Notebook/'

files_to_move = os.listdir(path_0)

for f in files_to_move:
    os.rename(f'{path_0}{f}', f'{path_1}{f}')

