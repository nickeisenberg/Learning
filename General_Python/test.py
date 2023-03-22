import numpy as np

class list_filler:
    
    def __init__(self):
        self.list = []
        self.fill_list()

    def fill_list(self):
        for k in range(10):
            self.list.append(k)

l = list_filler()

l.list

