import numpy as np

class P:
    def __init__(self, a, b):
        self.a = a
        self.b = b

# I believe that this notation was required in python2
class C(P):
    def __init__(self,c, d, *args):
        super(C, self).__init__(*args)
        self.c = c
        self.d = d

# This is ok in python3
class C_(P):
    def __init__(self,c, d, *args):
        super(, self).__init__(*args)
        self.c = c
        self.d = d

# example
c = C(1, 2, 3, 4)
c_ = C(1, 2, 3, 4)

c.a
c.b
c.c
c.d

c_.a
c_.b
c_.c
c_.d
