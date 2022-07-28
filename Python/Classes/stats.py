# Class example for computing statistics

import numpy as np

class Stats:

    def __init__(self, nums):
        self.nums = nums

    def mean(self):
        count = 0
        total = 0
        for i in self.nums:
            total += i
            count += 1
        mean = total / count
        return mean

    def max(self):
        guess = self.nums[0]
        for i in self.nums:
            if i >= guess:
                guess = i
        return guess

    def min(self):
        guess = self.nums[0]
        for i in self.nums:
            if i <= guess:
                guess = i
        return guess

test_data = (1,2,3,4,5)
S = Stats(test_data)
data = S.nums
average = S.mean()
big = S.max()
small = S.min()

print(f'\n data = {data} \n average = {average} \n max = {big} \n min = {small}')


