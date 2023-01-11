import numpy as np

class MinMaxScaler():

    def fit(self, data):
        self.m, self.M = np.min(data), np.max(data)
        self.data = data

    def transform(self):
        self.data_scale = (self.data - self.m) / (self.M - self.m)
        return self.data_scale

    def inverse_transform(self):
        orig = (self.M - self.m)
        return (self.data_scale * (self.M - self.m)) + self.m

if __name__ == '__main__':

    Mm = MinMaxScaler()
    data = np.array([1,2,3,4], dtype=float)
    print(f'the input data: {data}')

    Mm.fit(data)
    print(Mm.m, Mm.M)
    data_scaled = Mm.transform()
    original_data = Mm.inverse_transform()

    print(f'the scaled data: {data_scaled}')
    print(f'the reversed scaled data: {original_data}')
