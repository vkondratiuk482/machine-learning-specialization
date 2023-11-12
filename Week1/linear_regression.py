import numpy as np

class LinearRegression():
    def __init__(self):
        self.w = None
        self.b = None
        self.start = -100
        self.end = 100
        self.step = 0.5 
        self.trained = False

    def train(self, x_train, y_train):
        self.trained = True

        if len(x_train) != len(y_train):
            raise Exception("X_train length should equal y_train length")

        cost_dictionary = {}

        for w in np.arange(self.start, self.end, self.step):
            for b in np.arange(self.start, self.end, self.step):
                cost_dictionary[(w, b)] = self.__squared_error(w, b, x_train, y_train)
        
        (w, b) = min(cost_dictionary, key=cost_dictionary.get)

        self.w = w
        self.b = b

        return

    def predict(self, x):
        if self.trained != True:
            raise Exception("Model not trained yet")

        return self.w * x + self.b

    def __squared_error(self, w, b, x_train, y_train):
        total_cost = 0
        m = len(x_train)

        for i in range(m):
            f_wb =  w * x_train[i] + b
            step_cost = (f_wb  - y_train[i]) ** 2
            total_cost = total_cost + step_cost

        cost = (1 / (2 * m)) * total_cost

        return cost 