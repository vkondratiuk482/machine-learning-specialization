import numpy as np

class LinearRegression():
    def __init__(self, learning_rate = 0.01, start_w = 0, start_b = 0):
        self.w = None
        self.b = None
        self.trained = False
        self.start_w = start_w
        self.start_b = start_b
        self.learning_rate = learning_rate 

    def train(self, x_train, y_train, epochs=1000):
        self.trained = True

        if len(x_train) != len(y_train):
            raise Exception("X_train length should equal y_train length")

        w = self.start_w
        b = self.start_b

        for i in range(epochs):
            # Gradient Descent implementation
            step_w = w - self.learning_rate * self.__w_derivate(w, b, x_train, y_train)
            step_b = b - self.learning_rate * self.__b_derivate(w, b, x_train, y_train)

            w = step_w
            b = step_b

        self.w = w
        self.b = b

        return

    def predict(self, x):
        if self.trained != True:
            raise Exception("Model not trained yet")

        return self.w * x + self.b

    def __w_derivate(self, w, b, x_train, y_train):
        sum = 0
        m = len(x_train)

        for i in range(m):
            f_wb = w * x_train[i] + b
            sum = sum + (f_wb - y_train[i]) * x_train[i]

        return sum / m 

    def __b_derivate(self, w, b, x_train, y_train):
        sum = 0
        m = len(x_train)

        for i in range(m):
            f_wb = w * x_train[i] + b
            sum = sum + (f_wb - y_train[i])

        return sum / m

    def __squared_error(self, w, b, x_train, y_train):
        total_cost = 0
        m = len(x_train)

        for i in range(m):
            f_wb =  w * x_train[i] + b
            step_cost = (f_wb  - y_train[i]) ** 2
            total_cost = total_cost + step_cost

        cost = (1 / (2 * m)) * total_cost

        return cost 