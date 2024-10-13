import numpy as np

class MultipleLinearRegression():
    def __init__(self, learning_rate = 0.01, start_b = 0):
        self.w = None
        self.b = None
        self.trained = False
        self.start_b = start_b
        self.learning_rate = learning_rate 

    # x_train is a 2-d numpy array
    def train(self, x_train, y_train, epochs=1000):
        self.trained = True

        if x_train.shape[0] != y_train.shape[0]:
            raise Exception("X_train length should equal y_train length")

        # amount of elements in vector w should be equal 
        # to the amount of features in a single training example
        w = np.zeros(x_train.shape[1]) 
        b = self.start_b

        for i in range(epochs):
            print(f"Epoch={i}, error={self.__squared_error(w, b, x_train, y_train)} w = {w}, b = {b}")

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

        return np.dot(self.w, x) + self.b

    # derivative for vector w (not scalar)
    def __w_derivate(self, w, b, x_train, y_train):
        sum = np.zeros(x_train.shape[1])
        m = x_train.shape[0]

        for j in range (x_train.shape[1]):
            for i in range(m):
                f_wb = np.dot(w, x_train[i]) + b
                sum[j] = sum[j] + ((f_wb - y_train[i]) * x_train[i, j])

        return sum / m  # numpy array

    def __b_derivate(self, w, b, x_train, y_train):
        sum = 0
        m = x_train.shape[0]

        for i in range(m):
            f_wb = np.dot(w, x_train[i]) + b
            sum = sum + (f_wb - y_train[i])

        return sum / m

    def __squared_error(self, w, b, x_train, y_train):
        total_cost = 0
        m = x_train.shape[0]

        for i in range(m):
            f_wb =  np.dot(w, x_train[i]) + b
            step_cost = (f_wb  - y_train[i]) ** 2
            total_cost = total_cost + step_cost

        cost = (1 / (2 * m)) * total_cost

        return cost 