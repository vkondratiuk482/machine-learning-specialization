import numpy as np

class RegularizedLogisticRegression():
    def __init__(self, learning_rate = 0.01, threshold = 0.5, regularization_lambda = 1, start_b = 0):
        self.w = None
        self.b = None
        self.trained = False
        self.start_b = start_b
        self.threshold = threshold
        self.learning_rate = learning_rate 
        self.regularization_lambda = regularization_lambda

    def train(self, x_train, y_train, epochs=1000):
        self.trained = True

        if len(x_train) != len(y_train):
            raise Exception("X_train length should equal y_train length")

        w = np.zeros(x_train.shape[1])
        b = self.start_b

        for i in range(epochs):
            print(f"Epoch={i}, error={self.__cost(w, b, x_train, y_train)}")
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

        prediction = self.__sigmoid(np.dot(self.w, x) + self.b)

        if prediction >= self.threshold:
            return 1

        return 0

    # derivative for vector w (not scalar)
    def __w_derivate(self, w, b, x_train, y_train):
        sum = np.zeros(x_train.shape[1])
        m = x_train.shape[0]

        for j in range (x_train.shape[1]):
            for i in range(m):
                f_wb = self.__sigmoid(np.dot(w, x_train[i]) + b)
                sum[j] += ((f_wb - y_train[i]) * x_train[i, j])
            sum[j] += (self.regularization_lambda / m) * w[j]

        return sum / m  # numpy array

    def __b_derivate(self, w, b, x_train, y_train):
        sum = 0
        m = x_train.shape[0]

        for i in range(m):
            f_wb = self.__sigmoid(np.dot(w, x_train[i]) + b)
            sum += (f_wb - y_train[i])

        return sum / m

    def __cost(self, w, b, x_train, y_train):
        sum = 0
        m = x_train.shape[0]
        n = x_train.shape[1]

        for i in range(m):
            f_wb = self.__sigmoid(np.dot(w, x_train[i]) + b)
            sum += self.__loss(f_wb, y_train[i])
        
        regularization_sum = 0

        for j in range(n):
            regularization_sum += w[j] ** 2

        sum += (self.regularization_lambda / (2 * m)) * regularization_sum

        return sum / (2 * m)

    def __loss(self, f_wb_i, y_train_i):
        return -y_train_i * np.log(f_wb_i) - (1 - y_train_i) * np.log(1 - f_wb_i)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
