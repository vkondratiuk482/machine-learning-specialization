from multiple_linear_regression import MultipleLinearRegression 
import numpy as np

model = MultipleLinearRegression(learning_rate=0.01)

x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
y_train = np.array([2, 4, 6, 8, 10])

model.train(x_train, y_train, epochs=4000)

print(model.predict(np.array([200, 200, 200])))
