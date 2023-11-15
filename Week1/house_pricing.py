import linear_regression

model = linear_regression.LinearRegression(learning_rate=0.1)

x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

model.train(x_train, y_train, epochs=10000)

print(model.predict(11))