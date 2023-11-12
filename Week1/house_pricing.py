import linear_regression

model = linear_regression.LinearRegression()

x_train = [2, 4, 6, 8, 10]
y_train = [1, 2, 3, 4, 5]

model.train(x_train, y_train)

print(model.predict(7))