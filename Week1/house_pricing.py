from linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.1)

x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

model.train(x_train, y_train, epochs=100)

print(model.predict(100))