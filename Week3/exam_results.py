import numpy as np
from regularized_logistic_regression import RegularizedLogisticRegression

# marks for the past 2 exams
x_train = np.array([[100, 90], [88, 79], [55, 47], [60, 70], [80, 82]])
# result (1 - passed, 0 - failed)
y_train = np.array([1, 1, 0, 0, 1])

model = RegularizedLogisticRegression(learning_rate=0.0003, threshold=0.6)

model.train(x_train, y_train, epochs=300000)

prediction1 = model.predict([55, 47])
prediction2 = model.predict([80, 75])
prediction3 = model.predict([60, 60])
prediction4 = model.predict([81, 90])

print(f"For [55, 47] the model prediction is {prediction1}")
print(f"For [80, 75] the model prediction is {prediction2}")
print(f"For [60, 60] the model prediction is {prediction3}")
print(f"For [81, 90] the model prediction is {prediction4}")