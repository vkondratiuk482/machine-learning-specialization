import numpy as np
from multiple_linear_regression import MultipleLinearRegression 
from normalizers.mean_normalizer import MeanNormalizer 
from normalizers.z_score_normalizer import ZScoreNormalizer

# normalizer = MeanNormalizer()
normalizer = ZScoreNormalizer()
model = MultipleLinearRegression(learning_rate=0.0002)

x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
normalizer.fit(x_train)

normalized_x_train = normalizer.transform(x_train)

y_train = np.array([2, 5, 8, 11, 14])

model.train(normalized_x_train, y_train, epochs=100000)

x = normalizer.transform(np.array([200, 300, 400]))

prediction = model.predict(x)

print(f"Model prediction is {prediction}")