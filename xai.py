from explainable_ai import xai_wrapper
from sklearn.linear_model import LinearRegression
import numpy as np

#Linear Regreassion
@xai_wrapper
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

X = np.random.rand(100, 10)
y = np.random.rand(100)

model, explanation = train_model(X, y)

print("Model:", model)
print("Explanation:", explanation)