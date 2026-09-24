import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_train_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Initialize KNN Classifier (using k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# 4. Train Model & Predict
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# 5. Evaluate Performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_train_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN# knn
