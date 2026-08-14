import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Load dataset
df = pd.read_csv("iris.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# 2. Separate features and target
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 4. Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("\nModel training completed!")


# 5. Make predictions
y_pred = model.predict(X_test)


# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy:", accuracy * 100, "%")


# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 8. Confusion Matrix Graph
plt.figure(figsize=(8, 6))
plt.imshow(cm)

plt.title("Iris Flower Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1, 2],
    ["Setosa", "Versicolor", "Virginica"]
)

plt.yticks(
    [0, 1, 2],
    ["Setosa", "Versicolor", "Virginica"]
)

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )
plt.savefig("confusion_matrix.png")
plt.show()


# 9. Iris Flower Visualization
plt.figure(figsize=(10, 6))

for species in df["species"].unique():
    data = df[df["species"] == species]

    plt.scatter(
        data["petal_length"],
        data["petal_width"],
        label=species
    )

plt.title("Iris Flower Classification")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()

plt.savefig("iris_visualization.png")
plt.show()