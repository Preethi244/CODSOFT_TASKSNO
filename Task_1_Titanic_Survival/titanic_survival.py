import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# 1. Load the dataset

data = pd.read_csv("Titanic-Dataset.csv")


# -----------------------------
# 2. Data preprocessing

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Cabin contains many missing values, so it is removed
data = data.drop("Cabin", axis=1)


# -----------------------------
# 3. Select features and target

X = data[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

y = data["Survived"]


# -----------------------------
# 4. Convert categorical data
#    into numerical values

X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked"],
    drop_first=True
)


# -----------------------------
# 5. Split data into training
#    and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 6. Train the model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# -----------------------------
# 7. Make predictions

y_pred = model.predict(X_test)


# -----------------------------
# 8. Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print("Titanic Survival Prediction")
print("---------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# -----------------------------
# 9. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Titanic Survival Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.yticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# -----------------------------
# 10. Passenger Class
#     vs Survival

survival_by_class = pd.crosstab(
    data["Pclass"],
    data["Survived"]
)

survival_by_class.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Titanic Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)

plt.legend(
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()