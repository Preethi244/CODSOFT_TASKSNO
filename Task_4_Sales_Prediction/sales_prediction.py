import pandas as pd

# Load the dataset
data = pd.read_csv("advertising.csv")

# Display basic information
print("First 5 rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())
# Select features and target
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance:")
print("R² Score:", r2)
print("Mean Squared Error:", mse)
import matplotlib.pyplot as plt

# Plot actual vs predicted sales
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# TV Advertising vs Sales

plt.figure(figsize=(8, 5))

plt.scatter(data["TV"], data["Sales"])

plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")

plt.tight_layout()
plt.savefig("tv_vs_sales.png")
plt.show()