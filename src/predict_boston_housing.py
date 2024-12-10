import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset from the data directory
data_path = "../data/boston_housing.csv"  # Make sure the path to your CSV is correct
df = pd.read_csv(data_path)

# Display basic dataset information
print("Dataset Information:")
print(df.info())  # Overview of data types, null values, etc.
print("\nFirst 5 rows of the dataset:")
print(df.head())  # Show first 5 rows to understand the structure

# Check for missing values in the dataset
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())  # Check for missing values in each column

# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"\n{duplicates} duplicate rows found. Removing them.")
    df = df.drop_duplicates()

# Summary statistics of the dataset
print("\nSummary Statistics:")
print(df.describe())  # Get a statistical summary of the dataset

# Correlation Heatmap to visualize correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# Distribution of the target variable (MEDV)
plt.figure(figsize=(8, 6))
sns.histplot(df["medv"], kde=True, color="blue")
plt.title("Distribution of House Prices (MEDV)")
plt.show()

# Boxplot to check for outliers in 'MEDV'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df["medv"], color="green")
plt.title("Boxplot of House Prices (MEDV)")
plt.show()

# Prepare features (X) and target variable (y)
X = df.drop(columns=["medv"])  # Drop the target column
y = df["medv"]  # Target column (House prices)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining Data Size: {X_train.shape[0]} samples")
print(f"Test Data Size: {X_test.shape[0]} samples")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Calculate errors (absolute differences between actual and predicted)
errors = np.abs(y_test - y_pred)

# Identify the top 5 samples with the largest errors
top_error_indices = errors.argsort()[-5:][::-1]

# Get the indices of the test set samples (which are the original indices from the dataset)
test_indices = y_test.index

# Print the top 5 misclassified samples using the original indices
print("\nTop 5 Misclassified Samples (High Errors):")
for i in top_error_indices:
    actual_index = y_test.index[i]  # Get the original index from y_test
    actual_value = y_test.iloc[i]  # Access the actual value using iloc
    predicted_value = y_pred[i]  # Access the predicted value
    error_value = errors.iloc[i]  # Access the error using iloc

    print(
        f"Index: {actual_index}, Actual: {actual_value}, Predicted: {predicted_value:.2f}, Error: {error_value:.2f}"
    )
