# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# -------------------- Load Dataset --------------------

# Load dataset from CSV file
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/diabetes.csv")

# Display first five rows
print("First Five Rows:\n")
print(df.head())
print()

# Display last five rows
print("Last Five Rows:\n")
print(df.tail())
print()

# Display dataset information
print("Dataset Information:\n")
df.info()
print()

# Check for missing values
print("Missing Values:\n")
print(df.isnull().sum())
print()

# -------------------- Data Preprocessing --------------------

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Remove missing values
df = df.dropna()

# -------------------- Features and Target --------------------

# Select all input features
X = df.drop("Outcome",axis=1)

# Select target variable
y = df["Outcome"]

# -------------------- Train-Test Split --------------------

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# -------------------- Logistic Regression --------------------

# Create Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train, y_train)

# -------------------- Predict Probability --------------------

# Predict probability for positive class
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

# -------------------- ROC Curve --------------------

# Calculate False Positive Rate and True Positive Rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate Area Under Curve (AUC)
roc_auc = auc(fpr, tpr)

# -------------------- Plot ROC Curve --------------------

plt.figure(figsize=(8,6))

# Plot ROC Curve
plt.plot(fpr,tpr,color="blue",linewidth=2,label="Logistic Regression with ROC curve")

# Plot Random Guess Line
plt.plot([0,1],[0,1],'r--',label="Random Guess")

# Graph Labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Diabetes Dataset")

# Show Legend
plt.legend()

# Display Grid
plt.grid(True)

# Show Plot
plt.show()

# -------------------- Display AUC Score --------------------

print("AUC Score :", roc_auc)