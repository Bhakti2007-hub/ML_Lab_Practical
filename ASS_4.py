import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    recall_score,
    precision_score,
)

# ---------------- Read Dataset ----------------
df = pd.read_csv("Loan_data.csv")

# ---------------- Display Dataset ----------------
print("First 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nNull Values")
print(df.isnull().sum())

print("\nDataset Information")
print(df.info())

print("\nDataset Description")
print(df.describe())

# ---------------- Remove Missing Values ----------------
df = df.dropna()

# ---------------- Count Plot ----------------
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.xticks(rotation=45)
plt.title("Purpose vs Loan Status")
plt.show()

# ---------------- Convert Categorical Column ----------------
le = LabelEncoder()
df["purpose"] = le.fit_transform(df["purpose"])

# ---------------- Split Dataset ----------------
X = df.drop("not.fully.paid", axis=1)
y = df["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40,random_state=25)

# ---------------- Train Gaussian Naive Bayes ----------------
model = GaussianNB()
model.fit(X_train, y_train)

# ---------------- Prediction ----------------
y_pred = model.predict(X_test)

# ---------------- Evaluation ----------------
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)

print("\nAccuracy :", accuracy)
print("F1 Score :", f1)
print("Recall :", recall)
print("Precision :", precision)

# ---------------- Confusion Matrix ----------------
labels = ["Fully Paid", "Not Fully Paid"]
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap="vlag")
plt.title("Confusion Matrix")
plt.show()

