# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression0
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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

# Convert required columns into numeric values
df['Glucose'] = pd.to_numeric(df['Glucose'], errors='coerce')
df['Outcome'] = pd.to_numeric(df['Outcome'], errors='coerce')

# Remove rows containing missing values
df = df.dropna(subset=['Glucose', 'Outcome'])

# -------------------- Feature Selection --------------------

# Independent variable (Feature)
X = df[['Glucose']]

# Dependent variable (Target)
y = df['Outcome']

# -------------------- Train-Test Split --------------------

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Logistic Regression Model --------------------

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# -------------------- Logistic Regression Curve --------------------

# Generate glucose values for plotting regression curve
glucose_range = np.linspace(df['Glucose'].min(),df['Glucose'].max(),300).reshape(-1, 1)

# Convert generated values into DataFrame
glucose_range_df = pd.DataFrame(glucose_range, columns=['Glucose'])

# Predict probabilities for regression curve
predicted_probs = model.predict_proba(glucose_range_df)[:, 1]

# -------------------- Scatter Plot + Logistic Regression Curve --------------------

plt.figure(figsize=(8,5))

# Scatter plot of actual data
sns.scatterplot(x='Glucose',y='Outcome', data=df,color='Black',alpha=0.6,label='Actual Data')

# Logistic Regression probability curve
plt.plot(glucose_range,predicted_probs,color='orange',linewidth=2,label='Logistic Regression Curve')

# -------------------- Scatter Plot --------------------

plt.figure(figsize=(8, 5))

# Scatter Plot
sns.scatterplot(x="Glucose",y="Outcome",data=df,color="red")

plt.title("Scatter Plot: Glucose vs Outcome")
plt.xlabel("Glucose")
plt.ylabel("Outcome")
plt.grid(True)
plt.show()

# -------------------- Confusion Matrix --------------------

# Generate confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

# Display confusion matrix
print(cm)

# -------------------- Single Value Prediction --------------------

# Create new glucose input
single = pd.DataFrame([[80]], columns=['Glucose'])

# Predict diabetes outcome
y_pred = model.predict(single)

print("Predicted label for glucose=145:")
print(y_pred[0])

# -------------------- Multiple Feature Model --------------------

# Select all features except Outcome
a = df.drop('Outcome', axis=1)

# Target variable
b = df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(a,b,test_size=0.2,stratify=y,random_state=16)

# -------------------- Logistic Regression --------------------

# Create Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Train model
logreg.fit(X_train, y_train)

# Predict output
y_pred = logreg.predict(X_test)

# -------------------- Confusion Matrix --------------------

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# -------------------- Heatmap --------------------

# Plot confusion matrix as heatmap
plt.figure(figsize=(5, 4))

sns.heatmap(cm,annot=True,fmt='d',cmap='Greens',xticklabels=['No', 'Yes'],yticklabels=['No', 'Yes'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -------------------- Performance Metrics --------------------

# Extract confusion matrix values
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

# Calculate Accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)

# Calculate Precision
precision = TP/(TP+FP)

# Calculate Recall (Sensitivity)
recall = TP/(TP+FN)

# Calculate F1 Score
f1_score = 2*(precision*recall)/(precision+recall)

# Calculate Specificity
specificity = TN/(TN+FP)

# -------------------- Display Performance --------------------

print("Accuracy    :", accuracy)
print("Precision   :", precision)
print("Recall      :", recall)
print("F1 Score    :", f1_score)
print("Specificity :", specificity)