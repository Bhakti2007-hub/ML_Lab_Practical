import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#Load Dataset
bank_data = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/data.csv")
print("Dataset loaded successfully!")
print(bank_data.head())
print("\nDataset shape:", bank_data.shape)

#Convert Target Columns
bank_data['default'] = bank_data['default'].map({'no': 0,'yes': 1,'unknown': 0})

#Convert Categorical Data
bank_data['y'] = bank_data['y'].map({'no': 0,'yes': 1})
bank_data = pd.get_dummies(bank_data,drop_first=True)

#Remove Missing Values
bank_data = bank_data.dropna()

#Split Features and Target
X = bank_data.drop('y', axis=1)
y = bank_data['y']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Create Random Forest Model
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#Evaluation Matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1SCORE = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1_SCORE :", F1SCORE)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["No", "Yes"])
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

#Visualize First 3 Trees
for i in range(3):
    plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[i],feature_names=X_train.columns,class_names=["No","Yes"],filled=True,max_depth=2,fontsize=8)
    plt.title(f"Random Forest - Decision Tree {i + 1}")
    plt.show()
