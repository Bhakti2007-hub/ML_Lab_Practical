import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/breast-cancer.csv")
print("First 5 Rows of Dataset:\n")
print(data.head(5))

print("Dataset Information:\n")
print(data.info())

print("Column Names:\n")
print(data.columns)

df = data
X = df.drop(['id','diagnosis'], axis=1)
y = df['diagnosis']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, val_index in kf.split(X):
    print("=" * 50)
    print("FOLD:", fold)
    print("=" * 50)

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]
    y_train = y.iloc[train_index]
    y_val = y.iloc[val_index]

    ada_clf = AdaBoostClassifier()

    print("AdaBoost Parameters:")
    print(ada_clf.get_params())

    ada_clf.fit(X_train, y_train)

    predictions = ada_clf.predict(X_val)

    print("Classification Report:")
    print(classification_report(y_val, predictions))

    cm = confusion_matrix(y_val, predictions)
    print("Confusion Matrix:")
    print(cm)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['B', 'M'])
    display.plot()
    plt.title("Confusion Matrix - Fold "+str(fold))
    plt.show()
    fold += 1