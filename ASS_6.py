import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt


# =========================================================
# READ DATASET
# =========================================================

df = pd.read_csv(
    "C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/breast-cancer.csv"
)


# =========================================================
# PRINT INFORMATION
# =========================================================

print("Features :\n")
df.info()


# =========================================================
# DESCRIBE DATASET
# =========================================================

print("\nDescribe :\n")
print(df.describe())


# =========================================================
# PRINT FIRST FIVE ROWS
# =========================================================

print("\nFirst Five Rows :\n")
print(df.head())


# =========================================================
# PRINT LAST FIVE ROWS
# =========================================================

print("\nLast Five Rows :\n")
print(df.tail())


# =========================================================
# PRINT ROWS 5 TO 10
# =========================================================

print("\nPrint 5-10 Rows :\n")
print(df[5:10])


# =========================================================
# PRINT NUMBER OF ROWS AND COLUMNS
# =========================================================

print("\nPrint number of rows and columns:\n")
print(df.shape)


# =========================================================
# PRINT COLUMN NAMES
# =========================================================

print("\nFeatures:")
print(df.columns)


# =========================================================
# PRINT UNIQUE VALUES OF TARGET
# =========================================================

print("\nLabels:")
print(df["diagnosis"].unique())


# =========================================================
# REMOVE ID COLUMN IF PRESENT
# =========================================================

if "id" in df.columns:
    df = df.drop("id", axis=1)


# =========================================================
# CONVERT M AND B INTO 0 AND 1
# M = 0 = Malignant
# B = 1 = Benign
# =========================================================

df["diagnosis"] = df["diagnosis"].map({
    "M": 0,
    "B": 1
})


print("\nCancer Labels (0 = Malignant, 1 = Benign):")
print(df["diagnosis"])


# =========================================================
# SEPARATE FEATURES AND LABEL
# =========================================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# =========================================================
# SPLIT DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=109
)


print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)


# =========================================================
# CREATE SVM CLASSIFIER
# =========================================================

clf = svm.SVC(kernel="linear")


# =========================================================
# TRAIN MODEL
# =========================================================

clf.fit(X_train, y_train)


# =========================================================
# PREDICT TEST DATA
# =========================================================

y_pred = clf.predict(X_test)


# =========================================================
# ACCURACY
# =========================================================

accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# =========================================================
# PRECISION
# =========================================================

precision = metrics.precision_score(y_test, y_pred)

print("Precision:", precision)


# =========================================================
# RECALL
# =========================================================

recall = metrics.recall_score(y_test, y_pred)

print("Recall:", recall)


# =========================================================
# F1 SCORE USING FORMULA
# F1 = 2 * Precision * Recall / (Precision + Recall)
# =========================================================

f1 = (2 * precision * recall) / (precision + recall)

print("F1 Score:", f1)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = metrics.confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# GET TN, FP, FN, TP
# =========================================================

TN, FP, FN, TP = cm.ravel()

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)


# =========================================================
# SPECIFICITY
# Specificity = TN / (TN + FP)
# =========================================================

specificity = TN / (TN + FP)

print("Specificity:", specificity)


# =========================================================
# SVM HYPERPLANE VISUALIZATION
# =========================================================

# Select two features
X_plot = df[["radius_mean", "texture_mean"]].values

# Target values
y_plot = df["diagnosis"].values


# =========================================================
# SPLIT TWO FEATURES
# =========================================================

X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
    X_plot,
    y_plot,
    test_size=0.3,
    random_state=109
)


# =========================================================
# CREATE SVM MODEL FOR 2D VISUALIZATION
# =========================================================

svm_model = svm.SVC(kernel="linear")


# =========================================================
# TRAIN 2D SVM MODEL
# =========================================================

svm_model.fit(X_train_plot, y_train_plot)


# =========================================================
# PLOT DATA POINTS
# =========================================================

plt.scatter(
    X_train_plot[:, 0],
    X_train_plot[:, 1],
    c=y_train_plot,
    cmap="coolwarm",
    edgecolors="k"
)


# =========================================================
# GET HYPERPLANE PARAMETERS
# =========================================================

w = svm_model.coef_[0]

b = svm_model.intercept_[0]


# =========================================================
# CREATE X VALUES
# =========================================================

x_min = X_plot[:, 0].min()
x_max = X_plot[:, 0].max()

xx = np.linspace(
    x_min,
    x_max,
    100
)


# =========================================================
# HYPERPLANE EQUATION
#
# w0*x + w1*y + b = 0
#
# Therefore:
#
# y = -(w0*x + b) / w1
# =========================================================

yy = -(w[0] * xx + b) / w[1]


# =========================================================
# PLOT HYPERPLANE
# =========================================================

plt.plot(
    xx,
    yy,
    "k-",
    linewidth=2,
    label="Hyperplane"
)


# =========================================================
# CALCULATE MARGIN
# =========================================================

margin = 1 / np.sqrt(np.sum(w ** 2))


# =========================================================
# UPPER AND LOWER MARGINS
# =========================================================

yy_margin1 = yy + margin

yy_margin2 = yy - margin


# =========================================================
# PLOT MARGINS
# =========================================================

plt.plot(
    xx,
    yy_margin1,
    "k--",
    label="Margin"
)

plt.plot(
    xx,
    yy_margin2,
    "k--"
)


# =========================================================
# LABELS
# =========================================================

plt.xlabel("Radius Mean")

plt.ylabel("Texture Mean")

plt.title("SVM Linear Hyperplane")

plt.legend()

plt.grid(True)

plt.show()