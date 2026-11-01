#--------------------Import Libraries---------------------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


#--------------------Load Dataset---------------------
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/diabetes.csv")


#--------------------Select Features---------------------
feature_cols = ['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']

X = df[feature_cols]
y = df['Outcome']


#--------------------Split Dataset---------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


#--------------------Create Original Decision Tree---------------------
clf = DecisionTreeClassifier(random_state=5)

clf.fit(X_train, y_train)


#--------------------Prediction---------------------
y_pred = clf.predict(X_test)


#--------------------Performance---------------------
print(" ORIGINAL DECISION TREE ")
print()
print("************************************************************************************************************************")
print()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print()
print("************************************************************************************************************************")
print()
print("Recall:", metrics.recall_score(y_test, y_pred))
print()
print("************************************************************************************************************************")
print()
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print()
print("************************************************************************************************************************")
print()
print("Precision:", metrics.precision_score(y_test, y_pred))
print()
print("************************************************************************************************************************")
print()

print("Tree Depth:", clf.get_depth())
print()
print("************************************************************************************************************************")
print()
print("Tree Leaves:", clf.get_n_leaves())
print()
print("************************************************************************************************************************")
print()


#--------------------Display Tree---------------------
plt.figure(figsize=(20, 10))

plot_tree(clf,feature_names=feature_cols,class_names=['0', '1'],filled=True,rounded=True,fontsize=9)

plt.title("Original Decision Tree - Without Pruning")

plt.savefig("diabetes_original.png",dpi=300,bbox_inches="tight")

plt.show()
