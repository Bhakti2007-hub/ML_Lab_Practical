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

original_clf = DecisionTreeClassifier(random_state=5)

original_clf.fit(X_train, y_train)


path = original_clf.cost_complexity_pruning_path(X_train,y_train)

ccp_alphas = path.ccp_alphas


print("Number of CCP Alpha values:", len(ccp_alphas))


#--------------------Select Alpha---------------------

if len(ccp_alphas) > 1:
    alpha = ccp_alphas[-2]
else:
    alpha = ccp_alphas[0]

print("Selected CCP Alpha:", alpha)


clf = DecisionTreeClassifier(ccp_alpha=alpha,random_state=5)

clf.fit(X_train, y_train)


#--------------------Prediction---------------------

y_pred = clf.predict(X_test)


#--------------------Performance---------------------

print("\n========== POST-PRUNED DECISION TREE ==========")

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))

print("Tree Depth:", clf.get_depth())
print("Tree Leaves:", clf.get_n_leaves())


#--------------------Display Tree---------------------

plt.figure(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=['0', '1'],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Post-Pruned Decision Tree")

plt.savefig(
    "diabetes_post_pruned.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Post-Pruned Decision Tree saved as diabetes_post_pruned.png")