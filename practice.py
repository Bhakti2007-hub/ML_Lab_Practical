import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_square_score , mena_absolute_error_score

df = pd.read_csv("")

plt.scatter(df['Years of experence'],df['salary'],color='red')
plt.title("Years of ex VS salary")
plt.xlabel('Years of ex')
plt.ylabel('Salary')
plt.box(False)
plt.show()

x=[["Years of Experience"]]
y=["Salary"]

X_train , X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=202)

lm=LinearRegression()
lm.fit(X_train,y_train)

y_pred_test =lm.predict(X_test)
y_pred_train= lm.predict(X_train)

plt.scatter(y_pred_test,y_pred_train,color="blue")
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.lagend([X_train/y_pred,])








