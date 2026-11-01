import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,r2_score,mean_absolute_error,root_mean_squared_error

df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/Salary_Data.csv")

# Remove rows having missing values
df = df.dropna(subset=['Years of Experience', 'Salary'])

print("Head()\n")#print head
print(df.head())

print("Tail()\n")#print tail
print(df.tail())

print("describe() Function\n")#describe all dataset
print(df.describe())

#Print Scatter Plot
plt.scatter(df['Years of Experience'], df['Salary'], color='lightblue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

X = df[['Years of Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

#draw Regreesion Line
lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred_test = lm.predict(X_test)
y_pred_train = lm.predict(X_train)

plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, y_pred_train, color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='pink')
plt.show()

print()
print(f'Coefficient: {lm.coef_}')#print Coefficient
print(f'Intercept: {lm.intercept_}')#print intercept

mean_suq = mean_squared_error(y_test, y_pred_test)#print mean Square root error
mean_abs= mean_absolute_error(y_test, y_pred_test)#print Mean Absolute Error
r_squ = r2_score(y_test, y_pred_test)#print R square error
root_mean_squ= root_mean_squared_error(y_test,y_pred_test)#print Root mean Square Error

print("mean squared error:",mean_suq) 
print("mean absolute error:",mean_abs)
print("R-squared:",r_squ)
print("Root Mean Square Error :",root_mean_squ)