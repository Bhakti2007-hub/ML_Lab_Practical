import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

data=pd.read_csv('clustering.csv')
print(data.head())

X=data[["LoanAmount", "ApplicantIncome"]].copy()
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='orange')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()

K=3
Centroids=X.sample(n=K)
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='orange')
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='black')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()

diff=1
j=0
while diff!=0:
    XD=X.copy()
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["ApplicantIncome"]-row_d["ApplicantIncome"])**2
            d2=(row_c["LoanAmount"]-row_d["LoanAmount"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X.loc[:,i]=ED
        i=i+1
    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1]<min_dist:
                min_dist=row[i+1]
                pos=i+1
        C.append(pos)
    X.loc[:,"Cluster"]=C
    Centroids_new=X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
    if j==0:
        diff=1
        j=j+1
    else:
        diff=((Centroids_new["LoanAmount"]-Centroids["LoanAmount"]).sum()+(Centroids_new["ApplicantIncome"]-Centroids["ApplicantIncome"]).sum())
        print(diff)
    Centroids=Centroids_new

color=['blue','green','cyan']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=color[k])
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='black')
plt.xlabel('Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()