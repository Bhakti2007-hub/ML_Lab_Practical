import pandas as pd
import numpy as np
import matplotlib as mlt
from datetime import date

#initial
# use head()
print("----------- Display first 5 column ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.head(10))
print("***************************************************************************************************")

# use tail()
print("----------- Display last 5 column ----------- ")
df =pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.tail())
print("***************************************************************************************************")

#use info()
print("----------- Display All information ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.info())
print("***************************************************************************************************")

#use describe()
print("----------- Give Description ----------- ")
df=pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.describe())
print("***************************************************************************************************")

# data cleaning 
# find null value
print("----------- Find null values ----------- ")
df= pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.isnull()) 
print("***************************************************************************************************")

#sum of all null values
print("----------- Print Sum of all Null values ----------- ")
df= pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.isnull().sum())
print("***************************************************************************************************")

#handle missing values using dropna()
print("----------- Delete all null values ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.dropna())
print("***************************************************************************************************")

#handle missing values using fillna()
print("-----------Fill one number where null values ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.fillna(2))
print("***************************************************************************************************")

#delete dulicate values
print("----------- Remove duplicate values ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.drop_duplicates())
print("***************************************************************************************************")

#type conversion
print("----------- Convert one datatype to another datatype ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.astype(int))
print("***************************************************************************************************")

# give size of all datasheet
print("----------- Display Number of rows and number of Columns ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.shape)
print("***************************************************************************************************")

#showing values like count, mean, standard deviation, minimum and quartiles for each numerical column.
print("----------- display all count mena and all ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.describe().T)
print("***************************************************************************************************")

#show all column names 
print("----------- Display all column names ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.columns.tolist())
print("***************************************************************************************************")

#display sum of duplicate values
print("----------- Display sum of all duplicate values ----------- ")
df= pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/WineQT.csv")
print(df.duplicated().sum())
print("***************************************************************************************************")

#spliting string , date nd time , group by , find the age curr date nd purchase date , conversion of datatype 2
#date 14-july-2026
#find age and add age column

print("----------- Add Age column And find Age ----------- ")
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/used_cars_data.csv")
date.today().year
df['Age']= date.today().year-df['Year']
print(df.head())
print("***************************************************************************************************")


#split the column
print("----------- Split Name column into two another column Brand and Model ----------- ")
df['Brand'] = df.Name.str.split().str.get(0)
df['Model'] = df.Name.str.split().str.get(1)+df.Name.str.split().str.get(2)
print(df.head())
print("***************************************************************************************************")

#print 3 column
print("----------- Print 3 columns ----------- ")
print(df[['Name','Brand','Model']])
print("***************************************************************************************************")

print("----------- Print Categorical Data And Numnerical Data ----------- ")
#print Categorical Variables and Numerical Variables
cat_cols=df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)
print("***************************************************************************************************")


