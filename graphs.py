#17-7-2026

import pandas as pd
import matplotlib.pyplot as plt

#Bar Graph using for loop
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/used_cars_data.csv")

fuel_cou ={}
for fuel in df['Fuel_Type']:
    if fuel in fuel_cou:
        fuel_cou[fuel] +=1
    else:
        fuel_cou[fuel] =1
fuel_types=list(fuel_cou.keys())
counts = list (fuel_cou.values())

plt.figure(figsize=(8,16))
plt.bar(fuel_types, counts , color= 'skyblue')
plt.xlabel("Fuel Types")
plt.ylabel("Number of Cars")
plt.title("Car Count by fuel type")
plt.grid(axis='y', linestyle='--')
plt.show()

#without for loop bar graph

df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/used_cars_data.csv")
df.Fuel_Type.value_counts()

plt.figure(figsize=(8,10))
plt.bar(fuel_types,counts,color='red')
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.title("Car Count by Fuel Type")
plt.grid(axis='y',linestyle='--')
plt.show()

#pie chart 
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/used_cars_data.csv")
count = df['Owner_Type'].value_counts()
label = count.index
color = {'red','yellow','purple','orange'}
plt.pie(count,labels=label,colors=color)
plt.title("Car Count by there Owners")
plt.axis('equal')
plt.show()

#histogram 
df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/used_cars_data.csv")
fuel_types = df['Fuel_Type'].dropna().unique()
plt.figure(figsize=(8,10))

for fuel in fuel_types:
    prices = df[df['Fuel_Type']==fuel]['Price'].dropna()
    plt.hist(prices,bins=15, alpha=0.5,label=fuel,edgecolor='black')

plt.title('Price Distubution by car count')
plt.xlabel('Price')
plt.ylabel('Car Count')
plt.legend()
plt.grid(axis='y',linestyle='--')
plt.show()

#Scatter Graph

df = pd.read_csv("C:/Users/BHAKTI/OneDrive/Desktop/SCOE/MLL/archive/used_cars_data.csv")
plt.scatter(data=df, x = 'Location' , y ='Kilometers_Driven')
plt.title('Location vs Fuel_Type')
plt.xlabel('Location')
plt.ylabel('Fuel_Type')
plt.legend()
plt.grid()
plt.show()