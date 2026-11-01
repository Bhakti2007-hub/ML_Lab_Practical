import pandas as pd
import matplotlib.pyplot as plt
from kmedoids import KMedoids

# Load the dataset
df = pd.read_csv("archive/Mall_Customers.csv")

# Print dataset and clean the dataset
print(df.head())
print()

print(df.tail())
print()

print(df.info())
print()

print(df.describe().T)
print()

print(df.isnull().sum())
print()

# Remove missing values
df=df.dropna()

# Perform K-Medoids algorithm
X=df[['Annual Income (k$)','Spending Score (1-100)']].values

kmedoids = KMedoids(n_clusters=5,metric='manhattan',random_state=38)

kmedoids.fit(X)

labels=kmedoids.labels_
medoids=kmedoids.cluster_centers_

# Plot the clusters
plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis',alpha=0.6)

# Plot medoids
plt.scatter(medoids[:,0],medoids[:,1],c='orange',marker='X',s=200,label='Medoids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Medoids')
plt.legend()
plt.show()