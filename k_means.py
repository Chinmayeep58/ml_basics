# process of clustering the data into groups
# the data is clustered based on the distance between the cluster node and data point
# put the cluster node such that it is the center of gravity
# repeat the process till the distance between the cluster node and data point doesn't change

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df=pd.read_csv("income.csv")
plt.scatter(df['Age'],df['Income($)'])
# plt.show()

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
# print(y_predicted)
df['cluster']=y_predicted
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]

plt.scatter(df0['Age'],df0['Income($)'])
plt.scatter(df1['Age'],df1['Income($)'])
plt.scatter(df2['Age'],df2['Income($)'])
plt.show()