import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df=pd.read_csv('homeprices_2.csv')
# print(df)


median_bedrooms=math.floor(df.bedrooms.median())
print(median_bedrooms)

df.bedrooms=df.bedrooms.fillna(median_bedrooms)
# print(df)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.predict([[23564, 3, 31]]))

