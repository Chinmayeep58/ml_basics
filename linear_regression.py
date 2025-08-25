import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# df=pd.read_csv("homeprices.csv")
# print(df)

# plt.scatter(df.area,df.price, color='red',marker='*')
# plt.xlabel('area')
# plt.ylabel('price')
# plt.show()

# reg=linear_model.LinearRegression()
# reg.fit(df.area.values.reshape(-1,1),df.price)
# print(reg.predict([[3300]]))

# plt.scatter(df.area,df.price,color='red',marker='+')
# plt.xlabel('area')
# plt.ylabel('price')
# plt.plot(df.area,reg.predict(df[['area']]),color='blue')
# plt.show()


# homework

df2=pd.read_csv('canada_per_capita_income.csv')

plt.scatter(df2.year,df2.per_capita_income)
plt.xlabel('year')
plt.ylabel('pca')
# plt.show()

reg=linear_model.LinearRegression()
reg.fit(df2[['year']],df2.per_capita_income)
# print(reg.predict([[2025]]))

plt.scatter(df2.year,df2.per_capita_income)
plt.xlabel('year')
plt.ylabel('pca')
plt.plot(df2.year, reg.predict(df2[['year']]),color='green')
# plt.show()

import pickle

with open('model_pickle','wb') as f:
    pickle.dump(reg,f)

with open('model_pickle','rb') as f:
    model=pickle.load(f)

print(model.predict([[2043]]))