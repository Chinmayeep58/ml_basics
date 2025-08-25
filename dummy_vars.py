import pandas as pd
import numpy as np

df=pd.read_csv('homeprices_3.csv')
# print(df)

dummies=pd.get_dummies(df.town).astype(int)
# print(dummies)

merged=pd.concat([df,dummies],axis='columns')
final=merged.drop(['town','west windsor'],axis='columns')
# print(final)

from sklearn import linear_model

model=linear_model.LinearRegression()
x=final.drop('price',axis='columns')
# print(x)

y=final.price
model.fit(x,y)
prediction=model.predict([[2455,0,1]])
print(prediction)

print(model.score(x,y))

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
dfle=df
dfle.town=le.fit_transform(dfle.town)
x=df