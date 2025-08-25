import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("carprices.csv")
plt.scatter(df['Mileage'],df['Sell Price($)'])
# plt.show()

x=df[['Mileage','Age(yrs)']]
y=df['Sell Price($)']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)

model=LinearRegression()
model.fit(x_train,y_train)
print(model.predict(x_test))

print(model.score(x_test,y_test))