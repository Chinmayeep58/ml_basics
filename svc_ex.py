import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

digits=load_digits()
# print(dir(digits))
df=pd.DataFrame(digits.data,digits.target)
# print(df.head())
df['target']=digits.target
# print(df.head())

# print(digits.target_names)

# df['number']=df.target.apply(lambda x:digits.target_names[x])

# print(df.head())


x=df.drop(['target'],axis='columns')
y=df.target

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)

rbf_model=SVC(kernel='rbf')
rbf_model.fit(x_train,y_train)
print(rbf_model.score(x_test,y_test))

linear_model=SVC(kernel='linear')
linear_model.fit(x_train,y_train)
print(linear_model.score(x_test,y_test))