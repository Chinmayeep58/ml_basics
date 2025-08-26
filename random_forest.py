#we divide the dataset into random datasets
#then make decision trees for each random dataset
#since many trees are formed, its called a forest

import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits=load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])

# plt.show()

df=pd.DataFrame(digits.data)
df['target']=digits.target
# print(df.head())

x=df.drop(['target'],axis='columns')
x_train, x_test, y_train, y_test=train_test_split(x,df.target,test_size=0.2)


model=RandomForestClassifier(n_estimators=40)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

y_predicted=model.predict(x_test)
cm=confusion_matrix(y_test,y_predicted)
sn.heatmap(cm,annot=True)
# plt.show()