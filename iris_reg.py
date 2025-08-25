import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

flower=load_iris()
col=dir(flower)
# print(col)
# print(flower.data[0])

# print(flower.data_module)
x_train, x_test, y_train, y_test=train_test_split(flower.data,flower.target,test_size=0.2)

model=LogisticRegression()
model.fit(x_train,y_train)

score=model.score(x_test,y_test)
# print(score)

# print(flower.data[1])
y_predicted=model.predict(x_test)
print(y_predicted)
cm=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
# sn.heatmap(cm,annot=True)
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# plt.show()
