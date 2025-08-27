#k fold cross validation

# divide the dataset into k folds, with same amount of data in them
# then use one of them to test and rest to train
# repeat by changing the test fold, by taking different data segment each time
# take the average score
# this way the model is exposed to different data multiple times

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold

digits=load_digits()
x_train, x_test, y_train, y_test=train_test_split(digits.data, digits.target,test_size=0.2)

lr=LogisticRegression()
lr.fit(x_train,y_train)
# print(lr.score(x_test,y_test))

svc=SVC()
svc.fit(x_train,y_train)
# print(svc.score(x_test,y_test))

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
# print(rf.score(x_test,y_test))

kf=KFold(n_splits=3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)

def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)

# stratified k fold ensures the data division has uniformity
scores_l=[]
scores_svm=[]
scores_rf=[]

for train_index, test_index in kf.split(digits.data):
    x_train, x_test,y_train,y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]

    scores_l.append(get_score(LogisticRegression(),x_train,x_test,y_train,y_test))
    scores_svm.append(get_score(SVC(),x_train,x_test,y_train,y_test))
    scores_rf.append(get_score(RandomForestClassifier(),x_train,x_test,y_train,y_test))
    
print(scores_l)
print(scores_svm)
print(scores_rf)