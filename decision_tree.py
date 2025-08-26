import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

df=pd.read_csv("salaries.csv")
# print(df.head())

inputs=df.drop('salary_more_than_100k',axis='columns')
target=df['salary_more_than_100k']

# print(inputs)
# print(target)

le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])

# print(inputs)

inputs_n=inputs.drop(['company','job','degree'],axis='columns')
# print(inputs_n)

model=tree.DecisionTreeClassifier()
x_train,x_test,y_train,y_test=train_test_split(inputs_n,target,test_size=0.2)

model.fit(x_train,y_train)

sc=model.score(x_test,y_test)
# print(sc)

ans=model.predict([[2,2,1]])
print(ans)