import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("salaries.csv")
# print(df.head())

inputs=df.drop('salary_more_than_100k',axis='columns')
target=df['salary_more_than_100k']

# print(inputs)
# print(target)


