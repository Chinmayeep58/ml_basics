import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df=pd.read_csv("spam.csv")

# print(df.groupby('Category').describe())

df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
# print(df.head())

x_train, x_test, y_train, y_test=train_test_split(df.Message, df.spam, test_size=0.2)

# to convert the message to numbers, we use count vectorizer
v=CountVectorizer()
x_train_count=v.fit_transform(x_train.values)
# print(x_train_count.toarray()[:3])

# bernoulli naive bayes: it assumes that all the features are binary, can be represented by 0 and 1
# multinomial naive bayes: used when we have discrete data. in text learning we have to count the frequency of each word
# gaussian naive bayes: when all the features are continuous

model=MultinomialNB()
model.fit(x_train_count,y_train)

emails=[
    'hey mohan, can we meet today?',
    'upto 20% discount on parking, dont miss the offer'
]

emails_count=v.transform(emails)
# print(model.predict(emails_count))

x_test_count=v.transform(x_test)
# print(model.score(x_test_count,y_test))

# we use pipeline to avoid the repeated steps of vectorizing and then fitting or predicting
clf=Pipeline([('vectorizer',CountVectorizer()),('nb',MultinomialNB())])

clf.fit(x_test,y_test)
clf.predict(x_test,y_test)