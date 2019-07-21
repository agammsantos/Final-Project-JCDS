import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib as jb


data=pd.read_json('News_Category_Dataset_v2.json',lines=True)
# print(data[data['headline'].isnull()==True])
# print(data[data['short_description'].isnull()==True])
# print(data.columns.values)
# print(data['category'].value_counts())

data.category = data.category.map(lambda x: "ARTS & CULTURE" if x == "ARTS" or x == "CULTURE & ARTS" else x)
data.category = data.category.map(lambda x: "WEIRD NEWS & COMEDY" if x == "COMEDY" or x == "WEIRD NEWS" else x)
data.category = data.category.map(lambda x: "BUSINESS" if x == "MONEY" else x)
data.category = data.category.map(lambda x: "EDUCATION" if x == "COLLEGE" else x)
data.category = data.category.map(lambda x: "HOME & LIVING" if x == "HEALTHY LIVING" or x == "WELLNESS" or x == "FIFTY" else x)
data.category = data.category.map(lambda x: "PARENTS" if x == "PARENTING" else x)
data.category = data.category.map(lambda x: "IMPACT" if x == "GOOD NEWS" else x)
data.category = data.category.map(lambda x: "ENVIRONMENT" if x == "GREEN" else x)
data.category = data.category.map(lambda x: "STYLE & BEAUTY" if x == "STYLE" else x)
data.category = data.category.map(lambda x: "FOOD & DRINK" if x == "TASTE" else x)
data.category = data.category.map(lambda x: "SCIENCE & TECH" if x == "SCIENCE" or x == "TECH"  else x)
data.category = data.category.map(lambda x: "WORLD NEWS" if x == "THE WORLDPOST" or x == "WORLDPOST" else x)
print(data['category'].value_counts())
# print(data['features'][data['category']=='COMEDY'])
# print(data['features'][data['category']=='WEIRD NEWS'])
# data=data.drop(data[data['category']=='GOOD NEWS'].index.values)
# data=data.drop(data[data['category']=='IMPACT'].index.values)

data['features']=data['headline']+' '+data['short_description']
data['features']=data['features'].apply(lambda x: x.lower())
data['hwlen'] = data['headline'].apply(lambda x: x.split(' ')).apply(len)
data['dwlen'] = data['short_description'].apply(lambda x: x.split(' ')).apply(len)
data['fwlen'] = data['features'].apply(lambda x: x.split(' ')).apply(len)
# print(data['features'].head())
# print(data['fwlen'].max())


x = data['features']
y = data['category']
xtr, xts, ytr, yts = train_test_split(x, y, test_size=.05, random_state=42)

multinomialPipeline = Pipeline([
    ('cv',CountVectorizer(stop_words='english')),
    ('classifier',MultinomialNB())
])
multinomialPipeline.fit(xtr,ytr)
multinomialPrediksi = multinomialPipeline.predict(xts)

complementPipeline = Pipeline([
    ('cv',CountVectorizer(stop_words='english')),
    ('classifier',ComplementNB())
])
complementPipeline.fit(xtr,ytr)
complementPrediksi = complementPipeline.predict(xts)

sgdcPipeline = Pipeline([
    ('cv',CountVectorizer(stop_words='english')),
    ('classifier',SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=1e-3))
])
sgdcPipeline.fit(xtr,ytr)
sgdcPrediksi = sgdcPipeline.predict(xts)

print('Dengan metode MultinomialNB, diperoleh: ')
print(classification_report(yts,multinomialPrediksi))
print(confusion_matrix(yts,multinomialPrediksi))
print('\n')
print('Dengan metode ComplementNB, diperoleh: ')
print(classification_report(yts,complementPrediksi))
print(confusion_matrix(yts,complementPrediksi))
print('\n')
print('Dengan metode Stochastic Gradient Descent, diperoleh: ')
print(classification_report(yts,sgdcPrediksi))
print(confusion_matrix(yts,sgdcPrediksi))

head = "Tiger Woods misses Open cut, yearns for 'hot weeks'"
desc = 'Tiger Woods missed the cut in the Open Championship at Royal Portrush'
feat = head+' '+desc
print(complementPipeline.predict([feat]))
print(complementPipeline.predict_proba([feat]))

jb.dump(complementPipeline, 'modelComplement')
