import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_json('News_Category_Dataset_v2.json',lines=True)
print(data[data['headline'].isnull()==True])
print(data[data['short_description'].isnull()==True])
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


grouped=data.groupby('category')
print(grouped.ngroups)
print(grouped.describe())

plt.figure('Histogram 1',figsize=(25,25))
plt.suptitle('Histogram Jumlah Kata Headline Per Kategori',size=25)
i=1
for group in grouped:
    plt.subplot(5,5,i)
    plt.title(group[0])
    plt.hist(group[1]['hwlen'],bins='auto')
    i+=1    
plt.subplots_adjust(hspace=.6,wspace=.4)
plt.savefig('./histogramh.png',format='png')

plt.figure('Histogram 2',figsize=(25,25))
plt.suptitle('Histogram Jumlah Kata Deskripsi Per Kategori',size=25)
i=1
for group in grouped:
    plt.subplot(5,5,i)
    plt.title(group[0])
    plt.hist(group[1]['dwlen'],bins='auto')
    i+=1    
plt.subplots_adjust(hspace=.6,wspace=.4)
plt.savefig('./histogramd.png',format='png')

plt.figure('Histogram 3',figsize=(25,25))
plt.suptitle('Histogram Jumlah Kata Features Per Kategori',size=25)
i=1
for group in grouped:
    plt.subplot(5,5,i)
    plt.title(group[0])
    plt.hist(group[1]['fwlen'],bins='auto')
    i+=1    
plt.subplots_adjust(hspace=.6,wspace=.4)
plt.savefig('./histogramf.png',format='png')

plt.show()
