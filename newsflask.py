import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_mysqldb import MySQL 
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import joblib as jb

app=Flask(__name__,static_url_path='/static')
mysql=MySQL(app)

app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='agammsantos'
app.config['MYSQL_PASSWORD']=base64.b64decode(b"RGFuY2VyMTE5OQ==").decode('utf-8')
app.config['MYSQL_DB']='news'

@app.route('/')
def home():
    return render_template('land.htm')

@app.route('/prediksi', methods=['POST'])
def predict():
    head=str(request.form['head']).lower()
    desc=str(request.form['desc']).lower()
    feature=head+' '+desc
    
    prediksi=model.predict([feature])[0]
    prediksi_l=prediksi.lower().split(' ')
    prediksi_u=[]
    for i in prediksi_l:
        prediksi_u.append(i[0].upper()+i[1:])
    kategori=' '.join(prediksi_u)
    
    probabilitas=model.predict_proba([feature])
    enumprob=list(enumerate(probabilitas[0]))
    enumprob.sort(key=lambda x:x[1],reverse=True)
    probutama=enumprob[:5]

    df=pd.read_csv('news.csv')
    sortedlist=df['category'].unique()
    sortedlist.sort()
    
    plotkategori=[]
    plotprob=[]
    for i in probutama:
        plotkategori.append(sortedlist[i[0]])
        plotprob.append(i[1]*100)

    plt.close()
    sns.set(style="darkgrid")
    sns.set_context("talk")
    ax=sns.barplot(plotprob,plotkategori,palette="Blues_d")
    ax.set(xlabel='Probability (%)',ylabel='')
    plotlist=[(plotprob[i],plotkategori[i]) for i in range(0,len(plotprob))]
    xticks=np.arange(0,101,20)
    index=0
    for a,b in plotlist:
        ax.text(a+13.5,index+0.1,str(round(a,2))+'%',color='black',ha="center")
        index+=1
    ax.set_xticks(xticks)
    plt.tight_layout()
    fig=ax.get_figure()
    
    img=io.BytesIO()
    fig.savefig(img,format='png',transparent=True)
    img.seek(0)
    graph_url=base64.b64encode(img.getvalue()).decode()
    graph='data:image/png;base64,{}'.format(graph_url)

    if (plotprob[0]-plotprob[1])<5 and (plotprob[1]-plotprob[2])<5 and (plotprob[2]-plotprob[3])<5 and (plotprob[3]-plotprob[4])<5:
        statement="We must say this one is too hard to predict. The words are just too random. Here's the visualization for you: "
    elif (plotprob[0]-plotprob[1])<5 and (plotprob[1]-plotprob[2])<5:
        statement="Though it seems we're likely wrong about the prediction. Worry not, we have the tendencies for you: "
    elif (plotprob[0]-plotprob[1])<5:
        statement="We kind of doubt this prediction result ourselves, But it's basically one or another it seems: "
    elif 5<=(plotprob[0]-plotprob[1])<10:
        statement="We are on the stance of confidence right here about the prediction. In case we're wrong, we have the probabilities: "
    elif (plotprob[0]-plotprob[1])>=10:
        statement="We are pretty sure about this prediction. Here's why: "

    x=mysql.connection.cursor()
    x.execute('insert into news (headline,description,prediction) values (%s,%s,%s)',(str(request.form['head']),str(request.form['desc']),kategori))
    mysql.connection.commit()

    predictData=[kategori,statement,graph]
    return render_template('predict.htm',prediction=predictData)

@app.route('/NotFound')
def notFound():
    return render_template('error.htm')

@app.errorhandler(404)
def notFound404(error):
    return render_template('error.htm')

if __name__=='__main__':
    model=jb.load('modelComplement')
    app.run(debug=True)