from flask import Flask, render_template, request
import os
import joblib as jb

app=Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
    return render_template('home.htm')

@app.route('/prediksi', methods=['POST'])
def predict():
    head=str(request.form['head']).lower()
    desc=str(request.form['desc']).lower()
    feature=head+' '+desc
    model=jb.load('modelComplement')
    prediksi=model.predict([feature])[0]
    prediksi_l=prediksi.lower().split(' ')
    prediksi_u=[]
    for i in prediksi_l:
        prediksi_u.append(i[0].upper()+i[1:])
    kategori=' '.join(prediksi_u)
    probabilitas=model.predict_proba([feature])
    predictData=[kategori,probabilitas]
    return render_template('predict.htm',prediction=predictData)

@app.route('/NotFound')
def notFound():
    return render_template('error.htm')

@app.errorhandler(404)
def notFound404(error):
    return render_template('error.htm')

if __name__=='__main__':
    app.run(debug=True)