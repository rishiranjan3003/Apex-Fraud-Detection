from flask import Flask,request,jsonify,render_template, url_for
from flask import request
import pickle 
import pandas as pd
import numpy as np
app = Flask(__name__)

model = pickle.load(open("credit.pkl","rb"))

@app.route('/')
def hello_world():
    return render_template('homepage.html')

@app.route('/index1.html',methods=['POST','GET'])
def hello_world1():
    return render_template("index1.html")

@app.route('/predict',methods=['POST','GET'])
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
    
   
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    print(row_df)
    prediction = model.predict(row_df)

    output='{0:.{1}f}'.format(prediction[0][1], 2)
    
    index_target=pd.Series(["Normal Transaction", "Fraud Transaction"])
    result=index_target[output]
    #result=list(result.values)
    #result=str(result)
    
    output = str(result)
    if output>str(0.9):
        return render_template('result1.html',pred=f'You are safe.\nProbability of fraud transaction is {output}'.format(output))
    else:
        return render_template('result1.html',pred=f'You are not safe.\nProbability of fraud transaction is {output}'.format(ouput))

if __name__ == "__main__":
    app.run(debug=False)