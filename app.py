from flask import Flask, request, jsonify, render_template
from pycaret.classification import *
import pandas as pd

app = Flask(__name__)
model = load_model('models/model_rf')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    tmp= pd.read_csv("test.csv")
    inputs =[e for e in request.form.values()]
    inputs = inputs[1:14]
    tmp['age'] = inputs[0]
    tmp['sex'] = inputs[1]
    tmp['chest pain type'] = inputs[2]
    tmp['resting bp s']= inputs[3]
    tmp['cholesterol'] = inputs[4]
    tmp['fasting blood sugar'] = inputs[5]
    tmp['resting ecg'] = inputs[6]
    tmp['max heart rate'] = inputs[7]
    tmp['exercise angina'] = inputs[8]
    tmp['oldpeak'] = inputs[9]
    tmp['ST slope'] = inputs[10]

    print(tmp)


    pred_tmp= predict_model(model, data= tmp)
    print(pred_tmp['Score'])
    print(pred_tmp)
    if (pred_tmp['Label'] == 0).any():
        if inputs[1]== "0":
            return render_template('results.html', prediction_text= "Vous etes saine Madame"+ " " + str(pred_tmp[['Score']]))
        else:
                    return render_template('results.html', prediction_text= "Vous etes sain Monsieur" +  " " +str(pred_tmp[['Score']]))
    else:
        if inputs[1]== "0":
           return render_template('results.html', prediction_text= "Vous etes malade Madame" + " " + str( pred_tmp[['Score']]))
        else:
           return render_template('results.html', prediction_text= "Vous etes malade Monsieur"+ " " + str( pred_tmp[['Score']]))  
    return render_template('results.html', prediction_text= pred_tmp)

if __name__ == "__main__":
    app.run(debug=True)