import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)
model = pickle.load(open('models/knn_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    tmp= pd.read_csv("tabfinal.csv")
    inputs =[e for e in request.form.values()]
    inputs = inputs[1:14]
    tmp['age'] = inputs[0]
    tmp['sex_0'] = np.where(inputs[1]== '0',  1, tmp['sex_0'])
    tmp['sex_1'] = np.where(inputs[1]== '1', 1, tmp['sex_1'])
    tmp['cp_0'] = np.where(inputs[2]=='0', 1, tmp['cp_0'])
    tmp['cp_1'] = np.where(inputs[2]=='1', 1, tmp['cp_1'])
    tmp['cp_2'] = np.where(inputs[2]=='2', 1, tmp['cp_2'])
    tmp['cp_3'] = np.where(inputs[2]=='3', 1, tmp['cp_3'])
    tmp['trestbps']= inputs[3]
    tmp['chol'] = inputs[4]
    tmp['fbs_0'] = np.where(inputs[5]=='0', 1, tmp['fbs_0'])
    tmp['fbs_1'] = np.where(inputs[5]=='1', 1, tmp['fbs_1'])
    tmp['restecg_0'] = np.where(inputs[6]== '0', 1, tmp['restecg_0'])
    tmp['restecg_1'] = np.where(inputs[6]== '1', 1, tmp['restecg_1'])
    tmp['restecg_2'] = np.where(inputs[6]== '2', 1, tmp['restecg_2'])
    tmp['thalach'] = inputs[7]
    tmp['exang_0'] = np.where(inputs[8]== '0', 1, tmp['exang_0'])
    tmp['exang_1'] = np.where(inputs[8]== '1', 1, tmp['exang_1'])
    tmp['oldpeak'] = inputs[9]
    tmp['slope_0'] = np.where(inputs[10]== '0', 1, tmp['slope_0'])
    tmp['slope_1'] = np.where(inputs[10]== '1', 1, tmp['slope_1'])
    tmp['slope_2'] = np.where(inputs[10]== '2', 1, tmp['slope_2'])
    tmp['ca']= inputs[11]
    tmp['thal_0']= np.where(inputs[12]=='0', 1, tmp['thal_0'])
    tmp['thal_1']= np.where(inputs[12]=='1', 1, tmp['thal_1'])
    tmp['thal_2']= np.where(inputs[12]=='2', 1, tmp['thal_2'])
    tmp['thal_3']= np.where(inputs[12]=='3', 1, tmp['thal_3'])
    
    #tmp en array
    tmp= np.array(tmp)
    #Normalisation tmp
    scaler_tmp = preprocessing.StandardScaler()
    scaler_tmp.fit(tmp[:,:5])
    tmp[:, :5] = scaler_tmp.transform(tmp[:, :5])
    tmp= np.delete(tmp, 9)
    tmp= np.delete(tmp,10)
    tmp= np.delete(tmp,10)
    tmp= np.delete(tmp,10)
    tmp= np.delete(tmp,12)
    tmp= np.delete(tmp,14)
    tmp= np.delete(tmp,16)
    tmp= np.delete(tmp,16)
    tmp= tmp.reshape(1,-1)
    pred_tmp= model.predict(tmp)
    if pred_tmp == 0:
        if inputs[1]== "0":
            return render_template('results.html', prediction_text= "Felicitations Madame, vous vous portez bien ")
        else:
                    return render_template('results.html', prediction_text= "Felicitations Monsieur, vous vous portez bien")
    else:
        if inputs[1]== "0":
           return render_template('results.html', prediction_text= "Attention Madame, vous presentez des risques de maladies cardiaques")
        else:
           return render_template('results.html', prediction_text= "Attention Monsieur, vous presentez des risques de maladies cardiaques")


    
    print(pred_tmp)
    print(type(pred_tmp))

    return render_template('results.html', prediction_text= pred_tmp)


if __name__ == "__main__":
    app.run(debug=True)