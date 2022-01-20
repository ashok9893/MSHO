import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from transformers import pipeline
import pandas as pd
import torch

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    table = pd.read_csv("data.csv")
    table = table.astype(str)
    query = [x for x in request.form.values()]
    answer = tqa(table=table, query=query)
    for ans in answer:
        prediction= ans["answer"]
    
    return render_template('index.html', prediction_text='Your diabatic test result is --  {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
