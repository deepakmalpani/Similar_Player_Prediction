import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import recommend_me

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    name = request.form.get('player_name')
    #name=features[0]

    output = recommend_me(name)
    print(output)
    return render_template('index.html', prediction_text='The top 5 players similar to {} are '.format(name),players=output)



if __name__ == "__main__":
    app.run(debug=True)