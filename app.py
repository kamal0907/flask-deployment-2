from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(_name_)
CORS(app)

@app.route('/')
def welcome():
    return "Welcome to Crop Recommendation API!"

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        input_data = np.array([[
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
        ]])

        prediction = RF.predict(input_data)
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e), "message": "Error occurred during prediction!"})

@app.route('/last_prediction', methods=['GET'])
def get_last_prediction():
    # For demonstration, this will simply return the last prediction made
    # In a real-world scenario, you might want to use a database to store and retrieve this
    if prediction:
        return jsonify({"prediction": prediction[0]})
    else:
        return jsonify({"message": "No predictions made yet!"})

if _name_ == "_main_":
    prediction = None
    df = pd.read_csv("F:/Projects/crop-recommendation-system-based-on-machine-learning-using-python-master/Data/crop_recommendation.csv")
    features = df[['N','P','K','temperature','humidity']]
    target = df['label']
    RF = RandomForestClassifier(n_estimators=29, criterion = 'entropy',random_state=0)
    RF.fit(features, target)

    app.run(debug=True,host='127.0.0.1', port=5001)