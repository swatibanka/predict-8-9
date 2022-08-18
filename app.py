import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import datetime as dt
from model1 import *
# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return jsonify("Hello World")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    x_future_date = pd.date_range(start ="2022-08-01", end = "2023-01-31")
    x_future_dates = pd.DataFrame()
    x_future_dates["Dates"] = pd.to_datetime(x_future_date)
    x_future_dates.index = x_future_dates["Dates"]
    X, y = create_features(x_future_dates, label='Dates')
    y_future_total_tickets = model.predict(X)
    prediction_list = y_future_total_tickets.tolist()

    output = {'data': []}
    index = 0
    print(type(y_future_total_tickets.tolist()))

    for date in x_future_date:
        output['data'].append({'date': str(date), 'prediction': prediction_list[index]})
        index = index + 1
    return json.dumps(output)

if __name__ == "__main__":
        app.run()