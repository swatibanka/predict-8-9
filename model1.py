
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import json
from flask import Flask, request, jsonify
from datetime import date
import datetime as dt
import pickle



df = pd.read_excel("MG Data for insight.xlsx", sheet_name = 'Sheet4')
df["Posting Date"] = pd.to_datetime(df["Posting Date"])
dt_ticket_count = df.groupby(["Posting Date"]).agg({"Status" : "count"})
#Description = df.groupby('Posting Date')['Description'].value_counts().unstack().fillna(0).astype(int)
#Zone = df.groupby('Posting Date')['Zone'].value_counts().unstack().fillna(0).astype(int)
#TransactionType = df.groupby('Posting Date')['TRANSACTIONTYPE'].value_counts().unstack().fillna(0).astype(int)
#Refined_data = pd.concat([Description, dt_ticket_count], axis = 1, join = "inner")
#Distributed_data = Refined_data.reset_index()
d_ticket_count = dt_ticket_count.drop(dt_ticket_count.index[4::7])
_ticket_count = d_ticket_count.reset_index(inplace = False)


def create_features(df, label=None):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df['date']
    X = df[['hour','dayofweek','quarter','month','year',
       'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
X, y = create_features(dt_ticket_count, label='Status')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20)

#print(y)


reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False)

# Make pickle file of our model
pickle.dump(reg, open("model.pkl", "wb"))

# y_pred = pickle.load(open("model.pkl","rb"))
# yy_pred = y_pred.predict(X_test)
# print(yy_pred)

# print(X_test)



# def predict():
#     x_future_date = pd.date_range(start ="2022-08-01", end = "2023-01-31")
#     x_future_dates = pd.DataFrame()
#     x_future_dates["Dates"] = pd.to_datetime(x_future_date)
#     x_future_dates.index = x_future_dates["Dates"]
#     X, y = create_features(x_future_dates, label='Dates')
#     y_pred = pickle.load(open("model.pkl","rb"))
#     y_future_total_tickets = y_pred.predict(X)
#     return (y_future_total_tickets)

# output = predict()
# print(output)