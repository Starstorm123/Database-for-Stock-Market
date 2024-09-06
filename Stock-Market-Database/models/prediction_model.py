# models/prediction_model.py

# models/prediction_model.py

# models/prediction_model.py

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sqlite3
import os

def train_model(stock_symbol):
    # Connect to the SQLite database
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "../database/stock_data.db")
    conn = sqlite3.connect(db_path)

    # Retrieve data for the selected stock from the database
    query = f"SELECT date, close FROM stock_data WHERE stock_symbol='{stock_symbol}'"
    df = pd.read_sql_query(query, conn)

    # Prepare the data for Prophet
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    # Split the data into features (X) and target (y)
    X = df[['ds']].values.astype(np.float64)
    y = df['y'].values.astype(np.float64)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate multiple models
    models = [
        (Prophet(), "Prophet"),
        (XGBRegressor(objective='reg:squarederror'), "XGBoost"),
        (RandomForestRegressor(random_state=42), "Random Forest"),
        (SVR(kernel='rbf', C=1e3, gamma=0.1), "SVR"),
        (Ridge(alpha=1.0), "Ridge"),
        (Lasso(alpha=1.0), "Lasso")
        # Add more models here
    ]

    best_model = None
    best_mse = float('inf')

    for model, name in models:
        if name == "Prophet":
            model.fit(df)
            future = model.make_future_dataframe(periods=len(df))
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values[-len(X_test):]  # Use only predictions for the test data
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)

        print(f"Model: {name}, MSE: {mse}, Predictions: {y_pred}")

        if mse < best_mse:
            best_model = model
            best_mse = mse
            best_name = name

    print(f"Best model: {best_name}, MSE: {best_mse}")

    return best_model, df


# models/prediction_model.py

def predict_next_day(best_model, df, scaler=None):
    # Use the best model for prediction
    if best_model is None:
        return None

    if isinstance(best_model, Prophet):
        future = best_model.make_future_dataframe(periods=1)
        forecast = best_model.predict(future)
        prediction = forecast['yhat'].values[-1]
    else:
        # Prepare the last day's data for prediction
        last_day = df.iloc[-1]['ds']
        next_day = pd.Timestamp(last_day) + pd.DateOffset(days=1)
        next_day_data = [[next_day.dayofyear]]
        prediction = best_model.predict(next_day_data)[0]

    return prediction




