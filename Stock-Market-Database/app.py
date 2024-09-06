# app.py

from flask import Flask, render_template, request
from models.prediction_model import train_model, predict_next_day
import os
import sqlite3
import pandas as pd

app = Flask(__name__)

# Function to create a new SQLite connection
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "database/stock_data.db")
    return sqlite3.connect(db_path)

# Load data from the database
@app.route('/')
def home():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT stock_symbol FROM stock_data")
        stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
        return render_template('index.html', stocks=stocks)
    except Exception as e:
        app.logger.error(f"Error loading data: {e}")
        return "An error occurred while loading data."

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Get the selected stock symbol from the form
        stock_symbol = request.form['stock']

        # Train the prediction model for the selected stock
        model, data = train_model(stock_symbol)

        # Make prediction for the next day using the data
        prediction = predict_next_day(model, data)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS predicted_prices (stock_symbol TEXT, predicted_price REAL)")
        cursor.execute("INSERT INTO predicted_prices (stock_symbol, predicted_price) VALUES (?, ?)", (stock_symbol, prediction))
        conn.commit()
        conn.close()

        return render_template('prediction.html', prediction=prediction, stock=stock_symbol)
    except Exception as e:
        app.logger.error(f"Error predicting stock price: {e}")
        return "An error occurred while predicting stock price."

if __name__ == '__main__':
    app.run(debug=True)
