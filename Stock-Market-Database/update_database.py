import yfinance as yf
import sqlite3
import os

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "database/stock_data.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if the stock_data table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'")
table_exists = cursor.fetchone()

if table_exists:
    # Define the stocks you want to fetch data for
    stocks = ['AAPL', 'MSFT', 'GOOGL']

    for stock in stocks:
        # Fetch historical data from yfinance
        data = yf.download(stock, start='2021-01-01', end='2024-03-15')

        # Iterate over each row in the data and insert it into the database
        for index, row in data.iterrows():
            cursor.execute("INSERT INTO stock_data (stock_symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (stock, index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

    # Commit the changes
    conn.commit()
else:
    print("Error: stock_data table does not exist.")

# Close the connection
conn.close()
