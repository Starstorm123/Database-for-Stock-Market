# database/database_setup.py

import os
import sqlite3

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "stock_data.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a table to store stock price data
cursor.execute('''CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_symbol TEXT PRIMARY KEY NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )''')

# Commit the changes and close the connection
conn.commit()
conn.close()
