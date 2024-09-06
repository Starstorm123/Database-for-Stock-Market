# clear_database.py

import sqlite3
import os

# Connect to the SQLite database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "database/stock_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Clear all tuples from the stock_data table
cursor.execute("DELETE FROM stock_data")

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database cleared successfully.")
