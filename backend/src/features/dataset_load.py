import os
import duckdb as db
import cursor as cr
con = db.connect()

# Get the directory of the current script to build relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# using the sql file to clean the dataset
sql_path = os.path.join(base_dir, "..", "..", "sql", "flights_clean.sql")
with open(sql_path, "r") as f:
    query = f.read()
db.query(query)
con.execute(query)

# saving the cleaned dataset to parquet file
con.execute("""
COPY flights_clean
TO '../../../data/flights_clean.parquet'
(FORMAT PARQUET)
""")
