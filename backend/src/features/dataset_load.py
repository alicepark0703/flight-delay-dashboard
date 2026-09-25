import os
import duckdb as db

# Resolve paths from the project root so the script works from any directory
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
os.chdir(project_root)

con = db.connect()

# using the sql file to clean the dataset
sql_path = os.path.join("backend", "sql", "flights_clean.sql")
with open(sql_path, "r") as f:
    query = f.read()
con.execute(query)

# saving the cleaned dataset to parquet file
con.execute("""
COPY flights_clean
TO 'data/flights_clean.parquet'
(FORMAT PARQUET)
""")
