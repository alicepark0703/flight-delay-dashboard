import duckdb as db
import cursor as cr
con = db.connect()

with open(r"D:\Documents\Portfolio\flight-delay-dashboard\sql\flights_clean.sql", "r") as f:
    query = f.read()
db.query(query)
con.execute(query)

# save cleaned table
con.execute("""
COPY flights_clean
TO '../../data/flights_clean.parquet'
(FORMAT PARQUET)
""")