import duckdb as db
import cursor as cr
con = db.connect()


# using the sql file to clean the dataset
with open(r"D:\Documents\Portfolio\flight-delay-dashboard\sql\flights_clean.sql", "r") as f:
    query = f.read()
db.query(query)
con.execute(query)

# saving the cleaned dataset to parquet file
con.execute("""
COPY flights_clean
TO '../../data/flights_clean.parquet'
(FORMAT PARQUET)
""")
