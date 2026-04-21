CREATE OR REPLACE TABLE flights_clean AS

SELECT 
    Year, Quarter, Month, FlightDate,
    Reporting_Airline,
    Origin, Dest, 
    CRSDepTime, CRSArrTime, -- scheduled departure, arrival
    DepDelay, -- target (avoid other features as they happen after the delay occurs)
    Cancelled, CancellationCode, Diverted, -- filter out cancelled flights
    Distance
FROM 'D:\Documents\Portfolio\flight-delay-dashboard\data\us_flights_2024_all.parquet'
WHERE Cancelled = 0
AND Diverted = 0
AND DepDelay IS NOT NULL
AND Origin IS NOT NULL
AND Dest IS NOT NULL
AND CRSDepTime IS NOT NULL
AND CRSArrTime IS NOT NULL
AND Distance IS NOT NULL
