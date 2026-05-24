from contextlib import asynccontextmanager
# module that provides utilities for working with context managers and the with statement
# when setup involved I/O bound waiting like downloading data from API, fetching records from database, or connecting to online model registry
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
#FastAPI = actual engine of web server + HTTPException = stop execution and send proper readable errors (400, 404, etc.)
from fastapi.middleware.cors import CORSMiddleware
#Cross-Origin Resource Sharing = config tool that allows you to safely whitelist specific websites and let them communicate with your API
from pydantic import BaseModel, Field
#pydantic = data validation & settings management lib
#BaseModel: (basic data validation) create your own data blueprints by inheriting from BaserModel and using python type hints to define what data should look like
#Field: lets you add advanced, fine-grained constraints, default values, and metadata to attributes inside your BaseModel
from loguru import logger

from backend.src.features.build_features import build_features

MODEL_PATH = Path("backend/models/lgbm_delay.pkl")
ENCODERS_PATH = Path("backend/models/encoders.pkl")
DATA_PATH = Path("data/flights_clean.parquet")


state:dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts... ")
    state["model"] = joblib.load(MODEL_PATH) #joblib = aka save game button
    state["encoders"] = joblib.load(ENCODERS_PATH)
    logger.info("Loading analytics cache...")
    yield
    state.clear() #reset button for dictionaries

app = FastAPI(title = "Flight Delay API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], #vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

#data blueprint (schema) for ml model -> type locking and self-documenting code
class FlightInput(BaseModel):
    airline: str = Field(examples=["AA"])
    origin: str = Field(examples=["JFK"])
    dest: str = Field(examples=["LAX"])
    crs_dep_time: int = Field(examples=[1435], description="Scheduled departure HHMM")
    flight_date: str = Field(examples=["2024-07-15"])
    distance: float = Field(examples=[2475.0])

class PredictionResponse(BaseModel):
    predicted_delay_minutes: float
    is_likely_delayed: bool   #threshold >= 15min


#prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightInput):
    #translating internet data into pandas dataframe (aka adpater)
    row = pd.DataFrame([{
        "Reporting_Airline": flight.airline,
        "Origin": flight.origin,
        "Dest": flight.dest,
        "CRSDepTime": flight.crs_dep_time,
        "FlightDate": flight.flight_date,
        "Distance": flight.distance,
    }])

    #prediction machine -> go through target encoders, then pulls the model out of global state memory and makes calcs
    try:
        X, _ = build_features(row, encoders=state["encoders"])
        pred = float(state["model"].predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    #calc done, need to format the answer properly before sending back over the internet
    return PredictionResponse(
        predicted_delay_minutes=round(pred, 1),
        is_likely_delayed=pred >= 15,
    )

#analytics
@app.get("/analytics/airlines")
def airlines_analytics():
    df=state["df"]
    return(
        df.groupby("Reporting_Airline")["DepDelay"]
        .agg(mean_delay="mean", flight_count="count") #for every airline, calculates avg delay and total number of flights
        .round(2).reset_index().sort_values("mean_delay") #cleans up numbers to 2 decimal places, ranks them from the most on-time to the least
        .to_dict(orient="records") #turns pandas table into a list of JSON objects
    )

@app.get("/analytics/hourly")
def hourly_analytics():
    df=state["df"].copy()
    df["dep_hour"]=(df["CRSDepTime"] // 100).clip(0, 23) 
    return(
        df.groupby("dep_hour")["DepDelay"].mean().round(2).reset_index()
        .rename(columns={"dep_hour": "hour", "DepDelay": "mean_delay"})
        .to_dict(orient="records") #converting table into list of individual rows, where each row is its own self-contained dictionary
    ) #without orienting as records, pandas groups data by column, so it looks like nested dictionary. with the records, pandas uses 
      #structural orientation of a flat list where each item represents a complete record (row) from table
@app.get("/analytics/heatmap")
def heatmap_analytics():
    df=state["df"].copy()
    df["FlightDate"]=pd.to_datetime(df["FlightDate"])
    df["month"]=df["FlightDate"].dt.month
    df["day_of_week"]=df["FlightDate"].dt.dayofweek
    return(
        df.groupby(["month", "day_of_week"])["DepDelay"]
        .mean().round(2).reset_index()
        .to_dict(orient="records")
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}