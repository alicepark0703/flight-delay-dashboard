from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Flight Delay Prediction API"}
