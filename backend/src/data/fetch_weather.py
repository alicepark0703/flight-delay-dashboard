"""
Fetches hourly weather from NASA POWER for every unique Origin airport
in the cleaned flight parquet and saves a joined weather cache.

Run once before training:
    python -m backend.src.data.fetch_weather
"""

import time
from pathlib import Path

import airportsdata
import pandas as pd
import requests
from loguru import logger

FLIGHTS_PATH = Path("data/flights_clean.parquet")
WEATHER_OUT = Path("data/weather_cache.parquet")
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# NASA POWER parameters: wind speed (m/s), precipitation (mm/hr), temperature (°C)
POWER_PARAMS = "WS10M,PRECTOTCORR,T2M"


def get_airport_coords() -> dict[str, tuple[float, float]]:
    airports = airportsdata.load("IATA")
    return {
        code: (info["lat"], info["lon"])
        for code, info in airports.items()
        if info.get("lat") and info.get("lon")
    }


def _derive_severity(wspd_kmh: float, prcp: float, snow: float) -> int:
    """Derive 0–3 severity from wind speed (km/h), precipitation, and snow (mm/hr)."""
    if wspd_kmh > 70 or prcp > 10 or snow > 5:
        return 3
    if wspd_kmh > 40 or prcp > 2.5 or snow > 1:
        return 2
    if wspd_kmh > 20 or prcp > 0 or snow > 0:
        return 1
    return 0


def fetch_hourly_for_airport(
    iata: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch NASA POWER hourly weather and return a tidy DataFrame."""
    params = {
        "parameters": POWER_PARAMS,
        "community": "SB",
        "latitude": lat,
        "longitude": lon,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }
    resp = requests.get(NASA_POWER_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    raw = payload.get("properties", {}).get("parameter", {})
    ws10m = raw.get("WS10M", {})
    prcp_raw = raw.get("PRECTOTCORR", {})
    t2m = raw.get("T2M", {})

    if not ws10m:
        return pd.DataFrame()

    records = []
    for key, wspd_ms in ws10m.items():
        # key format: YYYYMMDDHH
        year, month, day, hour = int(key[:4]), int(key[4:6]), int(key[6:8]), int(key[8:10])
        flight_date = f"{year:04d}-{month:02d}-{day:02d}"
        wspd_kmh = (wspd_ms or 0.0) * 3.6
        prcp = max(0.0, prcp_raw.get(key) or 0.0)
        temp = t2m.get(key) or 0.0
        snow = prcp if temp < 2.0 else 0.0
        severity = _derive_severity(wspd_kmh, prcp, snow)
        records.append({
            "Origin": iata,
            "FlightDate": flight_date,
            "dep_hour": hour,
            "wspd": round(wspd_kmh, 2),
            "prcp": round(prcp, 3),
            "snow": round(snow, 3),
            "temp": round(temp, 1),
            "weather_severity": severity,
        })

    return pd.DataFrame(records)


def main():
    logger.info("Loading flight data...")
    df = pd.read_parquet(FLIGHTS_PATH, columns=["Origin", "FlightDate", "CRSDepTime"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"]).dt.strftime("%Y-%m-%d")

    origins = df["Origin"].unique().tolist()
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    logger.info(f"Fetching weather for {len(origins)} origin airports across 2024...")
    coords = get_airport_coords()

    missing = [o for o in origins if o not in coords]
    if missing:
        logger.warning(f"No coords found for: {missing}")

    chunks = []
    eligible = [iata for iata in origins if iata in coords]
    total = len(eligible)
    t0 = time.monotonic()
    for done, iata in enumerate(eligible, 1):
        lat, lon = coords[iata]
        try:
            chunk = fetch_hourly_for_airport(iata, lat, lon, start_date, end_date)
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Failed {iata}: {e}")
        elapsed = time.monotonic() - t0
        rate = done / elapsed
        eta_s = (total - done) / rate if rate > 0 else 0
        eta_min = int(eta_s // 60)
        eta_sec = int(eta_s % 60)
        logger.info(f"  {done}/{total} ({iata}) — ETA {eta_min}m {eta_sec}s")
        time.sleep(2)  # NASA POWER rate limit: 30 req/min

    if not chunks:
        logger.error("No weather data fetched — check NASA POWER connectivity.")
        return

    weather = pd.concat(chunks, ignore_index=True)
    weather["weather_severity"] = weather["weather_severity"].astype("int8")

    weather.to_parquet(WEATHER_OUT, index=False)
    logger.info(f"Saved weather cache to {WEATHER_OUT} ({len(weather):,} rows)")


if __name__ == "__main__":
    main()
