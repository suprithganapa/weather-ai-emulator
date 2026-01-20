from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.inference_v2 import ImprovedWeatherPredictor as WeatherPredictor
from backend.fetch_weather import get_current_weather

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="WeatherSense AI", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

VALID_CITIES = ["Bangalore", "Mumbai", "Chennai", "Delhi", "Shillong", "Wayanad", "Jaipur", "Dharali", "Ladakh"]
VALID_HORIZONS = ["1", "3", "6", "12", "24"]
CITY_COORDS = {"Bangalore": (12.9716, 77.5946), "Mumbai": (19.0760, 72.8777), "Chennai": (13.0827, 80.2707), "Delhi": (28.7041, 77.1025), "Shillong": (25.5788, 91.8933), "Wayanad": (11.6854, 76.1320), "Jaipur": (26.9124, 75.7873), "Dharali": (31.0192, 78.5685), "Ladakh": (34.1526, 77.5771)}

try:
    predictor = WeatherPredictor()
    logger.info("✅ Ready")
except Exception as e:
    logger.error(f"Init failed: {e}")
    predictor = None

class PredictRequest(BaseModel):
    city: str
    horizon: str

class PredictResponse(BaseModel):
    city: str
    horizon_hours: int
    temperature: float
    rainfall: float
    wind: float
    events: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "WeatherSense AI v2.0", "cities": len(VALID_CITIES)}

@app.get("/health")
async def health():
    if predictor is None:
        return {"status": "unhealthy"}
    return {"status": "healthy", "models_loaded": predictor.models_loaded, "cities": VALID_CITIES}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        if request.city not in VALID_CITIES:
            raise HTTPException(400, f"Invalid city")
        if request.horizon not in VALID_HORIZONS:
            raise HTTPException(400, f"Invalid horizon")
        if predictor is None:
            raise HTTPException(503, "Not ready")
        horizon_hours = int(request.horizon)
        lat, lon = CITY_COORDS[request.city]
        try:
            current_weather = get_current_weather(lat, lon)
        except:
            defaults = {"Bangalore": {"temperature": 24.0, "rainfall": 1.2, "wind": 2.5, "humidity": 65.0, "pressure": 1012.0}, "Mumbai": {"temperature": 28.0, "rainfall": 2.5, "wind": 3.5, "humidity": 75.0, "pressure": 1011.0}, "Chennai": {"temperature": 30.0, "rainfall": 1.5, "wind": 3.0, "humidity": 70.0, "pressure": 1010.0}, "Delhi": {"temperature": 25.0, "rainfall": 0.8, "wind": 2.2, "humidity": 55.0, "pressure": 1013.0}, "Shillong": {"temperature": 18.0, "rainfall": 8.0, "wind": 2.0, "humidity": 85.0, "pressure": 1008.0}, "Wayanad": {"temperature": 22.0, "rainfall": 3.5, "wind": 1.8, "humidity": 78.0, "pressure": 1009.0}, "Jaipur": {"temperature": 28.0, "rainfall": 0.3, "wind": 3.0, "humidity": 45.0, "pressure": 1014.0}, "Dharali": {"temperature": 15.0, "rainfall": 4.0, "wind": 2.5, "humidity": 70.0, "pressure": 1007.0}, "Ladakh": {"temperature": 8.0, "rainfall": 0.1, "wind": 4.0, "humidity": 35.0, "pressure": 863.0}}
            current_weather = defaults.get(request.city, defaults["Bangalore"])
        prediction = predictor.predict(request.city, horizon_hours, current_weather)
        return PredictResponse(city=request.city, horizon_hours=horizon_hours, temperature=prediction["temperature"], rainfall=prediction["rainfall"], wind=prediction["wind"], events=prediction["events"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))