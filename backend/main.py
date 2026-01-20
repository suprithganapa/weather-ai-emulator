from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import WeatherPredictor
from backend.fetch_weather import get_current_weather

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="WeatherSense AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_CITIES = ["Bangalore", "Mumbai", "Chennai", "Delhi", "Meghalaya", "Wayanad"]
VALID_HORIZONS = ["1", "3", "6", "12", "24"]

CITY_COORDS = {
    "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Meghalaya": (25.4670, 91.3662),
    "Wayanad": (11.6854, 76.1320)
}

try:
    predictor = WeatherPredictor()
    logger.info("✅ Predictor initialized")
except Exception as e:
    logger.error(f"❌ Predictor init failed: {e}")
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
    return {"message": "WeatherSense AI API", "version": "1.0.0"}

@app.get("/health")
async def health():
    if predictor is None:
        return {"status": "unhealthy"}
    return {"status": "healthy", "models_loaded": predictor.models_loaded}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        if request.city not in VALID_CITIES:
            raise HTTPException(400, f"Invalid city: {request.city}")
        if request.horizon not in VALID_HORIZONS:
            raise HTTPException(400, f"Invalid horizon: {request.horizon}")
        if predictor is None:
            raise HTTPException(503, "Predictor not initialized")
        
        horizon_hours = int(request.horizon)
        lat, lon = CITY_COORDS[request.city]
        
        try:
            current_weather = get_current_weather(lat, lon)
            logger.info(f"Weather for {request.city}: {current_weather}")
        except:
            current_weather = get_default_weather(request.city)
        
        prediction = predictor.predict(request.city, horizon_hours, current_weather)
        
        return PredictResponse(
            city=request.city,
            horizon_hours=horizon_hours,
            temperature=prediction["temperature"],
            rainfall=prediction["rainfall"],
            wind=prediction["wind"],
            events=prediction["events"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))

def get_default_weather(city: str) -> Dict:
    defaults = {
        "Bangalore": {"temperature": 24.5, "rainfall": 1.2, "wind": 2.8, "humidity": 65.0, "pressure": 1012.0},
        "Mumbai": {"temperature": 28.5, "rainfall": 2.5, "wind": 3.5, "humidity": 75.0, "pressure": 1011.0},
        "Chennai": {"temperature": 30.0, "rainfall": 1.8, "wind": 3.2, "humidity": 70.0, "pressure": 1010.0},
        "Delhi": {"temperature": 26.0, "rainfall": 0.8, "wind": 2.5, "humidity": 60.0, "pressure": 1013.0},
        "Meghalaya": {"temperature": 20.0, "rainfall": 5.5, "wind": 2.2, "humidity": 85.0, "pressure": 1008.0},
        "Wayanad": {"temperature": 23.0, "rainfall": 3.2, "wind": 2.0, "humidity": 80.0, "pressure": 1009.0}
    }
    return defaults.get(city, defaults["Bangalore"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)