"""
Weather AI Emulator - FastAPI Backend
Main API server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from .inference import WeatherPredictor
from .fetch_weather import fetch_current_weather

# Create FastAPI app
app = FastAPI(
    title="Weather AI Emulator API",
    description="AI-powered multi-horizon weather prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (lazy loading)
predictor = None

def get_predictor():
    """Lazy load predictor."""
    global predictor
    if predictor is None:
        predictor = WeatherPredictor()
    return predictor


# Request/Response models
class PredictRequest(BaseModel):
    city: str
    horizon: str  # "1", "3", "6", "12", "24" (or "1h", "3h", etc.)
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Bangalore",
                "horizon": "1"
            }
        }


class PredictResponse(BaseModel):
    city: str
    horizon_hours: int
    temperature: float
    rainfall: float
    wind: float
    events: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Bangalore",
                "horizon_hours": 1,
                "temperature": 25.3,
                "rainfall": 2.1,
                "wind": 3.2,
                "events": {
                    "cloudburst": 0.05,
                    "thunderstorm": 0.15,
                    "heatwave": 0.02,
                    "coldwave": 0.01,
                    "cyclone_like": 0.03,
                    "heavy_rain": 0.12,
                    "high_wind": 0.08,
                    "fog": 0.04,
                    "drought": 0.01,
                    "humidity_extreme": 0.06
                }
            }
        }


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Weather AI Emulator API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict future weather for a given city and time horizon.
    
    Args:
        request: PredictRequest with city and horizon
    
    Returns:
        PredictResponse with weather predictions and event probabilities
    """
    # Validate city
    valid_cities = ['Bangalore', 'Mumbai', 'Meghalaya', 'Wayanad', 'Chennai', 'Delhi']
    if request.city not in valid_cities:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid city. Must be one of: {valid_cities}"
        )
    
    # Parse horizon (remove 'h' if present)
    horizon_str = request.horizon.replace('h', '').replace('H', '')
    try:
        horizon = int(horizon_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be a number (1, 3, 6, 12, or 24)"
        )
    
    # Validate horizon
    valid_horizons = [1, 3, 6, 12, 24]
    if horizon not in valid_horizons:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon. Must be one of: {valid_horizons}"
        )
    
    try:
        # Get predictor
        pred = get_predictor()
        
        # Fetch current weather data
        current_weather = fetch_current_weather(request.city)
        
        # Make prediction
        result = pred.predict(current_weather, horizon)
        
        # Format response
        return PredictResponse(
            city=request.city,
            horizon_hours=horizon,
            temperature=round(result['temperature'], 2),
            rainfall=round(result['rainfall'], 2),
            wind=round(result['wind'], 2),
            events={
                'cloudburst': round(result['events']['cloudburst'], 3),
                'thunderstorm': round(result['events']['thunderstorm'], 3),
                'heatwave': round(result['events']['heatwave'], 3),
                'coldwave': round(result['events']['coldwave'], 3),
                'cyclone_like': round(result['events']['cyclone_like'], 3),
                'heavy_rain': round(result['events']['heavy_rain'], 3),
                'high_wind': round(result['events']['high_wind'], 3),
                'fog': round(result['events']['fog'], 3),
                'drought': round(result['events']['drought'], 3),
                'humidity_extreme': round(result['events']['humidity_extreme'], 3),
            }
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model not found: {str(e)}. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
