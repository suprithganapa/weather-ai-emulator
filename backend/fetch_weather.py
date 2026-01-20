import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def get_current_weather(lat: float, lon: float) -> Dict:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure",
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        current = data.get("current", {})
        
        return {
            "temperature": current.get("temperature_2m", 25.0),
            "rainfall": current.get("precipitation", 0.0),
            "wind": current.get("wind_speed_10m", 3.0),
            "humidity": current.get("relative_humidity_2m", 70.0),
            "pressure": current.get("surface_pressure", 1013.0)
        }
    except Exception as e:
        logger.warning(f"Weather API failed: {e}")
        return {"temperature": 25.0, "rainfall": 0.0, "wind": 3.0, "humidity": 70.0, "pressure": 1013.0}