"""
Weather AI Emulator - Weather Data Fetcher
Fetches current weather data for inference
"""

import requests
from typing import Dict, List
from datetime import datetime, timedelta


# City coordinates (for API requests)
CITY_COORDS = {
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Meghalaya': {'lat': 25.4670, 'lon': 91.3662},
    'Wayanad': {'lat': 11.6054, 'lon': 76.0830},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025},
}


def fetch_current_weather(city: str) -> List[Dict]:
    """
    Fetch current weather data for a city.
    Returns last 6 hours of data for inference.
    
    Args:
        city: City name
    
    Returns:
        List of weather observations (last 6 hours)
        Each observation has: rain, temp, wind, humidity, pressure
    """
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city: {city}")
    
    coords = CITY_COORDS[city]
    lat, lon = coords['lat'], coords['lon']
    
    # Try to fetch from Open-Meteo API (free, no API key needed)
    try:
        # Get historical hourly data for past 6 hours
        url = "https://api.open-meteo.com/v1/forecast"
        
        # Calculate time range (last 6 hours)
        now = datetime.utcnow()
        start_time = now - timedelta(hours=6)
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,precipitation,windspeed_10m,relativehumidity_2m,surface_pressure',
            'timezone': 'auto',
            'past_hours': 6,
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract hourly data
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        temps = hourly.get('temperature_2m', [])
        precip = hourly.get('precipitation', [])
        winds = hourly.get('windspeed_10m', [])
        humidity = hourly.get('relativehumidity_2m', [])
        pressure = hourly.get('surface_pressure', [])
        
        # Build weather sequence (last 6 observations)
        weather_sequence = []
        for i in range(-6, 0):  # Last 6 hours
            if i < -len(times):
                continue
            
            weather_sequence.append({
                'rain': precip[i] if precip[i] is not None else 0.0,
                'temp': temps[i] if temps[i] is not None else 25.0,
                'wind': winds[i] if winds[i] is not None else 3.0,
                'humidity': humidity[i] if humidity[i] is not None else 60.0,
                'pressure': pressure[i] if pressure[i] is not None else 1013.0,
            })
        
        # If we don't have enough data, use fallback
        if len(weather_sequence) < 6:
            print(f"Warning: Only got {len(weather_sequence)} hours of data, using fallback")
            return get_fallback_weather(city)
        
        return weather_sequence
    
    except Exception as e:
        print(f"Warning: Failed to fetch weather from API: {e}")
        print("Using fallback weather data...")
        return get_fallback_weather(city)


def get_fallback_weather(city: str) -> List[Dict]:
    """
    Return fallback weather data when API is unavailable.
    Uses typical values for each city.
    
    Args:
        city: City name
    
    Returns:
        List of 6 hourly observations
    """
    # Typical weather patterns for each city
    typical_weather = {
        'Bangalore': {'temp': 25.0, 'rain': 0.5, 'wind': 3.0, 'humidity': 65.0, 'pressure': 1013.0},
        'Mumbai': {'temp': 28.0, 'rain': 1.0, 'wind': 4.5, 'humidity': 75.0, 'pressure': 1012.0},
        'Meghalaya': {'temp': 20.0, 'rain': 3.0, 'wind': 2.5, 'humidity': 80.0, 'pressure': 1010.0},
        'Wayanad': {'temp': 23.0, 'rain': 2.0, 'wind': 2.0, 'humidity': 70.0, 'pressure': 1011.0},
        'Chennai': {'temp': 30.0, 'rain': 0.8, 'wind': 5.0, 'humidity': 70.0, 'pressure': 1012.0},
        'Delhi': {'temp': 27.0, 'rain': 0.3, 'wind': 3.5, 'humidity': 55.0, 'pressure': 1013.0},
    }
    
    base_weather = typical_weather.get(city, typical_weather['Bangalore'])
    
    # Create 6 hours of slightly varying data
    weather_sequence = []
    for i in range(6):
        # Add small random variation
        import random
        weather_sequence.append({
            'rain': base_weather['rain'] + random.uniform(-0.2, 0.2),
            'temp': base_weather['temp'] + random.uniform(-1.0, 1.0),
            'wind': base_weather['wind'] + random.uniform(-0.5, 0.5),
            'humidity': base_weather['humidity'] + random.uniform(-3.0, 3.0),
            'pressure': base_weather['pressure'] + random.uniform(-1.0, 1.0),
        })
    
    return weather_sequence


# Test function
if __name__ == '__main__':
    print("Testing weather fetcher...")
    
    for city in CITY_COORDS.keys():
        print(f"\n{city}:")
        try:
            weather = fetch_current_weather(city)
            print(f"  Got {len(weather)} hours of data")
            print(f"  Latest: Temp={weather[-1]['temp']:.1f}°C, "
                  f"Rain={weather[-1]['rain']:.1f}mm, "
                  f"Wind={weather[-1]['wind']:.1f}m/s")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n✅ Weather fetcher test complete")
