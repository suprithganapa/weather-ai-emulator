import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# City-specific realistic weather patterns
CITY_PATTERNS = {
    'Bangalore': {
        'temp_base': 24.0, 'temp_amplitude': 5.0, 'temp_std': 1.5,
        'rain_base': 1.2, 'rain_amplitude': 3.5, 'rain_std': 2.0,
        'wind_base': 2.5, 'wind_std': 0.8,
        'humidity_base': 65.0, 'humidity_std': 8.0,
        'monsoon_months': [6, 7, 8, 9], 'monsoon_factor': 4.0
    },
    'Mumbai': {
        'temp_base': 28.0, 'temp_amplitude': 4.0, 'temp_std': 1.2,
        'rain_base': 2.5, 'rain_amplitude': 8.0, 'rain_std': 5.0,
        'wind_base': 3.5, 'wind_std': 1.2,
        'humidity_base': 75.0, 'humidity_std': 10.0,
        'monsoon_months': [6, 7, 8, 9], 'monsoon_factor': 8.0
    },
    'Chennai': {
        'temp_base': 30.0, 'temp_amplitude': 5.0, 'temp_std': 1.8,
        'rain_base': 1.5, 'rain_amplitude': 4.0, 'rain_std': 3.0,
        'wind_base': 3.0, 'wind_std': 1.0,
        'humidity_base': 70.0, 'humidity_std': 12.0,
        'monsoon_months': [10, 11, 12], 'monsoon_factor': 5.0
    },
    'Delhi': {
        'temp_base': 25.0, 'temp_amplitude': 12.0, 'temp_std': 3.0,
        'rain_base': 0.8, 'rain_amplitude': 3.0, 'rain_std': 2.5,
        'wind_base': 2.2, 'wind_std': 0.9,
        'humidity_base': 55.0, 'humidity_std': 15.0,
        'monsoon_months': [7, 8], 'monsoon_factor': 6.0
    },
    'Shillong': {
        'temp_base': 18.0, 'temp_amplitude': 6.0, 'temp_std': 2.0,
        'rain_base': 8.0, 'rain_amplitude': 12.0, 'rain_std': 8.0,
        'wind_base': 2.0, 'wind_std': 0.7,
        'humidity_base': 85.0, 'humidity_std': 8.0,
        'monsoon_months': [5, 6, 7, 8, 9], 'monsoon_factor': 15.0
    },
    'Wayanad': {
        'temp_base': 22.0, 'temp_amplitude': 4.0, 'temp_std': 1.5,
        'rain_base': 3.5, 'rain_amplitude': 6.0, 'rain_std': 4.0,
        'wind_base': 1.8, 'wind_std': 0.6,
        'humidity_base': 78.0, 'humidity_std': 10.0,
        'monsoon_months': [6, 7, 8], 'monsoon_factor': 7.0
    },
    'Jaipur': {
        'temp_base': 28.0, 'temp_amplitude': 14.0, 'temp_std': 3.5,
        'rain_base': 0.3, 'rain_amplitude': 1.5, 'rain_std': 1.0,
        'wind_base': 3.0, 'wind_std': 1.5,
        'humidity_base': 45.0, 'humidity_std': 18.0,
        'monsoon_months': [7, 8], 'monsoon_factor': 3.0
    },
    'Dharali': {
        'temp_base': 15.0, 'temp_amplitude': 8.0, 'temp_std': 2.5,
        'rain_base': 4.0, 'rain_amplitude': 10.0, 'rain_std': 7.0,
        'wind_base': 2.5, 'wind_std': 1.0,
        'humidity_base': 70.0, 'humidity_std': 12.0,
        'monsoon_months': [6, 7, 8, 9], 'monsoon_factor': 12.0,
        'cloudburst_prone': True
    }
}

def generate_realistic_weather(city, start_date, hours=8760):
    """Generate realistic hourly weather data for a city"""
    pattern = CITY_PATTERNS[city]
    data = []
    
    current_date = start_date
    
    for hour in range(hours):
        # Time-based patterns
        day_of_year = current_date.timetuple().tm_yday
        hour_of_day = current_date.hour
        month = current_date.month
        
        # Temperature with daily and seasonal cycles
        seasonal_temp = pattern['temp_base'] + pattern['temp_amplitude'] * np.sin(2 * np.pi * day_of_year / 365)
        daily_temp_var = 3.0 * np.sin(2 * np.pi * hour_of_day / 24)
        temp_noise = np.random.randn() * pattern['temp_std']
        temperature = seasonal_temp + daily_temp_var + temp_noise
        
        # Rainfall with monsoon effect
        is_monsoon = month in pattern['monsoon_months']
        monsoon_factor = pattern['monsoon_factor'] if is_monsoon else 1.0
        
        base_rain = pattern['rain_base'] * monsoon_factor
        rain_noise = np.abs(np.random.randn() * pattern['rain_std'] * monsoon_factor)
        
        # Occasional heavy rain events
        if np.random.random() < 0.05 and is_monsoon:
            rain_noise += np.random.exponential(10.0)
        
        # Cloudburst events for Dharali and Shillong
        if city in ['Dharali', 'Shillong'] and is_monsoon and np.random.random() < 0.02:
            rain_noise += np.random.exponential(30.0)
        
        rainfall = max(0, base_rain + rain_noise)
        
        # Wind speed
        wind = max(0, pattern['wind_base'] + np.random.randn() * pattern['wind_std'])
        if rainfall > 10:  # High wind during heavy rain
            wind += np.random.exponential(2.0)
        
        # Humidity
        humidity_base = pattern['humidity_base']
        if rainfall > 5:
            humidity_base = min(95, humidity_base + 15)
        humidity = np.clip(humidity_base + np.random.randn() * pattern['humidity_std'], 20, 100)
        
        # Pressure
        pressure = 1013.0 + np.random.randn() * 3.0
        if rainfall > 15:  # Low pressure during storms
            pressure -= np.random.uniform(5, 15)
        
        # Extreme events
        events = {
            'cloudburst': 1 if rainfall > 50 else 0,
            'thunderstorm': 1 if (rainfall > 15 and wind > 8) else 0,
            'heatwave': 1 if temperature > 40 else 0,
            'coldwave': 1 if temperature < 5 else 0,
            'cyclone_like': 1 if (wind > 15 and rainfall > 20) else 0,
            'heavy_rain': 1 if rainfall > 25 else 0,
            'high_wind': 1 if wind > 12 else 0,
            'fog': 1 if (humidity > 90 and temperature < 15) else 0,
            'drought': 1 if (rainfall < 0.1 and not is_monsoon) else 0,
            'humidity_extreme': 1 if (humidity > 90 or humidity < 30) else 0
        }
        
        data.append({
            'timestamp': current_date,
            'city': city,
            'temperature': temperature,
            'rainfall': rainfall,
            'wind': wind,
            'humidity': humidity,
            'pressure': pressure,
            **events
        })
        
        current_date += timedelta(hours=1)
    
    return pd.DataFrame(data)

def generate_all_cities_data():
    """Generate data for all cities"""
    start_date = datetime(2023, 1, 1)
    all_data = []
    
    for city in CITY_PATTERNS.keys():
        print(f"Generating data for {city}...")
        city_data = generate_realistic_weather(city, start_date, hours=8760)
        all_data.append(city_data)
        print(f"  Generated {len(city_data)} samples")
        print(f"  Temp range: {city_data['temperature'].min():.1f} to {city_data['temperature'].max():.1f}°C")
        print(f"  Rain range: {city_data['rainfall'].min():.1f} to {city_data['rainfall'].max():.1f}mm")
        print(f"  Extreme events: {city_data[[col for col in city_data.columns if col not in ['timestamp', 'city', 'temperature', 'rainfall', 'wind', 'humidity', 'pressure']]].sum().sum()}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    combined_data.to_csv('data/weather_training_data.csv', index=False)
    print(f"\nTotal samples: {len(combined_data)}")
    print(f"Saved to: data/weather_training_data.csv")
    
    return combined_data

if __name__ == "__main__":
    print("="*60)
    print("GENERATING HIGH-QUALITY WEATHER TRAINING DATA")
    print("="*60)
    data = generate_all_cities_data()
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)