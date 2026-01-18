"""
Sample Data Generator
Generates synthetic weather data for testing when real dataset is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(
    cities=['Bangalore', 'Mumbai', 'Meghalaya', 'Wayanad', 'Chennai', 'Delhi'],
    hours=24*365,  # 1 year of data
    output_path='data/processed/nasa_power_labeled_v2.csv'
):
    """
    Generate synthetic weather data for testing.
    
    Args:
        cities: List of cities
        hours: Number of hours of data to generate
        output_path: Where to save the CSV
    """
    print(f"Generating {hours} hours of synthetic data for {len(cities)} cities...")
    
    # City-specific base parameters
    city_params = {
        'Bangalore': {'temp_base': 25, 'rain_prob': 0.15, 'wind_base': 3.0},
        'Mumbai': {'temp_base': 28, 'rain_prob': 0.20, 'wind_base': 4.5},
        'Meghalaya': {'temp_base': 20, 'rain_prob': 0.35, 'wind_base': 2.5},
        'Wayanad': {'temp_base': 23, 'rain_prob': 0.25, 'wind_base': 2.0},
        'Chennai': {'temp_base': 30, 'rain_prob': 0.18, 'wind_base': 5.0},
        'Delhi': {'temp_base': 27, 'rain_prob': 0.12, 'wind_base': 3.5},
    }
    
    all_data = []
    
    for city in cities:
        print(f"  Generating data for {city}...")
        params = city_params[city]
        
        # Start time
        start_time = datetime.now() - timedelta(hours=hours)
        
        for h in range(hours):
            time = start_time + timedelta(hours=h)
            
            # Time-based variations (seasonal, daily)
            hour_of_day = time.hour
            month = time.month
            
            # Temperature: varies by time of day and season
            temp = params['temp_base']
            temp += 5 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily cycle
            temp += 3 * np.sin(2 * np.pi * month / 12)  # Seasonal cycle
            temp += np.random.normal(0, 1.5)  # Random noise
            
            # Rainfall: more likely at certain times
            rain_prob = params['rain_prob']
            if hour_of_day >= 14 and hour_of_day <= 18:  # Afternoon rain more likely
                rain_prob *= 2
            if month in [6, 7, 8, 9]:  # Monsoon season
                rain_prob *= 3
            
            rain = np.random.exponential(2.0) if np.random.random() < rain_prob else 0
            
            # Wind
            wind = params['wind_base'] + np.random.normal(0, 1.0)
            wind = max(0, wind)
            
            # Humidity: inversely related to temperature
            humidity = 70 - (temp - params['temp_base']) * 2 + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 95)
            
            # Pressure
            pressure = 1013 + np.random.normal(0, 3)
            
            # Extreme events (binary labels based on thresholds)
            cloudburst = 1 if rain > 10 else 0
            thunderstorm = 1 if rain > 5 and wind > 6 else 0
            heatwave = 1 if temp > params['temp_base'] + 8 else 0
            coldwave = 1 if temp < params['temp_base'] - 8 else 0
            cyclone_like = 1 if wind > 10 and rain > 5 else 0
            heavy_rain = 1 if rain > 7 else 0
            high_wind = 1 if wind > 8 else 0
            fog = 1 if humidity > 85 and temp < 20 else 0
            drought = 1 if rain == 0 and humidity < 30 else 0
            humidity_extreme = 1 if humidity > 90 or humidity < 25 else 0
            
            all_data.append({
                'city': city,
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'rain': round(rain, 2),
                'temp': round(temp, 2),
                'wind': round(wind, 2),
                'humidity': round(humidity, 2),
                'pressure': round(pressure, 2),
                'cloudburst': cloudburst,
                'thunderstorm': thunderstorm,
                'heatwave': heatwave,
                'coldwave': coldwave,
                'cyclone_like': cyclone_like,
                'heavy_rain': heavy_rain,
                'high_wind': high_wind,
                'fog': fog,
                'drought': drought,
                'humidity_extreme': humidity_extreme,
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(df)} rows of data")
    print(f"   Saved to: {output_path}")
    print(f"\nData summary:")
    print(df.describe())
    print(f"\nEvent distribution:")
    event_cols = ['cloudburst', 'thunderstorm', 'heatwave', 'coldwave', 
                  'cyclone_like', 'heavy_rain', 'high_wind', 'fog', 
                  'drought', 'humidity_extreme']
    print(df[event_cols].sum())


if __name__ == '__main__':
    generate_sample_data()
