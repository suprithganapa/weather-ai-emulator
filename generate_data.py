import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

CITY_PATTERNS = {
    'Bangalore': {'temp_base': 24.0, 'temp_amplitude': 5.0, 'temp_std': 1.2, 'rain_base': 1.2, 'rain_amplitude': 3.5, 'rain_std': 1.8, 'wind_base': 2.5, 'wind_std': 0.7, 'humidity_base': 65.0, 'humidity_std': 8.0, 'monsoon_months': [6,7,8,9], 'monsoon_factor': 4.0},
    'Mumbai': {'temp_base': 28.0, 'temp_amplitude': 4.0, 'temp_std': 1.0, 'rain_base': 2.5, 'rain_amplitude': 8.0, 'rain_std': 4.5, 'wind_base': 3.5, 'wind_std': 1.0, 'humidity_base': 75.0, 'humidity_std': 10.0, 'monsoon_months': [6,7,8,9], 'monsoon_factor': 8.0},
    'Chennai': {'temp_base': 30.0, 'temp_amplitude': 5.0, 'temp_std': 1.5, 'rain_base': 1.5, 'rain_amplitude': 4.0, 'rain_std': 2.8, 'wind_base': 3.0, 'wind_std': 0.9, 'humidity_base': 70.0, 'humidity_std': 12.0, 'monsoon_months': [10,11,12], 'monsoon_factor': 5.0},
    'Delhi': {'temp_base': 25.0, 'temp_amplitude': 12.0, 'temp_std': 2.5, 'rain_base': 0.8, 'rain_amplitude': 3.0, 'rain_std': 2.0, 'wind_base': 2.2, 'wind_std': 0.8, 'humidity_base': 55.0, 'humidity_std': 15.0, 'monsoon_months': [7,8], 'monsoon_factor': 6.0},
    'Shillong': {'temp_base': 18.0, 'temp_amplitude': 6.0, 'temp_std': 1.8, 'rain_base': 8.0, 'rain_amplitude': 12.0, 'rain_std': 7.0, 'wind_base': 2.0, 'wind_std': 0.6, 'humidity_base': 85.0, 'humidity_std': 8.0, 'monsoon_months': [5,6,7,8,9], 'monsoon_factor': 15.0},
    'Wayanad': {'temp_base': 22.0, 'temp_amplitude': 4.0, 'temp_std': 1.3, 'rain_base': 3.5, 'rain_amplitude': 6.0, 'rain_std': 3.5, 'wind_base': 1.8, 'wind_std': 0.5, 'humidity_base': 78.0, 'humidity_std': 10.0, 'monsoon_months': [6,7,8], 'monsoon_factor': 7.0},
    'Jaipur': {'temp_base': 28.0, 'temp_amplitude': 14.0, 'temp_std': 3.0, 'rain_base': 0.3, 'rain_amplitude': 1.5, 'rain_std': 0.8, 'wind_base': 3.0, 'wind_std': 1.3, 'humidity_base': 45.0, 'humidity_std': 18.0, 'monsoon_months': [7,8], 'monsoon_factor': 3.0},
    'Dharali': {'temp_base': 15.0, 'temp_amplitude': 8.0, 'temp_std': 2.2, 'rain_base': 4.0, 'rain_amplitude': 10.0, 'rain_std': 6.0, 'wind_base': 2.5, 'wind_std': 0.9, 'humidity_base': 70.0, 'humidity_std': 12.0, 'monsoon_months': [6,7,8,9], 'monsoon_factor': 12.0},
    'Ladakh': {'temp_base': 8.0, 'temp_amplitude': 15.0, 'temp_std': 3.5, 'rain_base': 0.1, 'rain_amplitude': 0.5, 'rain_std': 0.2, 'wind_base': 4.0, 'wind_std': 1.8, 'humidity_base': 35.0, 'humidity_std': 15.0, 'monsoon_months': [], 'monsoon_factor': 1.0, 'winter_months': [11,12,1,2,3]}
}

def gen_city(city, start, hrs=8760):
    p = CITY_PATTERNS[city]
    data = []
    cd = start
    print(f'  {city}...')
    for h in range(hrs):
        doy = cd.timetuple().tm_yday
        hod = cd.hour
        m = cd.month
        t = p['temp_base'] + p['temp_amplitude']*np.sin(2*np.pi*doy/365) + 3.0*np.sin(2*np.pi*(hod-6)/24) + np.random.randn()*p['temp_std']
        if city=='Ladakh' and m in p.get('winter_months',[]): t -= 10
        ism = m in p['monsoon_months']
        mf = p['monsoon_factor'] if ism else 1.0
        r = p['rain_base']*mf + abs(np.random.randn()*p['rain_std']*mf)
        if ism and np.random.random()<0.05: r += np.random.exponential(10)
        if city in ['Dharali','Shillong'] and ism and np.random.random()<0.02: r += np.random.exponential(30)
        if city=='Ladakh' and m in p.get('winter_months',[]) and t<0 and np.random.random()<0.15: r += np.random.exponential(2)
        r = max(0,r)
        w = max(0, p['wind_base']+np.random.randn()*p['wind_std'])
        if r>10: w += np.random.exponential(2)
        if city=='Ladakh' and m in p.get('winter_months',[]): w += np.random.uniform(1,3)
        hb = p['humidity_base']
        if r>5: hb = min(95,hb+15)
        if city=='Ladakh' and m in p.get('winter_months',[]): hb = max(20,hb-20)
        hu = np.clip(hb+np.random.randn()*p['humidity_std'],20,100)
        pr = 1013+np.random.randn()*3
        if r>15: pr -= np.random.uniform(5,15)
        if city=='Ladakh': pr -= 150
        ev = {'cloudburst':1 if r>50 else 0,'thunderstorm':1 if (r>15 and w>8) else 0,'heatwave':1 if t>40 else 0,'coldwave':1 if t<5 else 0,'cyclone_like':1 if (w>15 and r>20) else 0,'heavy_rain':1 if r>25 else 0,'high_wind':1 if w>12 else 0,'fog':1 if (hu>90 and t<15) else 0,'drought':1 if (r<0.1 and not ism) else 0,'humidity_extreme':1 if (hu>90 or hu<30) else 0}
        data.append({'timestamp':cd,'city':city,'temperature':round(t,2),'rainfall':round(r,2),'wind':round(w,2),'humidity':round(hu,2),'pressure':round(pr,2),**ev})
        cd += timedelta(hours=1)
    return pd.DataFrame(data)

print('Generating data...')
start = datetime(2023,1,1)
all_data = []
for city in CITY_PATTERNS.keys():
    all_data.append(gen_city(city,start))
df = pd.concat(all_data,ignore_index=True)
os.makedirs('data',exist_ok=True)
df.to_csv('data/weather_training_data.csv',index=False)
print(f'Done! {len(df)} samples')