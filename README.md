# 🌦️ Weather AI Emulator

Full-stack AI-powered weather prediction system with multi-horizon forecasting using deep learning.

## 🎯 Features

- **Multi-Horizon Predictions**: 1h, 3h, 6h, 12h, 24h ahead forecasting
- **6 Cities Supported**: Bangalore, Mumbai, Meghalaya, Wayanad, Chennai, Delhi
- **Deep Learning Models**: CNN-LSTM hybrid architecture
- **Weather Parameters**: Temperature, Rainfall, Wind Speed
- **Extreme Events**: 10 event types (cloudburst, thunderstorm, heatwave, etc.)
- **REST API**: FastAPI backend with Swagger documentation
- **Web Interface**: Next.js frontend (coming soon)

## 🏗️ Project Structure

```
ai-emulator-weather/
├── backend/           # FastAPI server
│   ├── main.py       # API endpoints
│   ├── inference.py  # Model inference
│   └── fetch_weather.py  # Weather data fetcher
├── datasets/         # Dataset classes
│   ├── window_dataset.py  # Base dataset
│   └── window_dataset_horizon.py  # Multi-horizon dataset
├── models/           # PyTorch models
│   └── cnn_lstm.py   # CNN-LSTM architecture
├── train/            # Training scripts
│   └── train_cnn_lstm_1h.py  # 1-hour model training
├── checkpoints/      # Trained models (created after training)
└── data/
    └── processed/    # Dataset CSV files
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12+ (or 3.10+)
- pip

### 2. Installation

```bash
# Clone or navigate to project directory
cd ai-emulator-weather

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Dataset

Place your dataset CSV file at:
```
data/processed/nasa_power_labeled_v2.csv
```

**Required CSV columns:**
- `city` - City name
- `time` - Timestamp
- `rain` - Rainfall (mm)
- `temp` - Temperature (°C)
- `wind` - Wind speed (m/s)
- `humidity` - Relative humidity (%)
- `pressure` - Atmospheric pressure (hPa)
- Event columns: `cloudburst`, `thunderstorm`, `heatwave`, `coldwave`, `cyclone_like`, `heavy_rain`, `high_wind`, `fog`, `drought`, `humidity_extreme`

### 4. Train Models

Train the 1-hour ahead model:

```bash
cd train
python train_cnn_lstm_1h.py
```

**Expected output:**
- Model saved to: `checkpoints/model_1h.pt`
- Stats saved to: `checkpoints/stats_1h.npy`

**To train other horizons**, create similar scripts:
- `train_cnn_lstm_3h.py` (change `HORIZON_HOURS = 3`)
- `train_cnn_lstm_6h.py` (change `HORIZON_HOURS = 6`)
- `train_cnn_lstm_12h.py` (change `HORIZON_HOURS = 12`)
- `train_cnn_lstm_24h.py` (change `HORIZON_HOURS = 24`)

### 5. Start Backend API

```bash
# From project root directory
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- Main API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 6. Test API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Bangalore",
    "horizon": "1"
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"city": "Bangalore", "horizon": "1"}
)
print(response.json())
```

**Expected response:**
```json
{
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
```

## 📊 Model Architecture

**CNN-LSTM Hybrid:**

1. **Input**: 6-hour sliding window (6, 5) - 6 timesteps × 5 features
2. **CNN Layer**: Extract spatial patterns from features
3. **LSTM Layer**: Model temporal dependencies
4. **Regression Head**: Predict continuous values (rain, temp, wind)
5. **Classification Head**: Predict event probabilities (10 events)

**Training Configuration:**
- Lookback window: 6 hours
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam (lr=1e-3)
- Loss: MSE (regression) + BCE (classification)

## 🔧 Troubleshooting

### Issue: Module not found errors

**Solution:**
```bash
# Always run from project root
cd /path/to/ai-emulator-weather

# Run backend
python -m uvicorn backend.main:app --reload
```

### Issue: Model checkpoint not found

**Solution:**
Train the model first:
```bash
python train/train_cnn_lstm_1h.py
```

### Issue: CORS errors in frontend

**Solution:**
CORS is already configured in `backend/main.py`. Ensure backend is running on port 8000.

### Issue: Dataset loading errors

**Solution:**
1. Verify CSV file exists at `data/processed/nasa_power_labeled_v2.csv`
2. Check CSV has all required columns
3. Verify cities are spelled correctly

## 📈 Performance Metrics

**Target Performance:**
- Temperature MAE: < 2°C
- Rainfall MAE: < 3mm
- Wind MAE: < 2 m/s
- Event Detection F1: > 0.7

## 🌐 API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.

### `POST /predict`
Make weather prediction.

**Request Body:**
```json
{
  "city": "Bangalore",
  "horizon": "1"  // or "3", "6", "12", "24"
}
```

**Response:**
```json
{
  "city": "Bangalore",
  "horizon_hours": 1,
  "temperature": 25.3,
  "rainfall": 2.1,
  "wind": 3.2,
  "events": { ... }
}
```

## 🔮 Future Enhancements

- [ ] Add frontend (Next.js + Tailwind)
- [ ] Add charts and visualizations
- [ ] Deploy to cloud (Railway + Vercel)
- [ ] Add more cities
- [ ] Implement ensemble models
- [ ] Add uncertainty quantification
- [ ] Real-time weather data integration
- [ ] Historical predictions tracking

## 📝 Notes

- **Data Source**: Currently uses Open-Meteo API for current weather (free, no API key)
- **Fallback**: If API fails, uses typical weather patterns for each city
- **Training Time**: ~1-2 hours per model on CPU, ~10-20 minutes on GPU
- **Model Size**: ~500KB per checkpoint

## 🐛 Known Issues

1. ❌ Frontend not yet implemented
2. ❌ Only 1h model trained by default (need to train 3h, 6h, 12h, 24h)
3. ❌ No model versioning or A/B testing

## 📄 License

This project is for educational and portfolio purposes.

## 🤝 Contributing

This is a personal project. Feel free to fork and adapt for your needs.

---

**Built with ❤️ using PyTorch, FastAPI, and Next.js**
