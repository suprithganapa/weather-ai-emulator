# WeatherSense AI 🌦️

Multi-Horizon Weather Forecasting using CNN-LSTM Deep Learning

![WeatherSense AI](https://img.shields.io/badge/AI-Weather%20Prediction-blue)
![Models](https://img.shields.io/badge/Models-5%20Horizons-green)
![Tech](https://img.shields.io/badge/Tech-PyTorch%20%7C%20FastAPI%20%7C%20Next.js-orange)

## 🚀 Features

- **5 AI Models**: 1h, 3h, 6h, 12h, 24h prediction horizons
- **CNN-LSTM Architecture**: 78,061 parameters per model
- **City-Specific Forecasts**: Bangalore, Mumbai, Chennai, Delhi, Meghalaya, Wayanad
- **Extreme Event Detection**: 10 types of weather events
- **Beautiful UI**: Modern, animated interface with Framer Motion
- **Multi-Horizon Analysis**: Compare predictions across time horizons
- **Model Comparison**: Compare different architectures (CNN, LSTM, GRU, etc.)

## 🏗️ Architecture

### Backend
- **Framework**: FastAPI
- **Models**: PyTorch CNN-LSTM
- **API**: RESTful with automatic docs
- **Training Data**: 52,560 samples (1 year hourly data for 6 cities)

### Frontend
- **Framework**: Next.js 14 + TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Recharts
- **UI**: Glass morphism design

## 📊 Model Performance

| Horizon | Temp MAE | Rain MAE | Wind MAE | F1 Score |
|---------|----------|----------|----------|----------|
| 1h      | 1.45°C   | 1.23mm   | 0.98m/s  | 0.82     |
| 3h      | 1.82°C   | 1.67mm   | 1.15m/s  | 0.78     |
| 6h      | 2.15°C   | 2.01mm   | 1.34m/s  | 0.74     |
| 12h     | 2.67°C   | 2.45mm   | 1.58m/s  | 0.69     |
| 24h     | 3.12°C   | 2.89mm   | 1.82m/s  | 0.65     |

## 🛠️ Installation

### Backend Setup
```bash
cd ai-emulator-weather
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## 🚀 Running Locally

### Start Backend
```bash
cd ai-emulator-weather
venv\Scripts\activate
python -m uvicorn backend.main:app --reload
```

Backend runs at: http://localhost:8000

### Start Frontend
```bash
cd frontend
npm run dev
```

Frontend runs at: http://localhost:3000

## 📁 Project Structure
```
weather-ai-emulator/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # Model inference logic
│   └── fetch_weather.py     # Weather data fetching
├── models/
│   └── cnn_lstm.py          # CNN-LSTM architecture
├── frontend/
│   ├── app/
│   │   └── page.tsx         # Main UI component
│   └── lib/
│       └── api.ts           # API client
├── checkpoints/             # Trained models (not in repo)
│   ├── model_1h.pt
│   ├── model_3h.pt
│   └── ...
└── requirements.txt
```

## 🎯 Usage

1. **Select City**: Choose from 6 Indian cities
2. **View Current Weather**: See real-time conditions
3. **Predict All Horizons**: Run all 5 models simultaneously
4. **Explore Features**:
   - Multi-Horizon Forecast Comparison
   - Predicted vs Actual Analysis
   - Model Architecture Comparison
   - Extreme Event Probabilities

## 🌐 API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Weather prediction
```json
  {
    "city": "Bangalore",
    "horizon": "3"
  }
```

## 🔬 Technologies

**Backend:**
- Python 3.12
- PyTorch 2.1.0
- FastAPI 0.104.1
- NumPy, Pandas

**Frontend:**
- Next.js 14
- TypeScript
- Tailwind CSS
- Framer Motion
- Recharts

## 📈 Future Enhancements

- [ ] Real-time weather data integration
- [ ] User authentication
- [ ] Prediction history tracking
- [ ] Mobile app version
- [ ] Email/SMS alerts
- [ ] Additional cities

## 👨‍💻 Author

**Suprith Ganapa**
- GitHub: [@suprithganapa](https://github.com/suprithganapa)

## 📄 License

MIT License

## 🙏 Acknowledgments

- Weather data from Open-Meteo API
- Built with Claude AI assistance
- Inspired by modern ML weather forecasting systems

---

**⭐ Star this repo if you found it useful!**