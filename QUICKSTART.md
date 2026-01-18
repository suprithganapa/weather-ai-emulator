# 🚀 Weather AI Emulator - Quick Start Guide

## Step-by-Step Setup & Running Instructions

### ⚙️ PHASE 1: Environment Setup (5 minutes)

1. **Open Terminal/Command Prompt**

2. **Navigate to project directory:**
   ```bash
   cd ai-emulator-weather
   ```

3. **Create virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This may take 5-10 minutes depending on your internet speed.

---

### 📊 PHASE 2: Data Preparation (2-5 minutes)

**Option A: Using Your Real Dataset**
1. Place your CSV file at: `data/processed/nasa_power_labeled_v2.csv`
2. Verify it has all required columns (see README.md)

**Option B: Generate Sample Data (for testing)**
```bash
python generate_sample_data.py
```

This creates synthetic weather data for testing.

---

### 🧠 PHASE 3: Train Models (1-8 hours total)

**Important:** You need to train at least ONE model before the API will work.

#### Train 1-hour model (REQUIRED):
```bash
python train/train_cnn_lstm_1h.py
```

**Expected time:**
- CPU: 1-2 hours
- GPU: 10-20 minutes

**Output:** Creates `checkpoints/model_1h.pt` and `checkpoints/stats_1h.npy`

#### Train other horizons (OPTIONAL):

To train 3h, 6h, 12h, 24h models, you need to create similar training scripts.

**Quick way:** Copy and modify the 1h script:

```bash
# For 3-hour model
cp train/train_cnn_lstm_1h.py train/train_cnn_lstm_3h.py
# Then edit train_cnn_lstm_3h.py:
#   - Change HORIZON_HOURS = 3
#   - Change CHECKPOINT_PATH to 'model_3h.pt'
#   - Change STATS_PATH to 'stats_3h.npy'

# Repeat for 6h, 12h, 24h
```

---

### 🚀 PHASE 4: Start Backend API (30 seconds)

**From project root directory:**

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**You should see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**API is now running at:**
- Main: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

### 🧪 PHASE 5: Test the API (2 minutes)

#### Option 1: Use Web Browser

1. Go to http://localhost:8000/docs
2. Click on `POST /predict`
3. Click "Try it out"
4. Enter:
   ```json
   {
     "city": "Bangalore",
     "horizon": "1"
   }
   ```
5. Click "Execute"

#### Option 2: Use curl (Command Line)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"city": "Bangalore", "horizon": "1"}'
```

#### Option 3: Use Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"city": "Bangalore", "horizon": "1"}
)

print(response.json())
```

**Expected Response:**
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
    ...
  }
}
```

---

### 🎨 PHASE 6: Frontend (Next Steps)

The backend is now working! For the frontend:

**Option A: Build it yourself**
- Create Next.js app in `frontend/` directory
- Use the API at `http://localhost:8000/predict`
- See frontend design notes below

**Option B: Use Postman/curl for now**
- The API is fully functional
- You can make predictions for all 6 cities
- Test different time horizons

---

## 🐛 Common Issues & Solutions

### Issue 1: "Module not found"
**Solution:** Always run from project root:
```bash
cd ai-emulator-weather
python -m uvicorn backend.main:app --reload
```

### Issue 2: "Model checkpoint not found"
**Solution:** Train the model first:
```bash
python train/train_cnn_lstm_1h.py
```

### Issue 3: "CSV file not found"
**Solution:** Generate sample data:
```bash
python generate_sample_data.py
```

### Issue 4: Port 8000 already in use
**Solution:** Use a different port:
```bash
python -m uvicorn backend.main:app --reload --port 8001
```

---

## 📋 Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Dataset prepared (real or sample)
- [ ] At least 1h model trained
- [ ] Backend running successfully
- [ ] API tested and working
- [ ] (Optional) Other horizons trained
- [ ] (Optional) Frontend built

---

## 🎯 What You Should Have Now

✅ **Working Backend API** that can:
- Accept city and time horizon
- Fetch current weather
- Make predictions using trained models
- Return temperature, rainfall, wind, and event probabilities

✅ **Trained Model(s)** that can predict:
- 1 hour ahead (minimum)
- Optionally: 3h, 6h, 12h, 24h

✅ **Documentation** for:
- API endpoints
- Model architecture
- Training process

---

## 🔮 Next Steps

1. **Train remaining models** (3h, 6h, 12h, 24h)
2. **Build frontend** with Next.js + Tailwind
3. **Add charts** for visualization
4. **Deploy to cloud** (Railway + Vercel)
5. **Add more features:**
   - Historical predictions
   - Accuracy tracking
   - User authentication
   - Favorite cities
   - Alerts/notifications

---

## 📞 Need Help?

Check:
1. README.md (main documentation)
2. IMPLEMENTATION_ROADMAP.md (detailed plan)
3. ISSUE_RESOLUTION_GUIDE.md (troubleshooting)
4. Code comments in each file

---

**You're ready to go! 🚀**

Start with: `python generate_sample_data.py` → `python train/train_cnn_lstm_1h.py` → `python -m uvicorn backend.main:app --reload`
