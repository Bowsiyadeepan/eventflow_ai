# 🚦 EventFlow AI
**Event-Driven Traffic Impact Forecasting & Smart Enforcement Recommendation System**

## Problem Statement
Traffic disruptions from vehicle breakdowns, accidents, waterlogging, construction, and public events cause severe urban congestion. Today, enforcement is reactive, resource deployment is experience-driven, and there is no post-event learning system.

## Solution
EventFlow AI uses an ensemble ML model (LightGBM + XGBoost + CatBoost) trained on 8,000+ real traffic events from Bengaluru to:
- **Predict** traffic impact priority (High/Low) for any incoming event
- **Recommend** manpower, barricading, and diversion plans
- **Visualize** zone-wise and corridor-wise congestion hotspots
- **Learn** from post-event feedback to improve future predictions

## Tech Stack
- **Frontend:** Streamlit
- **ML Models:** LightGBM, XGBoost, CatBoost (Ensemble)
- **Visualization:** Plotly
- **Data:** Astram Event Dataset (8,173 events, 46 features)

## Project Structure
```
eventflow_ai/
├── app.py              # Main Streamlit application
├── model.py            # Feature engineering, training, prediction, recommendations
├── astram_events.csv   # Dataset
├── requirements.txt    # Dependencies
└── README.md
```

## Instructions to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/eventflow-ai.git
cd eventflow-ai
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

### Step 4 — Open in browser
```
http://localhost:8501
```

## Features
| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, event distribution, hourly trends, zone analysis |
| 🔮 Predict & Recommend | Enter event details → get priority + enforcement plan |
| 🗺️ Zone Heatmap | Filterable zone/corridor heatmap + incident map |
| 📈 Model Performance | Accuracy, precision, recall, F1, feature importance |

## Dataset
- **Source:** Astram Event Data (Bengaluru Traffic Management)
- **Records:** 8,173 events
- **Features:** 46 columns including event cause, zone, corridor, timestamps, vehicle type
- **Target:** Priority (High/Low)

## Model Performance
- **Accuracy:** ~85%+ on test set
- **Ensemble Strategy:** Simple average of LightGBM, XGBoost, CatBoost probabilities
- **Features:** 15 engineered features (temporal, spatial, event type)
