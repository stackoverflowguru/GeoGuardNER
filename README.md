# GeoGuard NER v0.5

Free local Smart India Hackathon prototype for AI-assisted landslide risk monitoring in Northeast India.

## What this version fixes

- Alert page now shows CRITICAL / WARNING / WATCH / REPORT / INFO items instead of appearing empty when there are no high alerts.
- AI prediction is clearly separated from weather screening. No screening score is mislabeled as an AI prediction.
- State AI inputs use the correct 1h / 3h / 6h / 12h / 24h / 48h / 72h / 7d rainfall windows instead of reusing the 24h value.
- Historical ML feature extraction uses antecedent weather only (ends the day before the landslide date) to reduce look-ahead leakage.
- NASA GLC queries are paginated and de-duplicated.
- NASA event dates are normalized from ISO dates or ArcGIS timestamps.
- Citizen report GPS coordinates accept six decimal places and the My Location button shows GPS accuracy.
- Risk map colors state monitoring zones by the actual model risk score when a validated research model is available, otherwise by live weather/soil screening.
- GSI/NGDR is verified as an official portal source; the application does not claim that a raw geology layer is connected when the official portal does not expose a public download endpoint.

## Data sources

- NASA Global Landslide Catalog: historical reported rainfall-triggered landslide inventory.
- Open-Meteo / ECMWF IFS HRES 9 km: current/forecast model weather and rainfall.
- Open-Meteo historical ECMWF IFS: historical rainfall and soil moisture for the ML baseline.
- USGS/NASA SRTM 1 Arc-Second: point elevation and slope from public HGT tiles.
- Geological Survey of India / NGDR and NLFC Bhusanket: official source traceability.
- OpenStreetMap / Overpass: infrastructure exposure.

## Important accuracy note

ECMWF IFS HRES 9 km is a numerical weather model, not a point weather-station observation. For official operational warnings and station observations in India, consult IMD and the official GSI/NLFC bulletins. The AI output is a research-model risk score, not a guaranteed probability of a landslide.

## Run locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Frontend in a second terminal:

```powershell
cd frontend
python -m http.server 5173
```

Open http://127.0.0.1:5173

## Train the real baseline model

Open **AI Prediction** and click **Train / update real AI model**. Training uses NASA GLC events from 2017 onward, historical ECMWF weather via Open-Meteo, and SRTM point features. The date-based train/validation/test split prevents overlap of the same event date across partitions.

Training can take several minutes and requires internet access for public data retrieval.
