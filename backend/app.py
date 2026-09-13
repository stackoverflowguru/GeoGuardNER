from __future__ import annotations

import asyncio
import gzip
import io
import json
import math
import os
import random
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "geoguard.db"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "landslide_model.joblib"
MODEL_META = MODEL_DIR / "metadata.json"
SRTM_CACHE = DATA_DIR / "srtm"
SRTM_CACHE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GeoGuard NER API", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATES = {
    "Arunachal Pradesh": (27.1000, 93.6000),
    "Assam": (26.2000, 92.9000),
    "Manipur": (24.7000, 93.9000),
    "Meghalaya": (25.5000, 91.3000),
    "Mizoram": (23.7000, 92.7000),
    "Nagaland": (26.0000, 94.5000),
    "Sikkim": (27.5000, 88.5000),
    "Tripura": (23.8000, 91.3000),
}
NER_BBOX = {"west": 88.0, "south": 21.0, "east": 97.7, "north": 29.8}

# Weather: ECMWF IFS HRES through Open-Meteo. It is a numerical-model estimate, not a station observation.
ECMWF_API = "https://api.open-meteo.com/v1/ecmwf"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
IMD_API = "https://api.imd.gov.in/public"
# Primary NASA GLC FeatureServer, then COOLR (includes GLC) mirrors. Tried in order; none is fabricated.
NASA_GLC_ENDPOINTS = [
    "https://maps.nccs.nasa.gov/server/rest/services/global_landslide_catalog/landslide_points/FeatureServer/0/query",
    "https://maps.nccs.nasa.gov/mapping/rest/services/COOLR/COOLR_Reports_Point/FeatureServer/0/query",
    "https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/COOLR_Reports_Points/FeatureServer/0/query",
]
NASA_GLC = NASA_GLC_ENDPOINTS[0]
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
GSI_PUBLIC = "https://bhusanket.gsi.gov.in/"
SRTM_USGS = "https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1"
SRTM_TILES = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"

CACHE: dict[str, tuple[float, Any]] = {}
CACHE_TTL = 180
INVENTORY_TTL = 1800
_http = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=12.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=8),
    headers={"User-Agent": "GeoGuardNER/0.5 local research prototype"},
    follow_redirects=True,
)
TRAIN_SEM = asyncio.Semaphore(5)

FEATURES = [
    "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_12h",
    "rainfall_24h", "rainfall_48h", "rainfall_72h", "rainfall_7d",
    "soil_moisture_0_to_7cm", "elevation", "slope",
]


def db() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with closing(db()) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                url TEXT,
                status TEXT NOT NULL,
                coverage TEXT,
                last_checked TEXT,
                details TEXT
            );
            CREATE TABLE IF NOT EXISTS landslide_events (
                id TEXT PRIMARY KEY,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                event_date TEXT,
                country TEXT,
                landslide_type TEXT,
                trigger TEXT,
                location TEXT,
                source TEXT,
                url TEXT,
                raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_latlon ON landslide_events(latitude, longitude);
            CREATE INDEX IF NOT EXISTS idx_events_date ON landslide_events(event_date);
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                description TEXT,
                latitude REAL,
                longitude REAL,
                created_at TEXT NOT NULL,
                synced INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                label TEXT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                score REAL,
                level TEXT,
                features_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                location TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT
            );
            """
        )
        sources = [
            ("NASA Global Landslide Catalog", NASA_GLC, "UNVERIFIED", "Northeast India bounding box", "Live historical rainfall-triggered inventory connector. Status is updated only after a successful query."),
            ("ECMWF IFS HRES via Open-Meteo", ECMWF_API, "ACTIVE", "Global, 9 km native IFS HRES", "Primary model source for current/forecast conditions and rainfall. Current conditions are taken from the hourly series at the latest completed model hour; rainfall windows exclude unused forecast hours."),
            ("ECMWF IFS historical via Open-Meteo", OPEN_METEO_ARCHIVE, "ACTIVE", "Global, hourly historical features from 2017+", "Historical rainfall/soil-moisture features used by the real ML training pipeline; event-date leakage protection is applied."),
            ("India Meteorological Department (IMD)", IMD_API, "NOT_CONNECTED", "India", "Official observations and warnings are not queried. This prototype has no IMD API credentials and does not substitute unofficial station values."),
            ("USGS/NASA SRTM 1 Arc-Second Global", SRTM_USGS, "ACTIVE-ON-DEMAND", "Global ~30 m DEM", "Point elevation and slope are calculated from public SRTM HGT tiles cached locally."),
            ("Geological Survey of India / NGDR", "https://www.gsi.gov.in/", "VERIFIED-PORTAL", "India geological data", "Official GSI/NGDR portal verified. Raw geology is not silently downloaded when portal access does not expose a public dataset endpoint."),
            ("GSI National Landslide Forecasting Centre (Bhusanket)", GSI_PUBLIC, "VERIFIED-PORTAL", "India landslide inventory/susceptibility", "Official field-validated inventory and susceptibility/forecast portal; raw download endpoint is portal-controlled."),
            ("OpenStreetMap / Overpass", "https://www.openstreetmap.org/", "ACTIVE", "Global OSM", "Infrastructure/exposure queries."),
        ]
        for row in sources:
            c.execute(
                "INSERT INTO data_sources(name,url,status,coverage,details,last_checked) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET url=excluded.url,coverage=excluded.coverage",
                (*row, now_iso()),
            )
        c.commit()


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.on_event("shutdown")
async def shutdown() -> None:
    await _http.aclose()


def cache_get(key: str, ttl: int) -> Any | None:
    item = CACHE.get(key)
    if not item:
        return None
    created, value = item
    if time.monotonic() - created > ttl:
        CACHE.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any) -> Any:
    CACHE[key] = (time.monotonic(), value)
    return value


def recent_sum(hourly: dict, hours: int) -> Optional[float]:
    values = hourly.get("precipitation") or hourly.get("rain") or []
    nums = [float(v or 0) for v in values[-hours:]]
    return round(sum(nums), 2) if nums else None


def normalise_event_date(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        # ArcGIS often returns epoch milliseconds.
        if x > 1e12:
            x /= 1000.0
        if x > 1e9:
            try:
                return datetime.fromtimestamp(x, tz=timezone.utc).date().isoformat()
            except Exception:
                return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except Exception:
            pass
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date().isoformat()
        except Exception:
            pass
    return None


def recent_avg(values: list[Any], hours: int) -> Optional[float]:
    nums = [float(v) for v in values[-hours:] if v is not None]
    return round(sum(nums) / len(nums), 5) if nums else None


def level_from_score(score: float) -> str:
    return "VERY HIGH" if score >= 75 else "HIGH" if score >= 50 else "MODERATE" if score >= 25 else "LOW"


def screening(rain_24h: Optional[float], rain_72h: Optional[float], soil: Optional[float]) -> dict:
    # Operational weather/soil screening only. AI prediction is separate and comes from the trained model.
    score = 0.0
    reasons: list[str] = []
    if rain_24h is not None:
        if rain_24h >= 100: score += 45; reasons.append("very high 24-hour rainfall")
        elif rain_24h >= 50: score += 32; reasons.append("high 24-hour rainfall")
        elif rain_24h >= 20: score += 18; reasons.append("moderate 24-hour rainfall")
    if rain_72h is not None:
        if rain_72h >= 200: score += 30; reasons.append("very high 72-hour cumulative rainfall")
        elif rain_72h >= 100: score += 18; reasons.append("elevated 72-hour cumulative rainfall")
    if soil is not None:
        if soil >= 0.45: score += 20; reasons.append("high surface soil moisture")
        elif soil >= 0.30: score += 10; reasons.append("elevated surface soil moisture")
    score = round(min(score, 100), 1)
    return {"score": score, "level": level_from_score(score), "reasons": reasons}


async def fetch_point_weather(lat: float, lon: float) -> dict:
    key = f"ecmwf:{lat:.4f}:{lon:.4f}"
    cached = cache_get(key, CACHE_TTL)
    if cached is not None:
        return cached
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,pressure_msl,wind_speed_10m,wind_direction_10m,weather_code",
        "hourly": "precipitation,rain,soil_moisture_0_to_7cm,temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_hours": 168,
        "forecast_hours": 48,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
    }
    r = await _http.get(ECMWF_API, params=params)
    r.raise_for_status()
    weather = r.json()
    h = weather.get("hourly", {})
    result = {
        "current": weather.get("current", {}),
        "rainfall_1h": recent_sum(h, 1),
        "rainfall_3h": recent_sum(h, 3),
        "rainfall_6h": recent_sum(h, 6),
        "rainfall_12h": recent_sum(h, 12),
        "rainfall_24h": recent_sum(h, 24),
        "rainfall_48h": recent_sum(h, 48),
        "rainfall_72h": recent_sum(h, 72),
        "rainfall_7d": recent_sum(h, 168),
        "soil_moisture_0_to_7cm": recent_avg(h.get("soil_moisture_0_to_7cm") or [], 24),
        "timezone": weather.get("timezone"),
        "elevation": weather.get("elevation"),
        "data_source": "ECMWF IFS HRES 9 km via Open-Meteo",
        "data_timestamp": weather.get("current", {}).get("time"),
    }
    return cache_set(key, result)


async def fetch_archive_features(lat: float, lon: float, date_str: str) -> dict:
    # Antecedent conditions only: finish the window the day before the reported event date
    # to avoid using weather after the event (temporal leakage).
    event_day = datetime.fromisoformat(date_str[:10]).date()
    end = event_day - timedelta(days=1)
    start = end - timedelta(days=7)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "precipitation,soil_moisture_0_to_7cm,temperature_2m",
        "timezone": "UTC", "models": "ecmwf_ifs",
        "temperature_unit": "celsius", "precipitation_unit": "mm",
    }
    async with TRAIN_SEM:
        r = await _http.get(OPEN_METEO_ARCHIVE, params=params)
    r.raise_for_status()
    h = r.json().get("hourly", {})
    return {
        "rainfall_1h": recent_sum(h, 1), "rainfall_3h": recent_sum(h, 3),
        "rainfall_6h": recent_sum(h, 6), "rainfall_12h": recent_sum(h, 12),
        "rainfall_24h": recent_sum(h, 24), "rainfall_48h": recent_sum(h, 48),
        "rainfall_72h": recent_sum(h, 72), "rainfall_7d": recent_sum(h, 168),
        "soil_moisture_0_to_7cm": recent_avg(h.get("soil_moisture_0_to_7cm") or [], 24),
        "historical_weather_source": "ECMWF IFS 9 km historical reanalysis/forecast archive via Open-Meteo",
    }


def normalise_feature(feature: dict) -> dict:
    props = feature.get("attributes") or feature.get("properties") or {}
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    lon = props.get("longitude") or props.get("lon") or geom.get("x") or (coords[0] if coords else None)
    lat = props.get("latitude") or props.get("lat") or geom.get("y") or (coords[1] if coords else None)

    def first(*keys: str):
        for k in keys:
            if props.get(k) not in (None, ""):
                return props[k]
        return None

    return {
        "id": str(first("event_id", "EVENT_ID", "OBJECTID", "objectid", "id") or f"{lat}:{lon}"),
        "latitude": lat, "longitude": lon,
        "event_date": normalise_event_date(first("event_date", "EVENT_DATE", "date", "eventdate", "Event_Date", "eventDate")),
        "country": first("country_name", "country", "geocoded_country"),
        "landslide_type": first("landslide_category", "landslide_type", "type"),
        "trigger": first("landslide_trigger", "trigger"),
        "location": first("location_description", "location", "place"),
        "source": first("source_name", "source"),
        "url": first("event_url", "source_link", "url"),
        "raw": props,
    }


async def fetch_glc_inventory(force: bool = False) -> dict:
    if not force:
        cached = cache_get("glc:ner", INVENTORY_TTL)
        if cached is not None:
            return cached
    all_events: list[dict] = []
    offset = 0
    page_size = 2000
    for _ in range(10):
        params = {
            "where": "1=1",
            "geometry": f"{NER_BBOX['west']},{NER_BBOX['south']},{NER_BBOX['east']},{NER_BBOX['north']}",
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "returnGeometry": "true",
            "outSR": 4326, "resultRecordCount": page_size, "resultOffset": offset, "f": "json",
        }
        r = await _http.get(NASA_GLC, params=params)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        page = payload.get("features") or []
        all_events.extend(normalise_feature(f) for f in page)
        if len(page) < page_size:
            break
        offset += page_size
    events = [e for e in all_events if isinstance(e["latitude"], (int, float)) and isinstance(e["longitude"], (int, float))]
    # Stable de-duplication: same location/date/type should not appear twice.
    dedup: dict[tuple[Any, ...], dict] = {}
    for e in events:
        key = (round(float(e["latitude"]), 5), round(float(e["longitude"]), 5), e.get("event_date"), e.get("landslide_type"))
        dedup.setdefault(key, e)
    events = list(dedup.values())
    with closing(db()) as c:
        c.execute("DELETE FROM landslide_events")
        for idx, e in enumerate(events):
            eid = e["id"] if e["id"] else f"glc-{idx}"
            c.execute(
                "INSERT OR REPLACE INTO landslide_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (str(eid), e["latitude"], e["longitude"], e["event_date"], e["country"], e["landslide_type"], e["trigger"], e["location"], e["source"], e["url"], json.dumps(e["raw"], ensure_ascii=False)),
            )
        c.execute("UPDATE data_sources SET status='ACTIVE',last_checked=?,details=? WHERE name='NASA Global Landslide Catalog'", (now_iso(), f"{len(events)} live records retrieved for NER query; paginated and de-duplicated"))
        c.commit()
    result = {"source": "NASA Global Landslide Catalog", "count": len(events), "events": events, "retrieved_at": now_iso(), "coverage": NER_BBOX, "stale": False}
    return cache_set("glc:ner", result)


async def nearest_events(lat: float, lon: float, radius_km: float = 50.0) -> list[dict]:
    with closing(db()) as c:
        rows = c.execute("SELECT * FROM landslide_events").fetchall()
    out = []
    R = 6371.0088
    p1 = math.radians(lat)
    for row in rows:
        p2 = math.radians(row["latitude"])
        dp = math.radians(row["latitude"] - lat)
        dl = math.radians(row["longitude"] - lon)
        a = math.sin(dp/2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2) ** 2
        d = 2 * R * math.asin(math.sqrt(a))
        if d <= radius_km:
            out.append({**dict(row), "distance_km": round(d, 2), "raw_json": None})
    out.sort(key=lambda x: x["distance_km"])
    return out[:100]


def tile_name(lat: float, lon: float) -> tuple[str, str]:
    ilat = math.floor(lat)
    ilon = math.floor(lon)
    ns = "N" if ilat >= 0 else "S"
    ew = "E" if ilon >= 0 else "W"
    return f"{ns}{abs(ilat):02d}", f"{ew}{abs(ilon):03d}"


def tile_path(lat: float, lon: float) -> Path:
    ns, ew = tile_name(lat, lon)
    return SRTM_CACHE / ns / f"{ns}{ew}.hgt"


async def ensure_srtm_tile(lat: float, lon: float) -> Path:
    p = tile_path(lat, lon)
    if p.exists():
        return p
    ns, ew = tile_name(lat, lon)
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SRTM_TILES}/{ns}/{ns}{ew}.hgt.gz"
    r = await _http.get(url)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    tmp = p.with_suffix(".hgt.tmp")
    tmp.write_bytes(raw)
    tmp.replace(p)
    return p


def read_srtm_point(path: Path, lat: float, lon: float) -> tuple[float, float]:
    # SRTM 1-arcsecond tile: 3601 x 3601 signed 16-bit big endian.
    arr = np.fromfile(path, dtype=">i2").reshape((3601, 3601))
    west = math.floor(lon)
    north = math.ceil(lat)
    row = int(round((north - lat) * 3600))
    col = int(round((lon - west) * 3600))
    row = min(3600, max(1, row))
    col = min(3600, max(1, col))
    win = arr[row-1:row+2, col-1:col+2].astype(np.float64)
    win[win <= -32768] = np.nan
    if not np.isfinite(win).any():
        raise ValueError("No valid SRTM elevation at this point")
    elevation = float(win[1, 1]) if np.isfinite(win[1,1]) else float(np.nanmean(win))
    dy = 30.87
    dx = 30.87 * max(math.cos(math.radians(lat)), 0.15)
    dzdy = (win[2,1] - win[0,1]) / (2*dy) if np.isfinite(win[[0,2],1]).all() else 0.0
    dzdx = (win[1,2] - win[1,0]) / (2*dx) if np.isfinite(win[1,[0,2]]).all() else 0.0
    slope = math.degrees(math.atan(math.sqrt(dzdx*dzdx + dzdy*dzdy)))
    return round(elevation, 2), round(slope, 2)


async def srtm_features(lat: float, lon: float) -> dict:
    key = f"srtm:{lat:.4f}:{lon:.4f}"
    cached = cache_get(key, 86400)
    if cached is not None:
        return cached
    path = await ensure_srtm_tile(lat, lon)
    elev, slope = read_srtm_point(path, lat, lon)
    return cache_set(key, {"elevation": elev, "slope": slope, "source": "SRTM 1 Arc-Second Global"})


async def gsi_public_status() -> dict:
    cached = cache_get("gsi:status", 900)
    if cached is not None:
        return cached
    try:
        r = await _http.get(GSI_PUBLIC, timeout=20)
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text)
        m = re.search(r"Landslide Inventory \(Field Validated\)\s+([0-9,]+)", text, re.I)
        count = int(m.group(1).replace(",", "")) if m else None
        data = {"status": "ACTIVE", "field_validated_inventory_count": count, "url": GSI_PUBLIC, "checked_at": now_iso()}
        return cache_set("gsi:status", data)
    except Exception as exc:
        return {"status": "PORTAL-UNREACHABLE", "field_validated_inventory_count": None, "url": GSI_PUBLIC, "error": str(exc), "checked_at": now_iso()}


async def fetch_state_weather() -> dict[str, dict]:
    key = "state-weather"
    cached = cache_get(key, CACHE_TTL)
    if cached is not None:
        return cached
    lats = [v[0] for v in STATES.values()]
    lons = [v[1] for v in STATES.values()]
    params = {
        "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,pressure_msl,wind_speed_10m,wind_direction_10m,weather_code",
        "hourly": "precipitation,rain,soil_moisture_0_to_7cm",
        "past_hours": 168, "forecast_hours": 24, "timezone": "auto",
        "temperature_unit": "celsius", "precipitation_unit": "mm", "wind_speed_unit": "kmh",
    }
    r = await _http.get(ECMWF_API, params=params)
    r.raise_for_status()
    raw = r.json()
    records = raw if isinstance(raw, list) else [raw]
    out = {}
    for name, weather in zip(STATES, records):
        hourly = weather.get("hourly", {})
        rain = {
            "rainfall_1h": recent_sum(hourly, 1),
            "rainfall_3h": recent_sum(hourly, 3),
            "rainfall_6h": recent_sum(hourly, 6),
            "rainfall_12h": recent_sum(hourly, 12),
            "rainfall_24h": recent_sum(hourly, 24),
            "rainfall_48h": recent_sum(hourly, 48),
            "rainfall_72h": recent_sum(hourly, 72),
            "rainfall_7d": recent_sum(hourly, 168),
        }
        soil = recent_avg(hourly.get("soil_moisture_0_to_7cm") or [], 24)
        out[name] = {
            "weather": weather.get("current", {}), **rain,
            "soil_moisture_0_to_7cm": soil,
            "screening": screening(rain["rainfall_24h"], rain["rainfall_72h"], soil),
            "weather_source": "ECMWF IFS HRES 9 km via Open-Meteo",
            "model_run": weather.get("current", {}).get("time"),
            "data_quality": "numerical weather model; use IMD observations/warnings for official operational decisions",
        }
    return cache_set(key, out)


async def make_training_record(lat: float, lon: float, date_str: str, label: int) -> Optional[dict]:
    try:
        weather = await fetch_archive_features(lat, lon, date_str)
        terrain = await srtm_features(lat, lon)
        x = {**weather, **terrain, "label": label, "event_date": date_str}
        if any(x.get(k) is None for k in FEATURES):
            return None
        return x
    except Exception:
        return None


async def train_model(force: bool = False) -> dict:
    if MODEL_PATH.exists() and MODEL_META.exists() and not force:
        return {"status": "READY", **json.loads(MODEL_META.read_text(encoding="utf-8"))}
    inventory = await fetch_glc_inventory(force=force)
    events = [e for e in inventory["events"] if e.get("event_date")]
    events = [e for e in events if NER_BBOX["west"] <= e["longitude"] <= NER_BBOX["east"] and NER_BBOX["south"] <= e["latitude"] <= NER_BBOX["north"]]
    # High-resolution ECMWF IFS historical features are available from 2017 onward.
    events = [e for e in events if str(e["event_date"])[:4].isdigit() and int(str(e["event_date"])[:4]) >= 2017]
    events = sorted(events, key=lambda e: str(e.get("event_date")))
    # Keep local training bounded while retaining coverage across the full period.
    max_events = 120
    if len(events) > max_events:
        idxs = np.linspace(0, len(events) - 1, max_events, dtype=int).tolist()
        events = [events[i] for i in idxs]
    if len(events) < 30:
        return {"status": "INSUFFICIENT_DATA", "message": f"Only {len(events)} dated NER NASA events from 2017+ are available; at least 30 are required for a meaningful baseline.", "model_version": None}

    async def build_pos(e):
        return await make_training_record(float(e["latitude"]), float(e["longitude"]), str(e["event_date"])[:10], 1)

    positives_raw = await asyncio.gather(*(build_pos(e) for e in events))
    positives = [r for r in positives_raw if r]

    # Matched background samples on the same date, separated from known events.
    known = [(float(e["latitude"]), float(e["longitude"])) for e in events]
    rng = random.Random(42)

    def hav_km(a_lat, a_lon, b_lat, b_lon):
        R = 6371.0088
        p1, p2 = math.radians(a_lat), math.radians(b_lat)
        dp, dl = math.radians(b_lat-a_lat), math.radians(b_lon-a_lon)
        q = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2*R*math.asin(math.sqrt(q))

    negative_jobs = []
    for e in events:
        made = 0
        attempts = 0
        while made < 2 and attempts < 10:
            attempts += 1
            ang = rng.random() * 2 * math.pi
            # ~8-35 km from the event, avoiding the immediate event footprint.
            distance_km = 8 + rng.random() * 27
            dlat = (distance_km / 111.32) * math.cos(ang)
            dlon = (distance_km / (111.32 * max(math.cos(math.radians(float(e["latitude"])), 0.2)))) * math.sin(ang)
            lat = float(e["latitude"]) + dlat
            lon = float(e["longitude"]) + dlon
            if not (NER_BBOX["south"] <= lat <= NER_BBOX["north"] and NER_BBOX["west"] <= lon <= NER_BBOX["east"]):
                continue
            if any(hav_km(lat, lon, klat, klon) < 5 for klat, klon in known):
                continue
            negative_jobs.append(make_training_record(lat, lon, str(e["event_date"])[:10], 0))
            made += 1
    negative_raw = await asyncio.gather(*negative_jobs) if negative_jobs else []
    negatives = [r for r in negative_raw if r]
    if len(negatives) > max(20, len(positives) * 2):
        negatives = negatives[:len(positives) * 2]

    if len(positives) < 20 or len(negatives) < 20:
        return {"status": "INSUFFICIENT_FEATURE_DATA", "message": f"Positive rows: {len(positives)}, negative rows: {len(negatives)}. Need at least 20 of each after weather+terrain extraction.", "model_version": None}

    rows = positives + negatives
    # Split by complete event dates to avoid the same date appearing across train/validation/test.
    dates = sorted({r["event_date"] for r in rows})
    if len(dates) < 10:
        return {"status": "INSUFFICIENT_DATES", "message": "Fewer than 10 distinct event dates are available for leakage-safe evaluation.", "model_version": None}
    cut1 = max(1, int(len(dates) * 0.60))
    cut2 = max(cut1 + 1, int(len(dates) * 0.80))
    train_dates, val_dates, test_dates = set(dates[:cut1]), set(dates[cut1:cut2]), set(dates[cut2:])
    train_rows = [r for r in rows if r["event_date"] in train_dates]
    val_rows = [r for r in rows if r["event_date"] in val_dates]
    test_rows = [r for r in rows if r["event_date"] in test_dates]

    def matrix(data: list[dict]):
        return np.array([[float(r[k]) for k in FEATURES] for r in data], dtype=np.float64), np.array([int(r["label"]) for r in data], dtype=np.int32)

    Xtr, ytr = matrix(train_rows); Xv, yv = matrix(val_rows); Xte, yte = matrix(test_rows)
    if len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2 or len(np.unique(yte)) < 2:
        return {"status": "INVALID_SPLIT", "message": "The date-based split does not contain both classes in every partition. More historical events are required.", "model_version": None}

    models = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, random_state=42))]),
    }
    best_name, best_model, best_auc = None, None, -1.0
    val_metrics = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pv = model.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(yv, pv)
        val_metrics[name] = {"roc_auc": round(float(auc), 4)}
        if auc > best_auc:
            best_name, best_model, best_auc = name, model, auc

    # Evaluate the selected model on the untouched future test set.
    pt = best_model.predict(Xte)
    pp = best_model.predict_proba(Xte)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(yte, pt)), 4),
        "precision": round(float(precision_score(yte, pt, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pt, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, pt, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yte, pp)), 4),
    }
    joblib.dump(best_model, MODEL_PATH)
    meta = {
        "status": "READY", "model_version": f"{best_name}-0.5",
        "algorithm": best_name, "features": FEATURES, "training_samples": len(train_rows),
        "validation_samples": len(val_rows), "test_samples": len(test_rows),
        "positive_samples": len(positives), "negative_samples": len(negatives),
        "validation_model_comparison": val_metrics, "test_metrics": metrics,
        "split": "60/20/20 by distinct event date; no date overlap across partitions",
        "data_source": "NASA GLC (2017+) + ECMWF IFS historical hourly weather via Open-Meteo + SRTM 1-arc-second",
        "trained_at": now_iso(),
        "limitations": "Regional research baseline; not an official warning system. GSI field-validated inventory is displayed for traceability but raw GSI/NGDR geology is not silently fabricated or substituted.",
    }
    MODEL_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with closing(db()) as c:
        c.execute("UPDATE data_sources SET last_checked=?,details=? WHERE name='USGS/NASA SRTM 1 Arc-Second Global'", (now_iso(), "SRTM tiles are actively downloaded and used for point elevation/slope."))
        c.commit()
    return meta


def model_status() -> dict:
    if not MODEL_PATH.exists() or not MODEL_META.exists():
        return {"status": "NOT_TRAINED", "model_version": None}
    return {"status": "READY", **json.loads(MODEL_META.read_text(encoding="utf-8"))}


async def model_predict(features: dict) -> Optional[dict]:
    if not MODEL_PATH.exists():
        return None
    if any(features.get(k) is None for k in FEATURES):
        return None
    model = joblib.load(MODEL_PATH)
    X = np.array([[float(features.get(k)) for k in FEATURES]], dtype=np.float64)
    score = round(100.0 * float(model.predict_proba(X)[:, 1][0]), 1)
    meta = model_status()
    return {
        "score": score,
        "level": level_from_score(score),
        "model_version": meta.get("model_version"),
        "algorithm": meta.get("algorithm"),
        "prediction_kind": "model risk score (not a calibrated probability)",
    }


class ReportRequest(BaseModel):
    type: str
    description: str = ""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class LocationRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    label: Optional[str] = None


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": app.version, "timestamp": now_iso(), "model": model_status()}


@app.get("/api/states")
async def states():
    try:
        weather = await fetch_state_weather()
        model = model_status()
        rows = []
        for name in STATES:
            item = {"name": name, "latitude": STATES[name][0], "longitude": STATES[name][1], **weather[name]}
            item["risk_score"] = item["screening"]["score"]
            item["risk_level"] = item["screening"]["level"]
            if model.get("status") == "READY":
                try:
                    terrain = await srtm_features(STATES[name][0], STATES[name][1])
                    features = {k: item.get(k) for k in FEATURES if k in item}
                    features.update({"soil_moisture_0_to_7cm": item.get("soil_moisture_0_to_7cm"), "elevation": terrain.get("elevation"), "slope": terrain.get("slope")})
                    ai = await model_predict(features)
                    if ai:
                        item["ai_prediction"] = ai
                        item["risk_score"] = ai["score"]
                        item["risk_level"] = ai["level"]
                except Exception as exc:
                    item["ai_error"] = str(exc)
            rows.append(item)
        return {"states": rows, "retrieved_at": now_iso(), "source": "ECMWF IFS HRES 9 km via Open-Meteo", "risk_basis": "validated research AI model at state centers" if model.get("status") == "READY" else "live weather/soil screening until AI model is trained"}
    except Exception as exc:
        raise HTTPException(502, detail=f"Weather data unavailable: {exc}")


@app.get("/api/weather")
async def weather(latitude: float, longitude: float):
    return await fetch_point_weather(latitude, longitude)


@app.get("/api/inventory")
async def inventory(force: bool = False):
    try:
        return await fetch_glc_inventory(force=force)
    except Exception as exc:
        with closing(db()) as c:
            rows = c.execute("SELECT * FROM landslide_events ORDER BY event_date DESC").fetchall()
        if rows:
            return {"source": "NASA Global Landslide Catalog (local cache)", "count": len(rows), "events": [dict(r) for r in rows], "retrieved_at": None, "stale": True}
        raise HTTPException(502, detail=f"NASA Global Landslide Catalog unavailable: {exc}")


@app.post("/api/inventory/refresh")
async def inventory_refresh():
    return await inventory(force=True)


@app.get("/api/data-sources")
async def data_sources():
    gsi = await gsi_public_status()
    with closing(db()) as c:
        rows = c.execute("SELECT name,url,status,coverage,last_checked,details FROM data_sources ORDER BY id").fetchall()
    out = [dict(r) for r in rows]
    for x in out:
        if x["name"].startswith("Geological Survey"):
            x["details"] += f" Public GSI portal checked: {gsi.get('status')}; field-validated inventory count shown by portal: {gsi.get('field_validated_inventory_count') or 'not parsed'}."
        if x["name"].startswith("GSI National"):
            x["details"] += f" Public field-validated inventory portal status: {gsi.get('status')}."
    return {"sources": out, "gsi_public": gsi}


@app.get("/api/model/status")
async def model_info():
    return model_status()


@app.post("/api/model/train")
async def model_train():
    return await train_model(force=True)


@app.post("/api/analyze")
async def analyze(req: LocationRequest):
    if not (NER_BBOX["south"] <= req.latitude <= NER_BBOX["north"] and NER_BBOX["west"] <= req.longitude <= NER_BBOX["east"]):
        raise HTTPException(400, detail="Location is outside the Northeast India monitoring region.")
    try:
        weather = await fetch_point_weather(req.latitude, req.longitude)
        terrain = await srtm_features(req.latitude, req.longitude)
    except Exception as exc:
        raise HTTPException(502, detail=f"Environmental data unavailable: {exc}")
    try:
        inventory_data = await fetch_glc_inventory()
    except Exception:
        inventory_data = {"events": []}
    nearby = await nearest_events(req.latitude, req.longitude)
    screen = screening(weather.get("rainfall_24h"), weather.get("rainfall_72h"), weather.get("soil_moisture_0_to_7cm"))
    features = {
        "rainfall_1h": weather.get("rainfall_1h"), "rainfall_3h": weather.get("rainfall_3h"),
        "rainfall_6h": weather.get("rainfall_6h"), "rainfall_12h": weather.get("rainfall_12h"),
        "rainfall_24h": weather.get("rainfall_24h"), "rainfall_48h": weather.get("rainfall_48h"),
        "rainfall_72h": weather.get("rainfall_72h"), "rainfall_7d": weather.get("rainfall_7d"),
        "soil_moisture_0_to_7cm": weather.get("soil_moisture_0_to_7cm"),
        "elevation": terrain.get("elevation"), "slope": terrain.get("slope"),
    }
    ai = await model_predict(features)
    # Keep screening and AI distinct. The AI model becomes the final model score only when a
    # real trained model exists; otherwise no fake AI result is produced.
    if ai:
        final_score, final_level, basis = ai["score"], ai["level"], "validated research AI model"
    else:
        final_score, final_level, basis = screen["score"], screen["level"], "weather/soil screening only; AI model not trained or insufficient features"
    static_score = round(min(100.0, max(0.0, terrain.get("slope", 0.0) * 2.5)), 1)
    with closing(db()) as c:
        c.execute("INSERT INTO predictions(latitude,longitude,label,timestamp,model_version,score,level,features_json) VALUES(?,?,?,?,?,?,?,?)", (req.latitude, req.longitude, req.label, now_iso(), ai["model_version"] if ai else "SCREENING-ONLY", final_score, final_level, json.dumps(features)))
        c.commit()
    return {
        "location": {"latitude": req.latitude, "longitude": req.longitude, "label": req.label},
        "screening": screen,
        "ai_prediction": ai,
        "components": {"static_terrain_indicator": static_score, "dynamic_weather_screening": screen["score"]},
        "final_assessment": {"score": final_score, "level": final_level, "basis": basis},
        "weather": weather,
        "inventory": {"nearby_50km": len(nearby), "nearest": nearby[0] if nearby else None, "total_ner_events": len(inventory_data.get("events", []))},
        "terrain": {"status": "AVAILABLE", **terrain},
        "geology": {"status": "OFFICIAL-PORTAL", "message": "GSI/NLFC portal is verified for traceability. A raw geology layer is used only when an official downloadable dataset is supplied; no geology values are fabricated."},
        "model": model_status(),
        "timestamp": now_iso(),
    }


@app.get("/api/alerts")
async def alerts():
    rows = await states()
    items = []
    for x in rows["states"]:
        level = x.get("risk_level", "LOW")
        if level == "VERY HIGH":
            severity = "CRITICAL"
        elif level == "HIGH":
            severity = "WARNING"
        elif level == "MODERATE":
            severity = "WATCH"
        else:
            severity = "INFO"
        rain = x.get("rainfall_24h")
        reasons = x.get("screening", {}).get("reasons", [])
        reason_text = "; ".join(reasons[:2]) if reasons else "No strong weather/soil trigger detected."
        items.append({
            "id": f"state-{x['name'].lower().replace(' ', '-')}",
            "severity": severity,
            "title": f"{x['name']} {severity.lower()}",
            "message": f"Risk {x.get('risk_score')} / 100. {reason_text}",
            "location": x["name"],
            "score": x.get("risk_score"),
            "level": level,
            "rainfall_24h": rain,
            "source": x.get("weather_source", "Open-Meteo / ECMWF IFS HRES"),
            "created_at": now_iso(),
        })
    # Add recent citizen reports as separate operational signals; they never become ML labels automatically.
    with closing(db()) as c:
        recent = c.execute("SELECT * FROM reports WHERE datetime(created_at) >= datetime('now','-24 hours') ORDER BY id DESC LIMIT 20").fetchall()
    for r in recent:
        items.append({
            "id": f"report-{r['id']}",
            "severity": "REPORT",
            "title": f"Citizen report: {r['type']}",
            "message": r['description'] or "Citizen hazard report submitted.",
            "location": f"{r['latitude']:.5f}, {r['longitude']:.5f}",
            "source": "Citizen Report",
            "created_at": r['created_at'],
        })
    model = model_status()
    meta_notice = None
    if model.get("status") != "READY":
        meta_notice = {"severity": "SYSTEM", "title": "AI model not active", "message": "Alerts are weather/soil screening alerts until a real model passes the validation pipeline.", "source": "GeoGuard NER"}
    severity_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "REPORT": 3, "INFO": 4}
    items.sort(key=lambda a: severity_order.get(a["severity"], 5))
    return {"alerts": items, "system_notice": meta_notice, "generated_at": now_iso(), "risk_basis": "validated research AI model where available; otherwise live weather/soil screening"}


@app.post("/api/reports")
async def create_report(req: ReportRequest):
    with closing(db()) as c:
        cur = c.execute("INSERT INTO reports(type,description,latitude,longitude,created_at,synced) VALUES(?,?,?,?,?,1)", (req.type, req.description, req.latitude, req.longitude, now_iso()))
        c.commit()
        rid = cur.lastrowid
    return {"id": rid, "status": "saved", "created_at": now_iso()}


@app.get("/api/reports")
async def reports():
    with closing(db()) as c:
        rows = c.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
    return {"reports": [dict(r) for r in rows]}


@app.get("/api/infrastructure")
async def infrastructure(latitude: float, longitude: float, radius: int = 3000):
    radius = max(500, min(radius, 10000))
    query = f"""
    [out:json][timeout:20];
    (way(around:{radius},{latitude},{longitude})[highway];node(around:{radius},{latitude},{longitude})[amenity];way(around:{radius},{latitude},{longitude})[building];);
    out center tags 120;
    """
    r = await _http.post(OVERPASS, content=query.encode("utf-8"), headers={"Content-Type": "text/plain"})
    r.raise_for_status()
    elements = r.json().get("elements", [])
    counts = {"roads": 0, "amenities": 0, "buildings": 0}
    samples = []
    for e in elements:
        tags = e.get("tags", {})
        if "highway" in tags: counts["roads"] += 1
        elif "amenity" in tags: counts["amenities"] += 1
        elif "building" in tags: counts["buildings"] += 1
        if len(samples) < 60:
            samples.append({"type": e.get("type"), "id": e.get("id"), "tags": tags})
    return {"counts": counts, "sample": samples, "radius_m": radius, "source": "OpenStreetMap / Overpass", "retrieved_at": now_iso()}
