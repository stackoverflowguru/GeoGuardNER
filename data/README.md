# GeoGuard NER data layer

## Primary landslide inventory
The application now connects to the published NASA Global Landslide Catalog (GLC) ArcGIS FeatureServer for the Northeast India project envelope at runtime.

NASA GLC public service used by the backend:
`https://maps.nccs.nasa.gov/server/rest/services/global_landslide_catalog/landslide_points/FeatureServer/0/query`

The NASA Open Data Portal also provides a one-time GLC CSV export; NASA states that export is current to 7 March 2016 and is maintained separately from the original catalog service. Do not present that historical export as a complete post-2016 inventory.

## Weather
Open-Meteo historical/forecast weather endpoint is used for live rainfall and soil moisture screening. Open-Meteo documents precipitation and soil-moisture variables and reanalysis coverage.

## Terrain
SRTM 1 Arc-Second Global is the planned terrain source. It provides roughly 30 m elevation data globally. The current build does not fabricate terrain values; the SRTM processor is the next ML-data step.

## Geology
Use official Geological Survey of India / NGDR spatial data where a usable public dataset is available. The application does not invent geology values.

## Infrastructure
OpenStreetMap Overpass is queried on demand for nearby roads, amenities and buildings.
