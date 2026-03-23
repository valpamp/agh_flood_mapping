# Flood Mapping with Sentinel-1 SAR Imagery

**AGH University of Krakow — Aerospace Engineering**

A hands-on Jupyter Notebook tutorial that demonstrates a complete SAR-based flood mapping workflow: from downloading Sentinel-1 GRD images to producing a georeferenced binary flood map.

**Case study:** Storm Boris flooding in Nysa, Poland (September 2024).

## Prerequisites

1. **ESA SNAP (v12+)** with the Sentinel-1 Toolbox.
   Download: https://step.esa.int/main/download/snap-download/

2. **Free CDSE account** for downloading Sentinel-1 data.
   Register: https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration

3. **Python 3.10+** with the required packages (see below).

## Setup

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate flood_mapping

# Option B: pip
pip install -r requirements.txt
```

`esa-snappy` (the SNAP–Python bridge) should be installed automatically with SNAP 12+. If not:
```bash
pip install esa-snappy
```
See the [official configuration guide](https://senbox.atlassian.net/wiki/spaces/SNAP/pages/3114106881/) for troubleshooting.

## Repository Structure

```
agh_flood_mapping/
├── README.md
├── .gitignore
├── environment.yml
├── requirements.txt
├── aoi/
│   └── nysa_aoi.geojson       # Sample AOI (Nysa region, Poland)
├── src/
│   ├── __init__.py
│   └── cdse.py                 # CDSE search & download utilities
├── notebooks/
│   └── flood_mapping.ipynb     # Main tutorial notebook
├── data/                       # Downloaded Sentinel-1 products (gitignored)
└── output/                     # Processed outputs (gitignored)
```

## Workflow

| Step | Description | Tool |
|------|-------------|------|
| 1 | Define AOI (shapefile / WKT) | GeoPandas |
| 2 | Search Sentinel-1 GRD on CDSE | OData API |
| 3 | Download archive + crisis images | CDSE API |
| 4 | Subset to AOI | SNAP (`esa_snappy`) |
| 5 | Multilooking (speckle reduction) | SNAP |
| 6 | Radiometric calibration to σ⁰ | SNAP |
| 7 | Range-Doppler Terrain Correction | SNAP |
| 8 | Change detection + thresholding | NumPy |
| 9 | Polygonize flood mask | rasterio (GDAL) |
| 10 | Interactive visualization | leafmap |

## Author

Valerio Pampanoni — [valerio.pampanoni@uniroma1.it](mailto:valerio.pampanoni@uniroma1.it)
