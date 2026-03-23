"""
Flood Mapping with Sentinel-1 SAR Imagery — Test Script
========================================================

This script mirrors the Jupyter notebook (notebooks/flood_mapping.ipynb)
step-by-step.  Use it to verify the full processing chain before updating
the notebook.

Case study: Storm Boris — Nysa, Poland (September 2024)

Usage
-----
    cd agh_flood_mapping
    python scripts/flood_mapping.py              # uses defaults
    python scripts/flood_mapping.py --download    # also search & download from CDSE
"""

import sys
import argparse
import getpass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for testing
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape
from shapely import wkt as shapely_wkt
import rasterio
from rasterio.features import shapes as rio_shapes

# ── project paths ────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
from cdse import get_access_token, search_sentinel1_grd, print_search_results, download_product

AOI_PATH   = PROJECT_DIR / "aoi" / "nysa_aoi.geojson"
DATA_DIR   = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── processing parameters ───────────────────────────────────────────────────
SPECKLE_FILTER      = "Refined Lee"   # "Refined Lee", "Lee Sigma", "Lee", "Gamma Map", "Frost", "Boxcar"
FILTER_WINDOW       = "7x7"           # "3x3" … "17x17"
DEM_NAME            = "SRTM 3Sec"
PIXEL_SPACING_M     = 20.0            # output pixel spacing after terrain correction
THRESHOLD_CHANGE_DB = 3.0             # minimum dB drop to flag as flood
THRESHOLD_WATER_DB  = -15.0           # maximum crisis σ⁰ (dB) to confirm water


# ==========================================================================
# 1. Define the Area of Interest
# ==========================================================================
def load_aoi(aoi_path: Path) -> tuple[gpd.GeoDataFrame, str]:
    """Load AOI from a vector file and return (GeoDataFrame, WKT string)."""
    aoi_gdf = gpd.read_file(aoi_path)
    aoi_wkt = aoi_gdf.geometry.iloc[0].wkt
    print(f"AOI WKT: {aoi_wkt}")
    print(f"AOI bounds: {aoi_gdf.geometry.iloc[0].bounds}")
    return aoi_gdf, aoi_wkt


# ==========================================================================
# 2–3. Search & Download from CDSE  (optional, only with --download)
# ==========================================================================
def search_and_download(aoi_wkt: str) -> tuple[Path, Path]:
    """
    Search CDSE for archive/crisis Sentinel-1 GRD products and download them.

    Returns the paths to the two zip files (archive, crisis).
    """
    products = search_sentinel1_grd(
        wkt=aoi_wkt,
        start_date="2024-09-01T00:00:00.000Z",
        end_date="2024-09-22T00:00:00.000Z",
        orbit_direction="ASCENDING",
        product_type="IW_GRDH_1S",
    )
    print_search_results(products)

    # Select archive (pre-event) and crisis (co-event) images
    ARCHIVE_IDX = 0
    CRISIS_IDX  = 1
    archive_product = products[ARCHIVE_IDX]
    crisis_product  = products[CRISIS_IDX]

    print(f"Archive: {archive_product['Name']}  ({archive_product['ContentDate']['Start']})")
    print(f"Crisis:  {crisis_product['Name']}  ({crisis_product['ContentDate']['Start']})")

    # Authenticate
    cdse_user = input("CDSE username (e-mail): ")
    cdse_pass = getpass.getpass("CDSE password: ")
    token = get_access_token(cdse_user, cdse_pass)
    del cdse_pass
    print("Authentication successful.")

    archive_zip = download_product(
        product_id=archive_product["Id"],
        filename=archive_product["Name"] + ".zip",
        access_token=token,
        output_dir=DATA_DIR,
    )
    crisis_zip = download_product(
        product_id=crisis_product["Id"],
        filename=crisis_product["Name"] + ".zip",
        access_token=token,
        output_dir=DATA_DIR,
    )
    return archive_zip, crisis_zip


def find_existing_data() -> tuple[Path, Path]:
    """Find the archive and crisis zip files already present in DATA_DIR."""
    zips = sorted(DATA_DIR.glob("S1*.zip"))
    if len(zips) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 Sentinel-1 zips in {DATA_DIR}, found {len(zips)}. "
            "Run with --download to fetch them from CDSE."
        )
    # The earlier date is the archive, the later is the crisis
    print(f"Archive: {zips[0].name}")
    print(f"Crisis:  {zips[1].name}")
    return zips[0], zips[1]


# ==========================================================================
# 4. Load products into SNAP
# ==========================================================================
def load_snap():
    """Import esa_snappy modules (triggers JVM start)."""
    from esa_snappy import ProductIO, GPF, HashMap, WKTReader
    print(f"esa_snappy loaded.  SNAP registry: {GPF.getDefaultInstance().getOperatorSpiRegistry()}")
    return ProductIO, GPF, HashMap, WKTReader


def print_product_info(product, label: str):
    """Print basic information about a SNAP product."""
    w = product.getSceneRasterWidth()
    h = product.getSceneRasterHeight()
    bands = list(product.getBandNames())
    print(f"  [{label}]  {product.getName()}  —  {w}×{h}  bands={bands}")


# ==========================================================================
# 5. Subset to AOI
# ==========================================================================
def snap_subset(GPF, HashMap, source_product, wkt_aoi, polarisation="Amplitude_VV"):
    """Subset a SNAP product to the given WKT geometry (single band)."""
    params = HashMap()
    params.put("geoRegion", wkt_aoi)
    params.put("bandNames", polarisation)
    params.put("copyMetadata", True)
    return GPF.createProduct("Subset", params, source_product)


# ==========================================================================
# 6. Speckle filtering
# ==========================================================================
def snap_speckle_filter(GPF, HashMap, source_product,
                        filter_name="Refined Lee", window_size="7x7"):
    """
    Apply an adaptive speckle filter using SNAP's Speckle-Filter operator.

    The 'Refined Lee' filter is direction-aware and edge-preserving,
    making it the best general-purpose choice for flood mapping:
    it suppresses speckle in open-water areas while keeping the sharp
    boundary between flooded fields and roads.

    Note: jpy maps Python int to Java Long by default; SNAP's filterSizeX/Y
    parameters expect Java Integer, so we use jpy to cast explicitly.
    """
    import jpy
    Integer = jpy.get_type("java.lang.Integer")

    size_x, size_y = window_size.split("x")
    params = HashMap()
    params.put("filter",              filter_name)
    params.put("filterSizeX",         Integer(int(size_x)))
    params.put("filterSizeY",         Integer(int(size_y)))
    params.put("dampingFactor",       Integer(2))  # used by Frost only
    params.put("estimateENL",         True)        # auto-estimate Equivalent Number of Looks
    params.put("enl",                 1.0)         # fallback ENL
    params.put("numLooksStr",         "1")         # for Lee Sigma
    params.put("windowSize",          "7x7")       # secondary window for Lee Sigma
    params.put("targetWindowSizeStr", "3x3")       # target window for Lee Sigma
    params.put("sigmaStr",            "0.9")       # sigma for Lee Sigma
    return GPF.createProduct("Speckle-Filter", params, source_product)


# ==========================================================================
# 7. Radiometric calibration
# ==========================================================================
def snap_calibrate(GPF, HashMap, source_product):
    """Radiometric calibration to σ⁰ (linear scale)."""
    params = HashMap()
    params.put("outputSigmaBand", True)
    params.put("outputImageScaleInDb", False)
    return GPF.createProduct("Calibration", params, source_product)


# ==========================================================================
# 8. Terrain correction
# ==========================================================================
def snap_terrain_correction(GPF, HashMap, source_product,
                            dem_name="SRTM 3Sec", pixel_spacing_m=20.0):
    """
    Range-Doppler Terrain Correction → WGS 84.

    Projects the image from SAR slant-range geometry to a geographic
    coordinate system using a DEM for orthorectification.
    """
    params = HashMap()
    params.put("demName",              dem_name)
    params.put("imgResamplingMethod",  "BILINEAR_INTERPOLATION")
    params.put("pixelSpacingInMeter",  pixel_spacing_m)
    params.put("mapProjection",        "WGS84(DD)")
    params.put("saveSelectedSourceBand", True)
    params.put("nodataValueAtSea",     True)
    return GPF.createProduct("Terrain-Correction", params, source_product)


# ==========================================================================
# 9. Export to GeoTIFF
# ==========================================================================
def export_geotiff(ProductIO, product, output_path: Path):
    """Write a SNAP product to GeoTIFF."""
    # ProductIO appends .tif, so strip the suffix
    ProductIO.writeProduct(product, str(output_path.with_suffix("")), "GeoTIFF")
    print(f"  Saved: {output_path}")


# ==========================================================================
# 10. Convert to dB
# ==========================================================================
def to_db(linear_array: np.ndarray) -> np.ndarray:
    """Convert linear backscatter values to decibels."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(linear_array)
    db[~np.isfinite(db)] = np.nan
    return db


# ==========================================================================
# 11–12. Change detection & threshold-based flood classification
# ==========================================================================
def detect_flood(archive_db: np.ndarray, crisis_db: np.ndarray,
                 threshold_change_db: float = 3.0,
                 threshold_water_db: float = -15.0) -> np.ndarray:
    """
    Binary flood detection via change detection.

    A pixel is classified as flooded when:
      Δσ⁰ = archive_db − crisis_db  >  threshold_change_db   AND
      crisis_db  <  threshold_water_db

    Returns a uint8 mask (1 = flood, 0 = no flood).
    """
    delta_db = archive_db - crisis_db
    flood_mask = (
        (delta_db > threshold_change_db) &
        (crisis_db < threshold_water_db) &
        np.isfinite(delta_db)
    ).astype(np.uint8)

    flooded_px = flood_mask.sum()
    total_valid = np.isfinite(delta_db).sum()
    print(f"  Flooded pixels : {flooded_px:,}")
    print(f"  Total valid    : {total_valid:,}")
    print(f"  Flooded frac.  : {flooded_px / total_valid * 100:.2f}%")
    return flood_mask


# ==========================================================================
# 13. Save flood mask as GeoTIFF
# ==========================================================================
def save_flood_mask(flood_mask: np.ndarray, reference_profile: dict,
                    output_path: Path):
    """Write the binary flood mask as a compressed GeoTIFF."""
    profile = reference_profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        height=flood_mask.shape[0],
        width=flood_mask.shape[1],
        compress="lzw",
        nodata=255,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(flood_mask, 1)
    print(f"  Flood mask saved: {output_path}")


# ==========================================================================
# 14. Polygonize & export
# ==========================================================================
def polygonize_flood(flood_tif: Path) -> gpd.GeoDataFrame:
    """Convert the raster flood mask to vector polygons."""
    with rasterio.open(flood_tif) as src:
        data = src.read(1)
        mask = data == 1
        transform = src.transform
        crs = src.crs

    polygons = [shape(geom) for geom, val in
                rio_shapes(data, mask=mask, connectivity=8, transform=transform)
                if val == 1]

    print(f"  Flood polygons: {len(polygons)}")

    flood_gdf = gpd.GeoDataFrame(
        {"class": ["flood"] * len(polygons)},
        geometry=polygons,
        crs=crs,
    )

    # Compute areas in metric CRS
    flood_gdf_metric = flood_gdf.to_crs(epsg=32633)   # UTM 33N
    flood_gdf["area_m2"] = flood_gdf_metric.geometry.area
    flood_gdf["area_ha"] = flood_gdf["area_m2"] / 10_000

    total_ha = flood_gdf["area_ha"].sum()
    print(f"  Total flooded area: {total_ha:.1f} ha  ({total_ha / 100:.2f} km²)")
    return flood_gdf


# ==========================================================================
# 15. Plots (saved to output/)
# ==========================================================================
def save_plots(archive_db, crisis_db, flood_mask, output_dir: Path):
    """Generate and save diagnostic plots."""

    # Crop to common extent
    min_r = min(archive_db.shape[0], crisis_db.shape[0])
    min_c = min(archive_db.shape[1], crisis_db.shape[1])
    a = archive_db[:min_r, :min_c]
    c = crisis_db[:min_r, :min_c]

    vmin, vmax = -25, 5

    # --- SAR images side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(a, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Archive (pre-event) — σ⁰ VV [dB]")
    axes[0].axis("off")
    axes[1].imshow(c, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Crisis (co-event) — σ⁰ VV [dB]")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "sar_images.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: sar_images.png")

    # --- Histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in [(axes[0], a, "Archive"), (axes[1], c, "Crisis")]:
        valid = data[np.isfinite(data)]
        ax.hist(valid.ravel(), bins=200, color="steelblue", edgecolor="none", alpha=0.7)
        ax.set_xlabel("σ⁰ [dB]"); ax.set_ylabel("Pixel count")
        ax.set_title(f"{title} σ⁰ VV [dB]"); ax.set_xlim(-30, 10)
    plt.tight_layout()
    fig.savefig(output_dir / "histograms.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: histograms.png")

    # --- False-color RGB composite ---
    def normalize(arr, lo=-25, hi=0):
        out = (arr - lo) / (hi - lo)
        return np.clip(np.nan_to_num(out, nan=0.0), 0, 1)
    rgb = np.stack([normalize(a), normalize(c), normalize(c)], axis=-1)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb)
    ax.set_title("False-Color Composite: R=Archive, G=Crisis, B=Crisis\n(Red = flooded areas)")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "false_color_composite.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: false_color_composite.png")

    # --- Change map ---
    delta = a - c
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-10, vmax=10)
    ax.set_title("Δσ⁰ = Archive − Crisis [dB]")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Δσ⁰ [dB]", shrink=0.7)
    plt.tight_layout()
    fig.savefig(output_dir / "change_map.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: change_map.png")

    # --- Change histogram ---
    valid_delta = delta[np.isfinite(delta)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid_delta.ravel(), bins=300, color="steelblue", edgecolor="none", alpha=0.7)
    ax.axvline(x=3.0, color="red", ls="--", lw=2, label="Threshold = 3 dB")
    ax.axvline(x=5.0, color="orange", ls="--", lw=2, label="Threshold = 5 dB")
    ax.set_xlabel("Δσ⁰ [dB]"); ax.set_ylabel("Pixel count")
    ax.set_title("Histogram of σ⁰ change (archive − crisis)")
    ax.set_xlim(-15, 20); ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "change_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: change_histogram.png")

    # --- Flood overlay ---
    fm = flood_mask[:min_r, :min_c]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(c, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Crisis image σ⁰ VV [dB]")
    axes[0].axis("off")
    axes[1].imshow(c, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = np.ma.masked_where(fm == 0, fm)
    axes[1].imshow(overlay, cmap="autumn_r", alpha=0.6)
    axes[1].set_title(f"Detected flood (Δσ⁰>{THRESHOLD_CHANGE_DB} dB & σ⁰<{THRESHOLD_WATER_DB} dB)")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "flood_overlay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: flood_overlay.png")


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="SAR flood mapping — test script")
    parser.add_argument("--download", action="store_true",
                        help="Search & download products from CDSE (requires account)")
    args = parser.parse_args()

    # ── 1. AOI ────────────────────────────────────────────────────────────
    print("\n══ 1. Loading AOI ══")
    aoi_gdf, aoi_wkt = load_aoi(AOI_PATH)

    # ── 2–3. Data acquisition ─────────────────────────────────────────────
    if args.download:
        print("\n══ 2–3. Searching & downloading from CDSE ══")
        archive_zip, crisis_zip = search_and_download(aoi_wkt)
    else:
        print("\n══ 2–3. Using existing data ══")
        archive_zip, crisis_zip = find_existing_data()

    # ── 4. Load into SNAP ─────────────────────────────────────────────────
    print("\n══ 4. Loading products into SNAP ══")
    ProductIO, GPF, HashMap, WKTReader = load_snap()

    archive_snap = ProductIO.readProduct(str(archive_zip))
    crisis_snap  = ProductIO.readProduct(str(crisis_zip))
    print_product_info(archive_snap, "Archive raw")
    print_product_info(crisis_snap,  "Crisis raw")

    # ── 5. Subset ─────────────────────────────────────────────────────────
    print("\n══ 5. Subsetting to AOI ══")
    archive_sub = snap_subset(GPF, HashMap, archive_snap, aoi_wkt)
    crisis_sub  = snap_subset(GPF, HashMap, crisis_snap, aoi_wkt)
    print_product_info(archive_sub, "Archive subset")
    print_product_info(crisis_sub,  "Crisis subset")

    # ── 6. Speckle filtering ──────────────────────────────────────────────
    print(f"\n══ 6. Speckle filtering ({SPECKLE_FILTER}, {FILTER_WINDOW}) ══")
    archive_spk = snap_speckle_filter(GPF, HashMap, archive_sub,
                                      filter_name=SPECKLE_FILTER,
                                      window_size=FILTER_WINDOW)
    crisis_spk  = snap_speckle_filter(GPF, HashMap, crisis_sub,
                                      filter_name=SPECKLE_FILTER,
                                      window_size=FILTER_WINDOW)
    print_product_info(archive_spk, "Archive speckle-filtered")
    print_product_info(crisis_spk,  "Crisis speckle-filtered")

    # ── 7. Calibration ────────────────────────────────────────────────────
    print("\n══ 7. Radiometric calibration (σ⁰) ══")
    archive_cal = snap_calibrate(GPF, HashMap, archive_spk)
    crisis_cal  = snap_calibrate(GPF, HashMap, crisis_spk)
    print_product_info(archive_cal, "Archive calibrated")
    print_product_info(crisis_cal,  "Crisis calibrated")

    # ── 8. Terrain correction ─────────────────────────────────────────────
    print(f"\n══ 8. Terrain correction ({DEM_NAME}, {PIXEL_SPACING_M} m) ══")
    archive_tc = snap_terrain_correction(GPF, HashMap, archive_cal,
                                         dem_name=DEM_NAME,
                                         pixel_spacing_m=PIXEL_SPACING_M)
    crisis_tc  = snap_terrain_correction(GPF, HashMap, crisis_cal,
                                         dem_name=DEM_NAME,
                                         pixel_spacing_m=PIXEL_SPACING_M)
    print_product_info(archive_tc, "Archive TC")
    print_product_info(crisis_tc,  "Crisis TC")

    # ── 9. Export to GeoTIFF ──────────────────────────────────────────────
    archive_tif = OUTPUT_DIR / "archive_sigma0_VV_TC.tif"
    crisis_tif  = OUTPUT_DIR / "crisis_sigma0_VV_TC.tif"

    print("\n══ 9. Exporting to GeoTIFF ══")
    export_geotiff(ProductIO, archive_tc, archive_tif)
    export_geotiff(ProductIO, crisis_tc,  crisis_tif)

    # ── 10. Load rasters & convert to dB ──────────────────────────────────
    print("\n══ 10. Loading rasters & converting to dB ══")
    with rasterio.open(archive_tif) as src:
        archive_sigma0 = src.read(1)
        archive_profile = src.profile.copy()
    with rasterio.open(crisis_tif) as src:
        crisis_sigma0 = src.read(1)

    archive_db = to_db(archive_sigma0)
    crisis_db  = to_db(crisis_sigma0)
    print(f"  Archive: {archive_db.shape}  Crisis: {crisis_db.shape}")

    # Crop to common extent
    min_r = min(archive_db.shape[0], crisis_db.shape[0])
    min_c = min(archive_db.shape[1], crisis_db.shape[1])
    archive_crop = archive_db[:min_r, :min_c]
    crisis_crop  = crisis_db[:min_r, :min_c]

    # ── 11–12. Flood detection ────────────────────────────────────────────
    print(f"\n══ 11–12. Flood detection (Δσ⁰>{THRESHOLD_CHANGE_DB} dB, σ⁰<{THRESHOLD_WATER_DB} dB) ══")
    flood_mask = detect_flood(archive_crop, crisis_crop,
                              threshold_change_db=THRESHOLD_CHANGE_DB,
                              threshold_water_db=THRESHOLD_WATER_DB)

    # ── 13. Save flood mask ───────────────────────────────────────────────
    flood_tif = OUTPUT_DIR / "flood_mask.tif"
    print("\n══ 13. Saving flood mask ══")
    save_flood_mask(flood_mask, archive_profile, flood_tif)

    # ── 14. Polygonize ────────────────────────────────────────────────────
    print("\n══ 14. Polygonizing ══")
    flood_gdf = polygonize_flood(flood_tif)

    flood_geojson = OUTPUT_DIR / "flood_polygons.geojson"
    flood_shp     = OUTPUT_DIR / "flood_polygons.shp"
    flood_gdf.to_file(flood_geojson, driver="GeoJSON")
    flood_gdf.to_file(flood_shp, driver="ESRI Shapefile")
    print(f"  Exported: {flood_geojson.name}, {flood_shp.name}")

    # ── 15. Plots ─────────────────────────────────────────────────────────
    print("\n══ 15. Generating plots ══")
    save_plots(archive_db, crisis_db, flood_mask, OUTPUT_DIR)

    print("\n══ Done ══")


if __name__ == "__main__":
    main()
