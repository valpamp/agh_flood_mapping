"""
Copernicus Data Space Ecosystem (CDSE) utilities for searching and downloading
Sentinel-1 products via the OData API.

CDSE OData API documentation:
    https://documentation.dataspace.copernicus.eu/APIs/OData.html

Account registration (free):
    https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration
"""

import requests
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def get_access_token(username: str, password: str) -> str:
    """
    Obtain a short-lived access token from the CDSE identity provider
    using the OAuth2 Resource Owner Password Credentials grant.

    The token is valid for ~600 seconds.

    Parameters
    ----------
    username : str
        CDSE account e-mail.
    password : str
        CDSE account password.

    Returns
    -------
    str
        Bearer access token.
    """
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_sentinel1_grd(
    wkt: str,
    start_date: str,
    end_date: str,
    orbit_direction: str | None = None,
    #sensor_mode: str = "IW",
    product_type: str = "IW_GRDH_1S",
    max_results: int = 50,
) -> list[dict]:
    """
    Query the CDSE OData catalogue for Sentinel-1 Level-1 GRD products.

    Parameters
    ----------
    wkt : str
        Well-Known Text geometry (POLYGON or POINT) in EPSG:4326.
    start_date : str
        ISO-8601 start date, e.g. ``"2024-09-01T00:00:00.000Z"``.
    end_date : str
        ISO-8601 end date.
    orbit_direction : str, optional
        ``"ASCENDING"`` or ``"DESCENDING"``.  If *None*, both are returned.
    sensor_mode : str
        Acquisition mode (default ``"IW"`` – Interferometric Wide Swath).
    product_type : str
        Product type code (default ``"IW_GRDH_1S"``).
    max_results : int
        Maximum number of products to return (default 50).

    Returns
    -------
    list[dict]
        List of product metadata dictionaries as returned by the OData API.
    """
    filters = [
        "Collection/Name eq 'SENTINEL-1'",
        f"ContentDate/Start gt {start_date}",
        f"ContentDate/Start lt {end_date}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
        (
            "Attributes/OData.CSC.StringAttribute/any("
            f"att:att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        ),
        # (
        #     "Attributes/OData.CSC.StringAttribute/any("
        #     f"att:att/Name eq 'sensorMode' and "
        #     f"att/OData.CSC.StringAttribute/Value eq '{sensor_mode}')"
        # ),
    ]

    if orbit_direction is not None:
        filters.append(
            "Attributes/OData.CSC.StringAttribute/any("
            f"att:att/Name eq 'orbitDirection' and "
            f"att/OData.CSC.StringAttribute/Value eq '{orbit_direction}')"
        )

    params = {
        "$filter": " and ".join(filters),
        "$orderby": "ContentDate/Start asc",
        "$top": max_results,
    }

    response = requests.get(CATALOGUE_URL, params=params)
    response.raise_for_status()
    return response.json()["value"]


def print_search_results(products: list[dict]) -> None:
    """Pretty-print a list of products returned by :func:`search_sentinel1_grd`."""
    print(f"Found {len(products)} product(s):\n")
    for i, p in enumerate(products):
        start = p["ContentDate"]["Start"]
        print(f"  [{i}] {p['Name']}")
        print(f"      Sensing start : {start}")
        print(f"      ID            : {p['Id']}")
        print(f"      Size          : {p.get('ContentLength', 'N/A')} bytes")
        print()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_product(
    product_id: str,
    filename: str,
    access_token: str,
    output_dir: str | Path = "data",
) -> Path:
    """
    Download a single product from CDSE.

    Parameters
    ----------
    product_id : str
        UUID of the product (the ``Id`` field from search results).
    filename : str
        Name to save the file as (typically the product name + ``.zip``).
    access_token : str
        Valid CDSE bearer token from :func:`get_access_token`.
    output_dir : str or Path
        Directory to save the downloaded file.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if out_path.exists():
        print(f"  File already exists, skipping: {out_path.name}")
        return out_path

    url = (
        f"https://download.dataspace.copernicus.eu/odata/v1/"
        f"Products({product_id})/$value"
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # The CDSE download endpoint may return a 302 redirect
    session = requests.Session()
    session.headers.update(headers)

    response = session.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        desc=f"  {filename}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"  Saved to {out_path}")
    return out_path
