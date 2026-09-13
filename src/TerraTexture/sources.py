"""
Open-data DEM sources: a generalised STAC query engine, plus thin
product-specific convenience methods on top of it, for public,
unauthenticated STAC catalogs -- currently the Polar Geospatial Center
(PGC: ArcticDEM covering the Arctic including Greenland, and REMA
covering Antarctica) and OpenTopography's raster DEM catalog (a large,
heterogeneous collection of global and regional DEM datasets).

This replaces an earlier version of this module that queried a private,
locally-installed STAC catalog (DEMSquad_STAC) via a hardcoded filesystem
path. Everything here instead talks to public STAC endpoints -- no
signup, no API key, no local software beyond `requests` (already a
dependency) -- so anyone can pull real open-source elevation data for an
AOI out of the box.

Two STAC access patterns, two engines
--------------------------------------
STAC catalogs come in two practically-different flavours, and a single
query function can't serve both correctly:

1. **Dynamic STAC APIs** (e.g. PGC's, https://stac.pgc.umn.edu/api/v1)
   implement the STAC API Item Search extension: a `POST /search` you
   can hand a bbox + collection list, and the server does the spatial
   filtering. ``stac_search()`` handles this.

2. **Static (or search-less) STAC catalogs** (e.g. OpenTopography's,
   https://portal.opentopography.org/stac/raster_catalog.json) are just
   a crawlable tree of Catalog -> Collection -> Item JSON documents with
   no `/search` endpoint at all -- there may be dozens to hundreds of
   collections (one per dataset), each with its own `items` link.
   ``list_stac_collections()`` lets a caller discover/choose *which*
   collection to query (the "ask the user which collection" step), and
   ``stac_collection_items()`` fetches that collection's items, passing
   `bbox=` as a best-effort server-side hint but ALWAYS re-filtering by
   bbox intersection client-side afterwards -- so it's correct whether
   or not the server actually honours the query parameter.

PGC-specific knowledge (base URL, collection-ID naming, which asset key
holds the DEM) lives only in the thin product wrappers
(``arcticdem_mosaic()``, ``rema_mosaic()``); OpenTopography-specific
defaults live only in ``opentopography_dem_urls()``/``opentopography_mosaic()``.
Both call the shared engines above -- point either engine at a different
STAC endpoint entirely and it works the same way for any other public
STAC catalog.

The DEM assets on both PGC's and OpenTopography's catalogs are plain
HTTPS Cloud-Optimized GeoTIFFs (not tar.gz archives), so the URLs
returned here are handed straight to ``io.load_dem_mosaic()`` --
rasterio/GDAL read HTTPS COGs transparently, so no downloading or
extraction step is needed.
"""

from urllib.parse import urljoin

from .io import load_dem, load_dem_mosaic

PGC_STAC_API = "https://stac.pgc.umn.edu/api/v1"
OT_STAC_ROOT = "https://portal.opentopography.org/stac/raster_catalog.json"

# Current mosaic product version per PGC's published collections, e.g.
# "arcticdem-mosaics-v4.1-32m". Override via the `version=` argument on
# the product functions below if PGC publishes a newer version later.
_MOSAIC_VERSION = {
    "arcticdem": "4.1",
    "rema": "2.0",
}

# Resolutions PGC actually publishes mosaic collections at, in meters.
_MOSAIC_RESOLUTIONS = (2, 10, 32)

_session = None


def _get_session():
    """Shared requests.Session for every STAC HTTP call in this module.

    Every call site here previously used the bare `requests.get`/`post`
    module functions, each opening a brand-new TCP connection (and, for
    HTTPS, a fresh TLS handshake) even when hitting the same host
    repeatedly -- e.g. opentopography_dem_urls() does catalog fetch ->
    collection fetch -> items fetch, three round trips to the same host.
    A shared Session pools and reuses connections across all of them.
    """
    global _session
    if _session is None:
        import requests

        _session = requests.Session()
    return _session


def stac_search(
    collections,
    bbox,
    api_url=PGC_STAC_API,
    datetime=None,
    max_items=None,
    page_size=100,
    timeout=30,
    extra_params=None,
):
    """Generic STAC API item search with pagination.

    Works against any STAC API 1.0-compliant endpoint (PGC's is the
    motivating case here, but nothing below is PGC-specific) -- this is
    the reusable "query engine"; dataset-specific details belong in a
    thin wrapper that calls this, not in here.

    Parameters
    ----------
    collections : sequence of str
        STAC collection IDs to search within.
    bbox : sequence of 4 floats
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 -- the STAC API
        spec requires the search bbox in WGS84 lon/lat regardless of the
        data's own native CRS.
    api_url : str
        Base URL of the STAC API (its `/search` endpoint is POSTed to).
    datetime : str or None
        Optional STAC datetime filter, e.g. "2020-01-01/2023-12-31".
    max_items : int or None
        Stop once this many items have been collected. None = no limit
        (follow pagination until the API reports no more results).
    page_size : int
        Items requested per page (STAC API "limit" parameter).
    timeout : float
        Per-request timeout in seconds.
    extra_params : dict or None
        Additional STAC API search body fields (e.g. {"query": {...}}
        for attribute filtering), merged into the request body.

    Returns
    -------
    list of dict
        Raw STAC Item objects (GeoJSON Features) matching the search.
    """
    session = _get_session()

    search_url = api_url.rstrip("/") + "/search"
    body = {"collections": list(collections), "bbox": list(bbox), "limit": page_size}
    if datetime is not None:
        body["datetime"] = datetime
    if extra_params:
        body.update(extra_params)

    items = []
    request = {"method": "POST", "url": search_url, "json": body}

    while request is not None:
        if request["method"] == "POST":
            response = session.post(request["url"], json=request["json"], timeout=timeout)
        else:
            response = session.get(request["url"], params=request.get("params"), timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        items.extend(payload.get("features", []))
        if max_items is not None and len(items) >= max_items:
            return items[:max_items]

        next_link = next(
            (link for link in payload.get("links", []) if link.get("rel") == "next"), None
        )
        if next_link is None:
            request = None
        elif next_link.get("method", "GET").upper() == "POST":
            request = {"method": "POST", "url": next_link["href"], "json": next_link.get("body", body)}
        else:
            request = {"method": "GET", "url": next_link["href"], "params": None}

    return items


def _item_asset_href(item, asset_key):
    """Pull an asset's href off a STAC Item, with a helpful error listing
    what asset keys actually exist if the requested one isn't there
    (asset naming varies by collection -- see module docstring)."""
    try:
        return item["assets"][asset_key]["href"]
    except KeyError as exc:
        available = sorted(item.get("assets", {}).keys())
        item_id = item.get("id", "<unknown>")
        raise KeyError(
            f"Asset '{asset_key}' not found on STAC item '{item_id}'. "
            f"Available assets: {available}"
        ) from exc


def _reproject_bbox_to_4326(bounds, src_crs):
    """STAC bbox search parameters must be WGS84 lon/lat regardless of the
    data's native CRS -- reproject the caller's AOI bounds if needed."""
    if str(src_crs).upper() in ("EPSG:4326", "OGC:CRS84", "WGS84"):
        return tuple(bounds)
    from rasterio.warp import transform_bounds

    return transform_bounds(src_crs, "EPSG:4326", *bounds)


def _bbox_intersects(item_bbox, query_bbox):
    """True if two (min_lon, min_lat, max_lon, max_lat) boxes overlap."""
    return not (
        item_bbox[2] < query_bbox[0] or item_bbox[0] > query_bbox[2]
        or item_bbox[3] < query_bbox[1] or item_bbox[1] > query_bbox[3]
    )


def _resolve_href(href, base_url):
    """Static STAC catalogs commonly use relative hrefs (e.g. "./collection.json")
    -- resolve against the URL they were found on. No-op for absolute URLs."""
    return urljoin(base_url, href)


def list_stac_collections(catalog_url, timeout=30):
    """Fetch a STAC Catalog's root document and list its child Collections
    -- the "ask the user which collection" step for catalogs like
    OpenTopography's that host a large, heterogeneous set of datasets
    rather than a handful of well-known ones.

    Works against any STAC Catalog root (static or dynamic): it just
    follows `rel="child"` links, which every STAC Catalog exposes
    regardless of whether it also supports search.

    Parameters
    ----------
    catalog_url : str
        URL of the root Catalog document (e.g. `OT_STAC_ROOT`).
    timeout : float

    Returns
    -------
    list of dict
        One {"id", "title", "href"} dict per child collection. `href` is
        already resolved to an absolute URL (relative hrefs are common in
        static catalogs) -- pass it straight to `stac_collection_items()`.
    """

    session = _get_session()
    response = session.get(catalog_url, timeout=timeout)
    response.raise_for_status()
    catalog = response.json()

    collections = []
    for link in catalog.get("links", []):
        if link.get("rel") == "child":
            href = _resolve_href(link["href"], catalog_url)
            collections.append({
                "id": link.get("title") or href,
                "title": link.get("title"),
                "href": href,
            })
    return collections


def describe_stac_collection(collection_href, timeout=30):
    """Fetch a single Collection's full document (id, description, extent,
    etc.) -- use after `list_stac_collections()` to inspect a candidate
    before committing to querying its items, since the child-link title
    alone is often not very descriptive."""
    session = _get_session()
    response = session.get(collection_href, timeout=timeout)
    response.raise_for_status()
    return response.json()


def stac_collection_items(
    collection,
    bbox=None,
    bbox_crs="EPSG:4326",
    max_items=None,
    page_size=250,
    timeout=30,
):
    """Fetch items from a STAC Collection's `items` link -- the
    generalised query engine for static (or search-less) STAC catalogs,
    as a counterpart to `stac_search()` for dynamic ones.

    `bbox` is passed as a best-effort `?bbox=` query parameter (many
    catalogs, even nominally "static" ones, are served by software that
    honours OGC API - Features' basic bbox filter on `/items` even
    without full Item Search support) -- but the results are ALWAYS
    re-filtered by bbox intersection client-side afterwards too, using
    each item's own `bbox` field, so this is correct regardless of
    whether the server actually applied the filter.

    Parameters
    ----------
    collection : str or dict
        Either a Collection href (as returned by `list_stac_collections()`
        or any URL to a collection.json/collection endpoint), or an
        already-fetched collection dict with that shape (must include
        "href" or be the full collection document with a "links" list).
    bbox : sequence of 4 floats or None
        (min_lon, min_lat, max_lon, max_lat) in `bbox_crs`. None fetches
        every item in the collection -- only do this for small
        collections, since there's no server-side limit applied.
    bbox_crs : str
        CRS of `bbox`. Reprojected to EPSG:4326 if not already WGS84.
    max_items : int or None
        Stop once this many items have been collected.
    page_size : int
        Items requested per page, where the server supports paging.
    timeout : float

    Returns
    -------
    list of dict
        Raw STAC Item objects (GeoJSON Features) intersecting `bbox`.
    """
    session = _get_session()

    if isinstance(collection, str):

        response = session.get(collection, timeout=timeout)
        response.raise_for_status()
        collection_doc = response.json()
        collection_url = collection
    else:
        collection_url = collection.get("href", "")
        if "links" in collection:
            collection_doc = collection
        else:
            # a {"id", "title", "href"} summary from list_stac_collections()
            # -- fetch the real document to get its "items" link
            response = session.get(collection_url, timeout=timeout)
            response.raise_for_status()
            collection_doc = response.json()

    items_link = next(
        (link["href"] for link in collection_doc.get("links", []) if link.get("rel") == "items"), None
    )
    if items_link is None:
        raise ValueError(f"Collection at '{collection_url}' has no 'items' link.")
    items_link = _resolve_href(items_link, collection_url)

    bbox_4326 = _reproject_bbox_to_4326(bbox, bbox_crs) if bbox is not None else None

    items = []
    url = items_link
    params = {"limit": page_size}
    if bbox_4326 is not None:
        params["bbox"] = ",".join(str(v) for v in bbox_4326)

    while url is not None:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        page_items = payload.get("features", [])

        if bbox_4326 is not None:
            page_items = [
                item for item in page_items
                if "bbox" not in item or _bbox_intersects(item["bbox"], bbox_4326)
            ]
        items.extend(page_items)
        if max_items is not None and len(items) >= max_items:
            return items[:max_items]

        next_link = next((link for link in payload.get("links", []) if link.get("rel") == "next"), None)
        if next_link is None:
            url = None
        else:
            # a "next" link's href is normally a complete, self-contained
            # URL (including its own query string), so don't re-send params
            url = _resolve_href(next_link["href"], url)
            params = None

    return items


def opentopography_dem_urls(
    collection,
    bounds,
    bbox_crs="EPSG:4326",
    asset_key="data",
    catalog_url=OT_STAC_ROOT,
    max_items=None,
):
    """DEM asset URLs from an OpenTopography raster collection
    intersecting `bounds`.

    OpenTopography hosts a large, heterogeneous set of raster DEM
    datasets (global products like SRTM/COP30/NASADEM alongside many
    regional lidar-derived DEMs) as a static STAC catalog with no
    search endpoint -- use `list_stac_collections(OT_STAC_ROOT)` first
    to see what's available and let the caller/user pick one, then pass
    its id or href here.

    Parameters
    ----------
    collection : str
        Either a collection `id`/title as returned by
        `list_stac_collections(OT_STAC_ROOT)`, or a direct collection
        href/URL.
    bounds : sequence of 4 floats
        AOI bounds in `bbox_crs`.
    bbox_crs : str
        CRS of `bounds`.
    asset_key : str
        Which item asset holds the DEM. Defaults to "data", a common
        convention, but this varies by collection -- if it's wrong you'll
        get a KeyError listing the actual available asset keys for that
        collection's items, so it's self-correcting on the first try.
    catalog_url : str
        Root catalog URL. Override only if OpenTopography restructures
        or mirrors their catalog elsewhere.
    max_items : int or None

    Returns
    -------
    list of str
        HTTPS URLs to the requested asset for each intersecting item.
    """
    if not str(collection).startswith(("http://", "https://")):
        matches = [c for c in list_stac_collections(catalog_url) if c["id"] == collection]
        if not matches:
            available = [c["id"] for c in list_stac_collections(catalog_url)]
            raise ValueError(
                f"No OpenTopography collection matches '{collection}'. "
                f"Available: {available}"
            )
        collection = matches[0]["href"]

    items = stac_collection_items(collection, bbox=bounds, bbox_crs=bbox_crs, max_items=max_items)
    return [_item_asset_href(item, asset_key) for item in items]


def opentopography_mosaic(
    collection,
    bounds,
    bbox_crs="EPSG:4326",
    asset_key="data",
    catalog_url=OT_STAC_ROOT,
    target_crs=None,
    max_items=None,
):
    """Fetch and merge OpenTopography DEM tiles from `collection`
    intersecting `bounds`. Returns (dem, cellsize, transform, crs); raises
    ValueError if no tiles intersect the given bounds. See
    `opentopography_dem_urls()` for parameters."""
    urls = opentopography_dem_urls(
        collection, bounds, bbox_crs=bbox_crs, asset_key=asset_key,
        catalog_url=catalog_url, max_items=max_items,
    )
    if not urls:
        raise ValueError(
            f"No items in OpenTopography collection '{collection}' intersect "
            f"the given bounds ({bounds} in {bbox_crs})."
        )
    return load_dem_mosaic(urls, target_crs=target_crs)


def _mosaic_collection_id(product, version, resolution):
    return f"{product}-mosaics-v{version}-{resolution}m"


def mosaic_dem_urls(
    product,
    bounds,
    resolution=32,
    version=None,
    bbox_crs="EPSG:4326",
    api_url=PGC_STAC_API,
    asset_key="dem",
    max_items=None,
):
    """Query PGC's public STAC API for `product` mosaic tiles intersecting
    `bounds`, returning the list of (HTTPS, Cloud-Optimized) DEM asset
    URLs -- ready to hand to `io.load_dem_mosaic()`.

    This is the shared implementation behind `arcticdem_mosaic_urls()` and
    `rema_mosaic_urls()`; call those directly unless you're adding support
    for a third PGC product that follows the same
    "{product}-mosaics-v{version}-{resolution}m" collection naming.

    Parameters
    ----------
    product : {"arcticdem", "rema"}
    bounds : sequence of 4 floats
        AOI bounds in `bbox_crs`.
    resolution : int
        Mosaic resolution in meters -- PGC publishes 2, 10, and 32.
    version : str or None
        Mosaic product version (e.g. "4.1" for ArcticDEM, "2.0" for
        REMA). Defaults to the current version per `_MOSAIC_VERSION`.
    bbox_crs : str
        CRS of `bounds`. Reprojected to EPSG:4326 for the STAC query if
        not already WGS84.
    api_url : str
        STAC API base URL. Override only if pointing at a mirror.
    asset_key : str
        Which item asset to pull the URL from. "dem" (the default) is
        the actual elevation COG; other useful keys on PGC mosaics
        include "hillshade", "count", "mad" (median absolute deviation),
        "mindate"/"maxdate".
    max_items : int or None
        Cap on the number of STAC items (tiles) returned.

    Returns
    -------
    list of str
        HTTPS URLs to the requested asset for each intersecting tile.
    """
    if resolution not in _MOSAIC_RESOLUTIONS:
        raise ValueError(
            f"resolution={resolution} not published for {product} mosaics; "
            f"PGC provides {_MOSAIC_RESOLUTIONS}"
        )
    if version is None:
        version = _MOSAIC_VERSION[product]

    bbox_4326 = _reproject_bbox_to_4326(bounds, bbox_crs)
    collection = _mosaic_collection_id(product, version, resolution)
    items = stac_search([collection], bbox_4326, api_url=api_url, max_items=max_items)
    return [_item_asset_href(item, asset_key) for item in items]


def arcticdem_mosaic_urls(bounds, resolution=32, version="4.1", bbox_crs="EPSG:4326", max_items=None):
    """ArcticDEM mosaic DEM tile URLs intersecting `bounds` (covers the
    Arctic, including Greenland). See `mosaic_dem_urls()` for parameters.
    """
    return mosaic_dem_urls(
        "arcticdem", bounds, resolution=resolution, version=version,
        bbox_crs=bbox_crs, max_items=max_items,
    )


def rema_mosaic_urls(bounds, resolution=32, version="2.0", bbox_crs="EPSG:4326", max_items=None):
    """REMA (Reference Elevation Model of Antarctica) mosaic DEM tile URLs
    intersecting `bounds`. See `mosaic_dem_urls()` for parameters."""
    return mosaic_dem_urls(
        "rema", bounds, resolution=resolution, version=version,
        bbox_crs=bbox_crs, max_items=max_items,
    )


def arcticdem_mosaic(bounds, resolution=32, version="4.1", bbox_crs="EPSG:4326", target_crs=None, max_items=None):
    """Fetch and merge ArcticDEM mosaic tiles intersecting `bounds` from
    PGC's public STAC API -- a fully public, no-signup replacement for the
    old private/local DEMSquad_STAC-based lookup.

    Returns the same 4-tuple as `io.load_dem_mosaic()`:
    (dem, cellsize, transform, crs).

    Raises ValueError if no tiles intersect the given bounds.
    """
    urls = arcticdem_mosaic_urls(
        bounds, resolution=resolution, version=version, bbox_crs=bbox_crs, max_items=max_items,
    )
    if not urls:
        raise ValueError(
            "No ArcticDEM mosaic tiles intersect the given bounds "
            f"({bounds} in {bbox_crs})."
        )
    return load_dem_mosaic(urls, target_crs=target_crs, bounds=bounds, bounds_crs=bbox_crs)


def rema_mosaic(bounds, resolution=32, version="2.0", bbox_crs="EPSG:4326", target_crs=None, max_items=None):
    """Fetch and merge REMA mosaic tiles intersecting `bounds` from PGC's
    public STAC API. Returns (dem, cellsize, transform, crs); raises
    ValueError if no tiles intersect the given bounds."""
    urls = rema_mosaic_urls(
        bounds, resolution=resolution, version=version, bbox_crs=bbox_crs, max_items=max_items,
    )
    if not urls:
        raise ValueError(
            f"No REMA mosaic tiles intersect the given bounds ({bounds} in {bbox_crs})."
        )
    return load_dem_mosaic(urls, target_crs=target_crs, bounds=bounds, bounds_crs=bbox_crs)


def make_demo_geotiff(
    out_path="/tmp/demo_dem.tif",
    bounds_4326=(-3.25, 54.42, -3.10, 54.53),  # west, south, east, north (Lake District, UK)
):
    """Write the synthetic DEM from load_dem() to a real GeoTIFF covering
    `bounds_4326`, purely so plot_dem_basemap_luminosity_relief() has
    something real-world-georeferenced to fetch imagery for. Elevation
    VALUES are still fake -- only the footprint is real. Use your own DEM
    file, or arcticdem_mosaic()/rema_mosaic() above, for real work."""
    import rasterio
    from rasterio.transform import from_bounds

    dem, _ = load_dem(None)
    west, south, east, north = bounds_4326
    transform = from_bounds(west, south, east, north, dem.shape[1], dem.shape[0])

    with rasterio.open(
        out_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype=dem.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    return out_path
