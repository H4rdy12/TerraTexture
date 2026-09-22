"""
Open-data DEM sources: a generic STAC query engine plus product wrappers.

Finds and loads real elevation data for an area of interest (AOI) from
public, unauthenticated STAC catalogs. There is no signup, no API key and
no locally installed catalog software. Two providers are wired up:

- **PGC** (Polar Geospatial Center): ArcticDEM (the Arctic, including
  Greenland) and REMA (Antarctica) mosaics, served by a dynamic STAC API
  at https://stac.pgc.umn.edu/api/v1.
- **OpenTopography**: a large, heterogeneous raster DEM catalog (global
  products such as SRTM, COP30 and NASADEM alongside many regional
  lidar-derived DEMs), served as a static STAC catalog at
  https://portal.opentopography.org/stac/raster_catalog.json.

This replaces an earlier version that queried a private, locally
installed catalog (DEMSquad_STAC) through a hardcoded filesystem path.

Two STAC access patterns, two engines:

    Dynamic STAC API (PGC)            Static catalog (OpenTopography)
    ----------------------            -------------------------------
    POST /search {bbox, collections}  root catalog.json
      -> server filters by bbox         -> child links (one per dataset)
      -> paginated FeatureCollection    -> collection.json -> "items" link
                                          -> paginated items, bbox
                                             re-filtered client-side
    engine: stac_search()             engines: list_stac_collections()
                                               stac_collection_items()

:func:`stac_collection_items` passes ``bbox`` to the server as a
best-effort hint but always re-filters by bbox intersection client-side,
so it is correct whether or not the server honours the parameter.

Provider-specific knowledge (base URLs, PGC's collection naming, which
asset key holds the DEM) lives only in the thin wrappers:
:func:`arcticdem_mosaic`, :func:`rema_mosaic`,
:func:`opentopography_dem_urls` and :func:`opentopography_mosaic`. Point
either engine at any other public STAC endpoint and it works the same way.

The DEM assets on both catalogs are plain HTTPS Cloud Optimized GeoTIFFs
(not ``.tar.gz`` archives), so returned URLs go straight to
:func:`TerraTexture.io.load_dem_mosaic`. GDAL reads remote COGs directly,
so there is no download or extraction step.

Error handling:
    Every HTTP failure -- connection errors, timeouts, non-2xx responses,
    responses that aren't JSON -- is raised as :class:`STACError`, which
    carries the URL and HTTP status. Transient failures (HTTP 429 and
    5xx) are retried automatically with exponential backoff before that
    happens. Bad arguments raise ``ValueError`` before any request is
    sent.

Dependencies:
    ``requests`` for all HTTP calls and ``rasterio`` for reprojecting
    non-WGS84 bounds and loading tiles. Both are imported lazily, so
    importing this module never fails; calling a function that needs a
    missing package raises ``ImportError`` naming the extra to install.

Examples:
    ArcticDEM around Ilulissat, Greenland, at 32 m::

        dem, cellsize, transform, crs = arcticdem_mosaic(
            bounds=(-51.3, 69.1, -50.9, 69.3),
        )

    Browse OpenTopography, then load one collection for an AOI::

        collections = list_stac_collections(OT_STAC_ROOT)
        print([c["id"] for c in collections][:10])
        dem, cellsize, transform, crs = opentopography_mosaic(
            collections[0]["id"], bounds=(-3.25, 54.42, -3.10, 54.53),
        )
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from .io import load_dem, load_dem_mosaic

if TYPE_CHECKING:
    import requests
    from affine import Affine
    from rasterio.crs import CRS

    import numpy as np
    import numpy.typing as npt

    # (min_x, min_y, max_x, max_y); lon/lat when in EPSG:4326.
    BBox = tuple[float, float, float, float]
    # (dem, cellsize, transform, crs), as returned by load_dem_mosaic().
    MosaicResult = tuple[npt.NDArray[np.float32], float, Affine, Any]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PGC_STAC_API = "https://stac.pgc.umn.edu/api/v1"
OT_STAC_ROOT = "https://portal.opentopography.org/stac/raster_catalog.json"

# Current mosaic product version per PGC's published collections, e.g.
# "arcticdem-mosaics-v4.1-32m". Override with `version=` on the product
# functions if PGC publishes a newer version.
_MOSAIC_VERSION: dict[str, str] = {
    "arcticdem": "4.1",
    "rema": "2.0",
}

# Resolutions (metres) PGC publishes mosaic collections at.
_MOSAIC_RESOLUTIONS = (2, 10, 32)

# CRS strings that already mean WGS84 lon/lat (no reprojection needed).
_WGS84_ALIASES = ("EPSG:4326", "OGC:CRS84", "WGS84")

# Safety net against servers whose "next" links never terminate.
_MAX_PAGES = 1000

# Transient HTTP statuses worth retrying (rate limit + server errors).
_RETRY_STATUSES = (429, 500, 502, 503, 504)

_session: requests.Session | None = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class STACError(RuntimeError):
    """
    Raised when a STAC request fails or returns something unusable.

    Covers connection errors, timeouts, non-2xx responses (after
    retries), non-JSON bodies and JSON that isn't an object. The original
    ``requests`` exception, if any, is chained as ``__cause__``.

    Attributes:
        url (str | None): URL of the failing request.
        status_code (int | None): HTTP status, if a response arrived.
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        """
        Initialise the error.

        Args:
            message (str): Human-readable description of the failure.
            url (str | None): URL of the failing request.
            status_code (int | None): HTTP status, if a response arrived.
        """
        super().__init__(message)
        self.url = url
        self.status_code = status_code


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """
    Return the shared ``requests.Session`` used by every call here.

    A shared session pools connections, so repeated calls to the same
    host (e.g. :func:`opentopography_dem_urls` does catalog -> collection
    -> items) reuse one TCP/TLS connection instead of opening a new one
    each time. It also retries HTTP 429 / 5xx responses and connection
    failures up to 3 times with exponential backoff (0.5 s, 1 s, 2 s),
    honouring any ``Retry-After`` header. POST is retried too: STAC
    ``/search`` is read-only, so repeating it is safe.

    Returns:
        requests.Session: The lazily created shared session.

    Raises:
        ImportError: If ``requests`` is not installed.
    """
    global _session
    if _session is not None:
        return _session

    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise ImportError(
            "Querying STAC catalogs needs `requests`. Install it with "
            "`uv add requests` (or `pip install requests`)."
        ) from exc

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _session = session
    return session


def _request_json(
    method: str,
    url: str,
    timeout: float,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Send one HTTP request and return the decoded JSON object.

    Args:
        method (str): ``"GET"`` or ``"POST"``.
        url (str): Absolute URL to request.
        timeout (float): Timeout in seconds for connect and read.
        params (dict[str, Any] | None): Query-string parameters.
        json_body (dict[str, Any] | None): JSON body (POST only).

    Returns:
        dict[str, Any]: The decoded JSON object.

    Raises:
        ImportError: If ``requests`` is not installed.
        STACError: On connection failure, timeout, a non-2xx status
            (after retries), a non-JSON body, or JSON that isn't an
            object.
    """
    import requests

    session = _get_session()
    try:
        response = session.request(
            method, url, params=params, json=json_body, timeout=timeout
        )
    except requests.Timeout as exc:
        raise STACError(
            f"{method} {url} timed out after {timeout} s", url=url
        ) from exc
    except requests.RequestException as exc:
        raise STACError(f"{method} {url} failed: {exc}", url=url) from exc

    if not response.ok:
        snippet = response.text[:300].strip()
        raise STACError(
            f"{method} {response.url} returned HTTP {response.status_code} "
            f"{response.reason}" + (f": {snippet}" if snippet else ""),
            url=response.url,
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise STACError(
            f"{method} {response.url} did not return JSON "
            f"(Content-Type: {response.headers.get('Content-Type')})",
            url=response.url,
            status_code=response.status_code,
        ) from exc

    if not isinstance(payload, dict):
        raise STACError(
            f"{method} {response.url} returned JSON "
            f"{type(payload).__name__}, expected an object",
            url=response.url,
            status_code=response.status_code,
        )
    return payload


# ---------------------------------------------------------------------------
# Validation and small helpers
# ---------------------------------------------------------------------------

def _validate_bbox(
    bbox: Sequence[float],
    name: str = "bbox",
    allow_antimeridian: bool = False,
) -> BBox:
    """
    Check that a bounding box is four finite, correctly ordered numbers.

    Args:
        bbox (Sequence[float]): Candidate ``(min_x, min_y, max_x, max_y)``.
        name (str): Argument name, used in error messages.
        allow_antimeridian (bool): Allow ``min_x > max_x``, which the STAC
            spec uses for boxes crossing the antimeridian. Only safe
            where the server does the filtering.

    Returns:
        BBox: The bounds as a tuple of floats.

    Raises:
        ValueError: If ``bbox`` is not four finite numbers, has
            ``min_y >= max_y``, or has ``min_x >= max_x`` when
            antimeridian crossing is not allowed.
    """
    try:
        min_x, min_y, max_x, max_y = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be four numbers (min_x, min_y, max_x, max_y); "
            f"got {bbox!r}"
        ) from exc

    if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
        raise ValueError(f"{name} must be finite; got {bbox!r}")
    if min_y >= max_y:
        raise ValueError(f"{name} needs min_y < max_y; got {bbox!r}")
    if min_x == max_x or (min_x > max_x and not allow_antimeridian):
        raise ValueError(
            f"{name} needs min_x < max_x; got {bbox!r}. Boxes crossing "
            "the antimeridian are not supported here -- split into two "
            "queries instead."
        )
    return min_x, min_y, max_x, max_y


def _validate_positive(name: str, value: float | None) -> None:
    """
    Check that an optional numeric argument is positive.

    Args:
        name (str): Argument name, used in the error message.
        value (float | None): Value to check. ``None`` is accepted.

    Returns:
        None

    Raises:
        ValueError: If ``value`` is not ``None`` and is not > 0.
    """
    if value is not None and not value > 0:
        raise ValueError(f"{name} must be positive; got {value!r}")


def _find_link(doc: dict[str, Any], rel: str) -> dict[str, Any] | None:
    """
    Return the first link in a STAC document with the given ``rel``.

    Args:
        doc (dict[str, Any]): STAC Catalog, Collection or page document.
        rel (str): Link relation to look for, e.g. ``"next"``.

    Returns:
        dict[str, Any] | None: The link object, or ``None`` if absent or
            if the link has no ``href``.
    """
    for link in doc.get("links") or []:
        if isinstance(link, dict) and link.get("rel") == rel and link.get("href"):
            return link
    return None


def _item_asset_href(item: dict[str, Any], asset_key: str) -> str:
    """
    Return the ``href`` of one asset on a STAC Item.

    Asset naming varies by collection, so on failure the error lists the
    asset keys that do exist -- one failed call tells you the right key.

    Args:
        item (dict[str, Any]): STAC Item (GeoJSON Feature).
        asset_key (str): Asset key to look up, e.g. ``"dem"``.

    Returns:
        str: The asset's URL.

    Raises:
        KeyError: If the item has no such asset, or the asset has no
            ``href``. The message lists the available asset keys.
    """
    try:
        return item["assets"][asset_key]["href"]
    except (KeyError, TypeError) as exc:
        assets = item.get("assets") if isinstance(item, dict) else None
        available = sorted(assets) if isinstance(assets, dict) else []
        item_id = item.get("id", "<unknown>") if isinstance(item, dict) else "?"
        raise KeyError(
            f"Asset '{asset_key}' not found on STAC item '{item_id}'. "
            f"Available assets: {available}"
        ) from exc


def _reproject_bbox_to_4326(bounds: Sequence[float], src_crs: str) -> BBox:
    """
    Reproject AOI bounds to WGS84 lon/lat, as STAC bbox queries require.

    Args:
        bounds (Sequence[float]): ``(min_x, min_y, max_x, max_y)`` in
            ``src_crs``.
        src_crs (str): CRS of ``bounds``.

    Returns:
        BBox: Bounds in EPSG:4326. Unchanged if ``src_crs`` is already
            WGS84.

    Raises:
        ImportError: If reprojection is needed and rasterio is missing.
        ValueError: If ``src_crs`` is not a valid CRS.
    """
    if str(src_crs).upper() in _WGS84_ALIASES:
        return tuple(float(v) for v in bounds)

    try:
        from rasterio.errors import CRSError
        from rasterio.warp import transform_bounds
    except ImportError as exc:
        raise ImportError(
            f"Reprojecting bounds from {src_crs} needs rasterio; install "
            "the `raster` extra, or pass bounds in EPSG:4326."
        ) from exc

    try:
        return tuple(transform_bounds(src_crs, "EPSG:4326", *bounds))
    except CRSError as exc:
        raise ValueError(
            f"Cannot reproject bounds: invalid CRS {src_crs!r} ({exc})"
        ) from exc


def _bbox_intersects(item_bbox: Sequence[float], query_bbox: BBox) -> bool:
    """
    Return whether two lon/lat bounding boxes overlap (touching counts).

    Handles both 2-D ``[w, s, e, n]`` and 3-D ``[w, s, zmin, e, n, zmax]``
    item bboxes, as allowed by the STAC spec.

    Args:
        item_bbox (Sequence[float]): The item's ``bbox`` field.
        query_bbox (BBox): Query box in EPSG:4326.

    Returns:
        bool: ``True`` if the boxes overlap.

    Raises:
        ValueError: If ``item_bbox`` has neither 4 nor 6 values.
    """
    if len(item_bbox) == 6:
        west, south, _, east, north, _ = item_bbox
    elif len(item_bbox) == 4:
        west, south, east, north = item_bbox
    else:
        raise ValueError(f"Unexpected item bbox length: {item_bbox!r}")

    return not (
        east < query_bbox[0] or west > query_bbox[2]
        or north < query_bbox[1] or south > query_bbox[3]
    )


def _resolve_href(href: str, base_url: str) -> str:
    """
    Resolve a possibly relative href against the URL it was found on.

    Static STAC catalogs often use relative hrefs such as
    ``"./collection.json"``. Absolute URLs pass through unchanged.

    Args:
        href (str): Link target from a STAC document.
        base_url (str): URL of the document the link came from.

    Returns:
        str: An absolute URL.
    """
    return urljoin(base_url, href)


# ---------------------------------------------------------------------------
# Engine 1: dynamic STAC APIs (POST /search)
# ---------------------------------------------------------------------------

def stac_search(
    collections: Sequence[str],
    bbox: Sequence[float],
    api_url: str = PGC_STAC_API,
    datetime: str | None = None,
    max_items: int | None = None,
    page_size: int = 100,
    timeout: float = 30,
    extra_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Search a STAC API for items, following pagination.

    Works against any STAC API 1.0 endpoint with Item Search. PGC's is the
    motivating case, but nothing here is PGC-specific: dataset details
    belong in a thin wrapper that calls this.

    Pagination follows each page's ``rel="next"`` link, by GET or POST as
    the link specifies. A repeated ``next`` link, or more than
    ``_MAX_PAGES`` pages, stops the loop with a warning instead of
    looping forever.

    Args:
        collections (Sequence[str]): STAC collection IDs to search.
        bbox (Sequence[float]): ``(min_lon, min_lat, max_lon, max_lat)``
            in EPSG:4326. The STAC API spec requires WGS84 here whatever
            the data's native CRS. ``min_lon > max_lon`` is allowed and
            means the box crosses the antimeridian.
        api_url (str): Base URL of the STAC API; ``/search`` is appended.
        datetime (str | None): Optional STAC datetime filter, e.g.
            ``"2020-01-01/2023-12-31"``.
        max_items (int | None): Stop after this many items. ``None``
            follows pagination to the end.
        page_size (int): Items per page (the STAC ``limit`` parameter).
        timeout (float): Per-request timeout in seconds.
        extra_params (dict[str, Any] | None): Extra search-body fields,
            e.g. ``{"query": {...}}`` for attribute filtering. Merged into
            the body and override the defaults above.

    Returns:
        list[dict[str, Any]]: Raw STAC Items (GeoJSON Features).

    Raises:
        ValueError: If ``collections`` is empty or a bare string, or
            ``bbox``, ``max_items``, ``page_size`` or ``timeout`` are
            invalid.
        STACError: If any request fails or returns invalid JSON.
        ImportError: If ``requests`` is not installed.

    Examples:
        >>> items = stac_search(
        ...     ["arcticdem-mosaics-v4.1-32m"], (-51.3, 69.1, -50.9, 69.3)
        ... )
    """
    if isinstance(collections, str):
        raise ValueError(
            f"collections must be a list of IDs, got the string "
            f"{collections!r}; wrap it: [{collections!r}]"
        )
    collections = list(collections)
    if not collections:
        raise ValueError("collections must contain at least one ID")
    bbox = _validate_bbox(bbox, allow_antimeridian=True)
    _validate_positive("max_items", max_items)
    _validate_positive("page_size", page_size)
    _validate_positive("timeout", timeout)

    body: dict[str, Any] = {
        "collections": collections,
        "bbox": list(bbox),
        "limit": page_size,
    }
    if datetime is not None:
        body["datetime"] = datetime
    if extra_params:
        body.update(extra_params)

    method = "POST"
    url = api_url.rstrip("/") + "/search"
    request_body: dict[str, Any] | None = body
    seen: set[str] = set()
    items: list[dict[str, Any]] = []

    for page in range(1, _MAX_PAGES + 1):
        key = f"{method} {url} {json.dumps(request_body, sort_keys=True)}"
        if key in seen:
            logger.warning(
                "STAC search at %s returned a repeated 'next' link; "
                "stopping after %d items.", url, len(items),
            )
            return items
        seen.add(key)

        if method == "POST":
            payload = _request_json("POST", url, timeout, json_body=request_body)
        else:
            payload = _request_json("GET", url, timeout)

        items.extend(payload.get("features") or [])
        logger.debug("Page %d from %s: %d items so far", page, url, len(items))
        if max_items is not None and len(items) >= max_items:
            return items[:max_items]

        next_link = _find_link(payload, "next")
        if next_link is None:
            return items

        url = _resolve_href(next_link["href"], url)
        method = str(next_link.get("method", "GET")).upper()
        if method == "POST":
            # STAC's pagination extension: "merge": true means the link's
            # body only holds changes (e.g. a token) on top of ours.
            link_body = next_link.get("body")
            if link_body is None:
                request_body = body
            elif next_link.get("merge"):
                request_body = {**body, **link_body}
            else:
                request_body = link_body
        else:
            request_body = None

    logger.warning(
        "STAC search hit the %d-page safety limit; returning %d items.",
        _MAX_PAGES, len(items),
    )
    return items


# ---------------------------------------------------------------------------
# Engine 2: static / search-less STAC catalogs
# ---------------------------------------------------------------------------

def list_stac_collections(
    catalog_url: str,
    timeout: float = 30,
) -> list[dict[str, str | None]]:
    """
    List the child collections of a STAC Catalog.

    This is the "ask the user which collection" step for catalogs like
    OpenTopography's, which host many heterogeneous datasets rather than
    a handful of well-known ones. It follows ``rel="child"`` links, which
    every STAC Catalog exposes whether or not it supports search.

    Args:
        catalog_url (str): URL of the root Catalog, e.g.
            :data:`OT_STAC_ROOT`.
        timeout (float): Request timeout in seconds.

    Returns:
        list[dict[str, str | None]]: One dict per child collection:

            - ``id`` (str): The link title, or the href if untitled.
            - ``title`` (str | None): The link title.
            - ``href`` (str): Absolute URL of the collection, ready for
              :func:`stac_collection_items`.

            Empty if the catalog has no child links (a warning is
            logged).

    Raises:
        ValueError: If ``timeout`` is not positive.
        STACError: If the request fails or returns invalid JSON.
        ImportError: If ``requests`` is not installed.
    """
    _validate_positive("timeout", timeout)
    catalog = _request_json("GET", catalog_url, timeout)

    collections: list[dict[str, str | None]] = []
    for link in catalog.get("links") or []:
        if not isinstance(link, dict) or link.get("rel") != "child":
            continue
        if not link.get("href"):
            logger.warning("Skipping child link without href in %s", catalog_url)
            continue
        href = _resolve_href(link["href"], catalog_url)
        collections.append({
            "id": link.get("title") or href,
            "title": link.get("title"),
            "href": href,
        })

    if not collections:
        logger.warning("Catalog %s lists no child collections.", catalog_url)
    return collections


def describe_stac_collection(
    collection_href: str,
    timeout: float = 30,
) -> dict[str, Any]:
    """
    Fetch one Collection's full document (id, description, extent, ...).

    Use after :func:`list_stac_collections` to inspect a candidate
    before querying its items; the child-link title alone is often not
    very descriptive.

    Args:
        collection_href (str): URL of the Collection document.
        timeout (float): Request timeout in seconds.

    Returns:
        dict[str, Any]: The Collection document.

    Raises:
        ValueError: If ``timeout`` is not positive.
        STACError: If the request fails or returns invalid JSON.
        ImportError: If ``requests`` is not installed.
    """
    _validate_positive("timeout", timeout)
    return _request_json("GET", collection_href, timeout)


def _resolve_collection_doc(
    collection: str | dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], str]:
    """
    Turn any accepted ``collection`` argument into ``(document, base_url)``.

    Args:
        collection (str | dict[str, Any]): A collection URL, a summary
            dict from :func:`list_stac_collections`, or a full Collection
            document.
        timeout (float): Request timeout in seconds.

    Returns:
        tuple[dict[str, Any], str]: The Collection document and the URL
            to resolve its relative links against.

    Raises:
        TypeError: If ``collection`` is neither a string nor a dict.
        ValueError: If a dict has neither ``links`` nor ``href``, or a
            full document has no usable base URL for relative links.
        STACError: If fetching the document fails.
    """
    if isinstance(collection, str):
        return _request_json("GET", collection, timeout), collection

    if not isinstance(collection, dict):
        raise TypeError(
            "collection must be a URL string or a dict, got "
            f"{type(collection).__name__}"
        )

    if "links" in collection:
        self_link = _find_link(collection, "self")
        base_url = collection.get("href") or (
            self_link["href"] if self_link else ""
        )
        return collection, base_url

    href = collection.get("href")
    if not href:
        raise ValueError(
            "collection dict needs an 'href' (as returned by "
            "list_stac_collections()) or a 'links' list (a full "
            f"Collection document); got keys {sorted(collection)}"
        )
    return _request_json("GET", href, timeout), href


def stac_collection_items(
    collection: str | dict[str, Any],
    bbox: Sequence[float] | None = None,
    bbox_crs: str = "EPSG:4326",
    max_items: int | None = None,
    page_size: int = 250,
    timeout: float = 30,
) -> list[dict[str, Any]]:
    """
    Fetch a Collection's items, filtered to a bounding box.

    The engine for static or search-less catalogs; the counterpart to
    :func:`stac_search`. ``bbox`` is sent as a best-effort ``?bbox=``
    parameter (many "static" catalogs are served by software that
    supports OGC API - Features' basic bbox filter on ``/items``), and
    results are *always* re-filtered client-side by each item's own
    ``bbox``. Correct whether or not the server applied the filter.

    Args:
        collection (str | dict[str, Any]): A Collection URL, a summary
            dict from :func:`list_stac_collections`, or an already
            fetched Collection document.
        bbox (Sequence[float] | None): ``(min_x, min_y, max_x, max_y)`` in
            ``bbox_crs``. ``None`` fetches every item -- only sensible for
            small collections.
        bbox_crs (str): CRS of ``bbox``; reprojected to EPSG:4326.
        max_items (int | None): Stop after this many matching items.
        page_size (int): Items per page, where the server pages.
        timeout (float): Per-request timeout in seconds.

    Returns:
        list[dict[str, Any]]: Raw STAC Items intersecting ``bbox``. Items
            without a ``bbox`` field are kept (they can't be ruled out),
            and a warning reports how many there were.

    Raises:
        TypeError: If ``collection`` is not a string or dict.
        ValueError: If arguments are invalid, the bbox crosses the
            antimeridian, or the Collection has no ``items`` link.
        STACError: If any request fails or returns invalid JSON.
        ImportError: If ``requests`` (or, for a non-WGS84 ``bbox_crs``,
            rasterio) is not installed.
    """
    _validate_positive("max_items", max_items)
    _validate_positive("page_size", page_size)
    _validate_positive("timeout", timeout)

    bbox_4326: BBox | None = None
    if bbox is not None:
        _validate_bbox(bbox)
        bbox_4326 = _validate_bbox(
            _reproject_bbox_to_4326(bbox, bbox_crs), name="bbox (in EPSG:4326)"
        )

    collection_doc, base_url = _resolve_collection_doc(collection, timeout)
    items_link = _find_link(collection_doc, "items")
    if items_link is None:
        raise ValueError(
            f"Collection at '{base_url or '<unknown>'}' has no 'items' link."
        )
    url: str | None = _resolve_href(items_link["href"], base_url)

    params: dict[str, Any] | None = {"limit": page_size}
    if bbox_4326 is not None:
        params["bbox"] = ",".join(str(v) for v in bbox_4326)

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    unfiltered = 0

    for _ in range(_MAX_PAGES):
        if url is None:
            break
        if url in seen:
            logger.warning(
                "Repeated 'next' link %s; stopping after %d items.",
                url, len(items),
            )
            break
        seen.add(url)

        payload = _request_json("GET", url, timeout, params=params)
        page_items = payload.get("features") or []

        if bbox_4326 is not None:
            kept = []
            for item in page_items:
                item_bbox = item.get("bbox")
                if not item_bbox:
                    unfiltered += 1
                    kept.append(item)
                    continue
                try:
                    if _bbox_intersects(item_bbox, bbox_4326):
                        kept.append(item)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping item %s with malformed bbox: %s",
                        item.get("id", "<unknown>"), exc,
                    )
            page_items = kept

        items.extend(page_items)
        if max_items is not None and len(items) >= max_items:
            items = items[:max_items]
            break

        next_link = _find_link(payload, "next")
        # A "next" href is normally a complete URL with its own query
        # string, so the original params are not re-sent.
        url = _resolve_href(next_link["href"], url) if next_link else None
        params = None
    else:
        logger.warning(
            "Hit the %d-page safety limit; returning %d items.",
            _MAX_PAGES, len(items),
        )

    if unfiltered:
        logger.warning(
            "%d item(s) had no bbox and were kept without spatial "
            "filtering.", unfiltered,
        )
    return items


# ---------------------------------------------------------------------------
# Tile loading shared by the product wrappers
# ---------------------------------------------------------------------------

def _load_tiles(
    urls: Sequence[str],
    source: str,
    bounds: Sequence[float],
    bbox_crs: str,
    target_crs: str | CRS | None,
) -> MosaicResult:
    """
    Merge tile URLs into one DEM clipped to ``bounds``.

    Args:
        urls (Sequence[str]): DEM asset URLs.
        source (str): Description of where the URLs came from, for the
            empty-result error message.
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        bbox_crs (str): CRS of ``bounds``.
        target_crs (str | CRS | None): CRS to merge into; ``None`` keeps
            the first tile's CRS.

    Returns:
        MosaicResult: ``(dem, cellsize, transform, crs)``.

    Raises:
        ValueError: If ``urls`` is empty.
        TerraTexture.io.DEMReadError: If any tile cannot be read.
    """
    if not urls:
        raise ValueError(
            f"No tiles in {source} intersect the given bounds "
            f"({tuple(bounds)} in {bbox_crs})."
        )
    logger.info("Loading %d tile(s) from %s", len(urls), source)
    return load_dem_mosaic(
        list(urls), target_crs=target_crs, bounds=bounds, bounds_crs=bbox_crs
    )


# ---------------------------------------------------------------------------
# OpenTopography wrappers
# ---------------------------------------------------------------------------

def opentopography_dem_urls(
    collection: str,
    bounds: Sequence[float],
    bbox_crs: str = "EPSG:4326",
    asset_key: str = "data",
    catalog_url: str = OT_STAC_ROOT,
    max_items: int | None = None,
) -> list[str]:
    """
    Return DEM asset URLs from an OpenTopography collection for an AOI.

    OpenTopography serves a static STAC catalog with no search endpoint.
    Call ``list_stac_collections(OT_STAC_ROOT)`` first to see what exists
    and pick one, then pass its ``id`` or ``href`` here.

    Args:
        collection (str): A collection ``id`` as returned by
            :func:`list_stac_collections`, or a collection URL.
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        bbox_crs (str): CRS of ``bounds``.
        asset_key (str): Item asset holding the DEM. ``"data"`` is a
            common convention but varies by collection; if it's wrong,
            the ``KeyError`` lists the real keys.
        catalog_url (str): Root catalog URL; override only for a mirror.
        max_items (int | None): Cap on the number of items returned.

    Returns:
        list[str]: One HTTPS URL per intersecting item (may be empty).

    Raises:
        ValueError: If no collection matches ``collection`` (the message
            lists close matches), or the arguments are invalid.
        KeyError: If an item has no ``asset_key`` asset.
        STACError: If any request fails.
        ImportError: If ``requests`` is not installed.
    """
    if not str(collection).startswith(("http://", "https://")):
        available = list_stac_collections(catalog_url)
        matches = [c for c in available if c["id"] == collection]
        if not matches:
            ids = [str(c["id"]) for c in available]
            needle = str(collection).lower()
            close = [i for i in ids if needle in i.lower()]
            hint = f"Did you mean: {close[:10]}?" if close else (
                f"{len(ids)} collections available; see "
                "list_stac_collections()."
            )
            raise ValueError(
                f"No OpenTopography collection matches '{collection}'. {hint}"
            )
        collection = str(matches[0]["href"])

    items = stac_collection_items(
        collection, bbox=bounds, bbox_crs=bbox_crs, max_items=max_items
    )
    return [_item_asset_href(item, asset_key) for item in items]


def opentopography_mosaic(
    collection: str,
    bounds: Sequence[float],
    bbox_crs: str = "EPSG:4326",
    asset_key: str = "data",
    catalog_url: str = OT_STAC_ROOT,
    target_crs: str | CRS | None = None,
    max_items: int | None = None,
) -> MosaicResult:
    """
    Fetch and merge OpenTopography DEM tiles for an AOI.

    Only the part of each tile inside ``bounds`` is read, not the full
    tile footprint.

    Args:
        collection (str): Collection ``id`` or URL; see
            :func:`opentopography_dem_urls`.
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        bbox_crs (str): CRS of ``bounds``.
        asset_key (str): Item asset holding the DEM.
        catalog_url (str): Root catalog URL.
        target_crs (str | CRS | None): CRS to merge into; ``None`` keeps
            the first tile's CRS.
        max_items (int | None): Cap on the number of tiles.

    Returns:
        MosaicResult: ``(dem, cellsize, transform, crs)``, as returned by
            :func:`TerraTexture.io.load_dem_mosaic`.

    Raises:
        ValueError: If no tiles intersect ``bounds``, or arguments are
            invalid.
        KeyError: If an item has no ``asset_key`` asset.
        STACError: If a STAC request fails.
        TerraTexture.io.DEMReadError: If a tile cannot be read.
    """
    urls = opentopography_dem_urls(
        collection, bounds, bbox_crs=bbox_crs, asset_key=asset_key,
        catalog_url=catalog_url, max_items=max_items,
    )
    return _load_tiles(
        urls, f"OpenTopography collection '{collection}'",
        bounds, bbox_crs, target_crs,
    )


# ---------------------------------------------------------------------------
# PGC wrappers (ArcticDEM / REMA)
# ---------------------------------------------------------------------------

def _mosaic_collection_id(product: str, version: str, resolution: int) -> str:
    """
    Build a PGC mosaic collection ID.

    Args:
        product (str): ``"arcticdem"`` or ``"rema"``.
        version (str): Product version, e.g. ``"4.1"``.
        resolution (int): Resolution in metres.

    Returns:
        str: e.g. ``"arcticdem-mosaics-v4.1-32m"``.
    """
    return f"{product}-mosaics-v{version}-{resolution}m"


def mosaic_dem_urls(
    product: str,
    bounds: Sequence[float],
    resolution: int = 32,
    version: str | None = None,
    bbox_crs: str = "EPSG:4326",
    api_url: str = PGC_STAC_API,
    asset_key: str = "dem",
    max_items: int | None = None,
) -> list[str]:
    """
    Return PGC mosaic tile URLs intersecting an AOI.

    Shared implementation behind :func:`arcticdem_mosaic_urls` and
    :func:`rema_mosaic_urls`. Call those unless adding a new PGC product
    that follows the ``{product}-mosaics-v{version}-{resolution}m``
    collection naming.

    Args:
        product (str): ``"arcticdem"`` or ``"rema"``.
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        resolution (int): Mosaic resolution in metres: 2, 10 or 32.
        version (str | None): Product version, e.g. ``"4.1"``. ``None``
            uses the current version from ``_MOSAIC_VERSION``.
        bbox_crs (str): CRS of ``bounds``; reprojected to EPSG:4326.
        api_url (str): STAC API base URL; override only for a mirror.
        asset_key (str): Item asset to return. ``"dem"`` is the elevation
            COG; others include ``"hillshade"``, ``"count"``, ``"mad"``
            (median absolute deviation), ``"mindate"`` and ``"maxdate"``.
        max_items (int | None): Cap on the number of tiles.

    Returns:
        list[str]: One HTTPS COG URL per intersecting tile (may be empty).

    Raises:
        ValueError: If ``product`` or ``resolution`` is unsupported, or
            ``bounds`` is invalid.
        KeyError: If a tile has no ``asset_key`` asset.
        STACError: If the STAC search fails. A 404 usually means the
            ``version`` / ``resolution`` collection doesn't exist.
        ImportError: If ``requests`` (or, for a non-WGS84 ``bbox_crs``,
            rasterio) is not installed.
    """
    if product not in _MOSAIC_VERSION:
        raise ValueError(
            f"Unknown PGC product {product!r}; expected one of "
            f"{sorted(_MOSAIC_VERSION)}"
        )
    if resolution not in _MOSAIC_RESOLUTIONS:
        raise ValueError(
            f"resolution={resolution} not published for {product} mosaics; "
            f"PGC provides {_MOSAIC_RESOLUTIONS}"
        )
    if version is None:
        version = _MOSAIC_VERSION[product]

    _validate_bbox(bounds, name="bounds")
    bbox_4326 = _reproject_bbox_to_4326(bounds, bbox_crs)
    collection = _mosaic_collection_id(product, version, resolution)

    try:
        items = stac_search(
            [collection], bbox_4326, api_url=api_url, max_items=max_items
        )
    except STACError as exc:
        if exc.status_code in (400, 404):
            raise STACError(
                f"{exc} -- check that collection '{collection}' exists "
                f"(version={version!r}, resolution={resolution}).",
                url=exc.url,
                status_code=exc.status_code,
            ) from exc
        raise
    return [_item_asset_href(item, asset_key) for item in items]


def arcticdem_mosaic_urls(
    bounds: Sequence[float],
    resolution: int = 32,
    version: str | None = None,
    bbox_crs: str = "EPSG:4326",
    max_items: int | None = None,
) -> list[str]:
    """
    Return ArcticDEM mosaic tile URLs for an AOI (Arctic incl. Greenland).

    Args:
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        resolution (int): 2, 10 or 32 metres.
        version (str | None): Product version; ``None`` means current
            (``"4.1"``).
        bbox_crs (str): CRS of ``bounds``.
        max_items (int | None): Cap on the number of tiles.

    Returns:
        list[str]: One HTTPS COG URL per intersecting tile.

    Raises:
        ValueError: See :func:`mosaic_dem_urls`.
        STACError: If the STAC search fails.
    """
    return mosaic_dem_urls(
        "arcticdem", bounds, resolution=resolution, version=version,
        bbox_crs=bbox_crs, max_items=max_items,
    )


def rema_mosaic_urls(
    bounds: Sequence[float],
    resolution: int = 32,
    version: str | None = None,
    bbox_crs: str = "EPSG:4326",
    max_items: int | None = None,
) -> list[str]:
    """
    Return REMA (Reference Elevation Model of Antarctica) tile URLs.

    Args:
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        resolution (int): 2, 10 or 32 metres.
        version (str | None): Product version; ``None`` means current
            (``"2.0"``).
        bbox_crs (str): CRS of ``bounds``.
        max_items (int | None): Cap on the number of tiles.

    Returns:
        list[str]: One HTTPS COG URL per intersecting tile.

    Raises:
        ValueError: See :func:`mosaic_dem_urls`.
        STACError: If the STAC search fails.
    """
    return mosaic_dem_urls(
        "rema", bounds, resolution=resolution, version=version,
        bbox_crs=bbox_crs, max_items=max_items,
    )


def arcticdem_mosaic(
    bounds: Sequence[float],
    resolution: int = 32,
    version: str | None = None,
    bbox_crs: str = "EPSG:4326",
    target_crs: str | CRS | None = None,
    max_items: int | None = None,
) -> MosaicResult:
    """
    Fetch and merge ArcticDEM mosaic tiles for an AOI from PGC.

    A fully public, no-signup replacement for the old DEMSquad_STAC
    lookup. Only the part of each tile inside ``bounds`` is read.

    Args:
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        resolution (int): 2, 10 or 32 metres.
        version (str | None): Product version; ``None`` means current.
        bbox_crs (str): CRS of ``bounds``.
        target_crs (str | CRS | None): CRS to merge into; ``None`` keeps
            the tiles' own (polar stereographic, EPSG:3413).
        max_items (int | None): Cap on the number of tiles.

    Returns:
        MosaicResult: ``(dem, cellsize, transform, crs)``, as returned by
            :func:`TerraTexture.io.load_dem_mosaic`.

    Raises:
        ValueError: If no tiles intersect ``bounds``, or arguments are
            invalid.
        STACError: If the STAC search fails.
        TerraTexture.io.DEMReadError: If a tile cannot be read.

    Examples:
        >>> dem, cellsize, transform, crs = arcticdem_mosaic(
        ...     bounds=(-51.3, 69.1, -50.9, 69.3), resolution=32,
        ... )
    """
    urls = arcticdem_mosaic_urls(
        bounds, resolution=resolution, version=version,
        bbox_crs=bbox_crs, max_items=max_items,
    )
    return _load_tiles(urls, "ArcticDEM mosaics", bounds, bbox_crs, target_crs)


def rema_mosaic(
    bounds: Sequence[float],
    resolution: int = 32,
    version: str | None = None,
    bbox_crs: str = "EPSG:4326",
    target_crs: str | CRS | None = None,
    max_items: int | None = None,
) -> MosaicResult:
    """
    Fetch and merge REMA mosaic tiles for an AOI from PGC.

    Only the part of each tile inside ``bounds`` is read.

    Args:
        bounds (Sequence[float]): AOI bounds in ``bbox_crs``.
        resolution (int): 2, 10 or 32 metres.
        version (str | None): Product version; ``None`` means current.
        bbox_crs (str): CRS of ``bounds``.
        target_crs (str | CRS | None): CRS to merge into; ``None`` keeps
            the tiles' own (Antarctic polar stereographic, EPSG:3031).
        max_items (int | None): Cap on the number of tiles.

    Returns:
        MosaicResult: ``(dem, cellsize, transform, crs)``.

    Raises:
        ValueError: If no tiles intersect ``bounds``, or arguments are
            invalid.
        STACError: If the STAC search fails.
        TerraTexture.io.DEMReadError: If a tile cannot be read.
    """
    urls = rema_mosaic_urls(
        bounds, resolution=resolution, version=version,
        bbox_crs=bbox_crs, max_items=max_items,
    )
    return _load_tiles(urls, "REMA mosaics", bounds, bbox_crs, target_crs)


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def make_demo_geotiff(
    out_path: str | Path | None = None,
    bounds_4326: Sequence[float] = (-3.25, 54.42, -3.10, 54.53),
) -> str:
    """
    Write the synthetic DEM to a GeoTIFF with a real-world footprint.

    Exists so ``plot_dem_basemap_luminosity_relief()`` has something
    georeferenced to fetch imagery for. The elevation *values* are still
    synthetic; only the footprint is real. Use your own DEM, or
    :func:`arcticdem_mosaic` / :func:`rema_mosaic`, for real work.

    Args:
        out_path (str | Path | None): Output path. ``None`` writes
            ``demo_dem.tif`` in the system temp directory (works on
            macOS, Linux and Windows).
        bounds_4326 (Sequence[float]): ``(west, south, east, north)`` in
            EPSG:4326. Defaults to part of the Lake District, UK.

    Returns:
        str: Path of the written GeoTIFF.

    Raises:
        ValueError: If ``bounds_4326`` is invalid.
        FileNotFoundError: If the output directory does not exist.
        ImportError: If rasterio is not installed.
        OSError: If the file cannot be written.
    """
    west, south, east, north = _validate_bbox(bounds_4326, name="bounds_4326")
    if out_path is None:
        out_path = Path(tempfile.gettempdir()) / "demo_dem.tif"
    out_path = Path(out_path)
    if not out_path.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {out_path.parent}"
        )

    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise ImportError(
            "make_demo_geotiff needs rasterio; install the `raster` extra."
        ) from exc

    dem, _ = load_dem(None)
    transform = from_bounds(west, south, east, north, dem.shape[1], dem.shape[0])

    with rasterio.open(
        out_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype=dem.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    return str(out_path)
