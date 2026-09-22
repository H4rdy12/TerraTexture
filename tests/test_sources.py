"""
Tests for :mod:`TerraTexture.sources` (STAC queries for open DEM data).

No test touches the network. Every HTTP call goes through a
:class:`FakeSTAC` router patched over ``requests.Session.request``,
which returns real ``requests.Response`` objects built from
canned JSON. Because the responses are real, the module's own status,
JSON and content checks run exactly as they would against a server.
Any request to a URL without a route fails the test immediately, so
unexpected calls can't pass silently.

Covers:

- The dynamic-API engine (:func:`stac_search`): request body,
  GET/POST pagination (including the STAC ``merge`` flag), ``max_items``
  and the repeated-link loop guard.
- The static-catalog engine (:func:`list_stac_collections`,
  :func:`stac_collection_items`): relative hrefs, client-side bbox
  filtering, 3-D bboxes, items without a bbox, and pagination.
- Product wrappers: PGC collection naming, OpenTopography collection
  lookup, and that the ``*_mosaic`` functions clip to ``bounds``.
- Error handling: every documented ``Raises`` path, including HTTP
  errors, timeouts, non-JSON bodies and bad arguments.

Dependencies:
    pytest and requests (the module is skipped without requests). A few
    tests additionally need rasterio and skip individually without it.

Examples:
    Run just this file::

        pytest tests/test_sources.py -v
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest

requests = pytest.importorskip("requests")

import TerraTexture.sources as sources  # noqa: E402
from TerraTexture.sources import (  # noqa: E402
    STACError,
    _bbox_intersects,
    _item_asset_href,
    arcticdem_mosaic,
    arcticdem_mosaic_urls,
    list_stac_collections,
    mosaic_dem_urls,
    opentopography_dem_urls,
    opentopography_mosaic,
    rema_mosaic_urls,
    stac_collection_items,
    stac_search,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE = "https://stac.example.com"
_SEARCH_URL = f"{_BASE}/search"
_CATALOG_URL = f"{_BASE}/catalog.json"
_AOI = (0.0, 0.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# Fake HTTP layer
# ---------------------------------------------------------------------------

class FakeSTAC:
    """
    In-memory router standing in for ``requests.Session.request``.

    Routes map ``(method, url)`` to a canned response, or to an exception
    to raise. Every call is recorded in :attr:`calls` so tests can assert
    on request bodies and parameters.

    Attributes:
        calls (list[dict[str, Any]]): One dict per request with keys
            ``method``, ``url``, ``params`` and ``json``.
    """

    def __init__(self) -> None:
        """Create an empty router."""
        self._routes: dict[tuple[str, str], Callable[[], Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def add(
        self,
        method: str,
        url: str,
        payload: Any = None,
        status: int = 200,
        raw: bytes | None = None,
        content_type: str = "application/json",
    ) -> None:
        """
        Register a response for ``method url``.

        Args:
            method (str): ``"GET"`` or ``"POST"``.
            url (str): Exact URL, including any query string.
            payload (Any): Object to serialise as the JSON body.
            status (int): HTTP status code.
            raw (bytes | None): Raw body, used instead of ``payload``.
            content_type (str): ``Content-Type`` header value.

        Returns:
            None
        """
        body = raw if raw is not None else json.dumps(payload).encode()

        def _respond() -> requests.Response:
            response = requests.Response()
            response.status_code = status
            response.reason = HTTPStatus(status).phrase
            response.url = url
            response.encoding = "utf-8"
            response.headers["Content-Type"] = content_type
            response._content = body
            return response

        self._routes[(method.upper(), url)] = _respond

    def fail(self, method: str, url: str, exc: Exception) -> None:
        """
        Make ``method url`` raise ``exc`` instead of responding.

        Args:
            method (str): ``"GET"`` or ``"POST"``.
            url (str): Exact URL.
            exc (Exception): Exception to raise, e.g. ``requests.Timeout``.

        Returns:
            None
        """
        def _raise() -> None:
            raise exc

        self._routes[(method.upper(), url)] = _raise

    def __call__(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,  # noqa: A002 (mirrors requests)
        **kwargs: Any,
    ) -> requests.Response:
        """
        Handle one request, as ``requests.Session.request`` would.

        Args:
            method (str): HTTP method.
            url (str): Request URL.
            params (dict[str, Any] | None): Query parameters.
            json (dict[str, Any] | None): JSON body.
            **kwargs (Any): Other ``requests`` options (ignored).

        Returns:
            requests.Response: The canned response.

        Raises:
            AssertionError: If no route matches -- an unexpected request.
        """
        self.calls.append(
            {"method": method.upper(), "url": url, "params": params, "json": json}
        )
        route = self._routes.get((method.upper(), url))
        if route is None:
            raise AssertionError(f"Unexpected request: {method} {url}")
        return route()


@pytest.fixture
def stac(monkeypatch: pytest.MonkeyPatch) -> FakeSTAC:
    """
    Patch all HTTP traffic through a fresh :class:`FakeSTAC`.

    Also resets the module's cached session, so each test starts clean.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        FakeSTAC: The router; register routes on it before calling code.
    """
    fake = FakeSTAC()

    # A plain function (not the FakeSTAC instance itself) so Python binds
    # it as a method and passes the session in, which we then drop.
    def _request(session: Any, method: str, url: str, **kwargs: Any) -> Any:
        return fake(method, url, **kwargs)

    monkeypatch.setattr(sources, "_session", None)
    monkeypatch.setattr(requests.Session, "request", _request)
    return fake


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _item(
    item_id: str,
    href: str,
    bbox: list[float] | None = None,
    asset_key: str = "dem",
) -> dict[str, Any]:
    """
    Build a minimal STAC Item.

    Args:
        item_id (str): Item ID.
        href (str): URL of the item's single asset.
        bbox (list[float] | None): Item bbox; omitted when ``None``.
        asset_key (str): Key of the asset.

    Returns:
        dict[str, Any]: A GeoJSON Feature dict.
    """
    item: dict[str, Any] = {
        "type": "Feature",
        "id": item_id,
        "assets": {asset_key: {"href": href}},
    }
    if bbox is not None:
        item["bbox"] = bbox
    return item


def _page(
    items: list[dict[str, Any]],
    next_link: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build one page of results (a FeatureCollection).

    Args:
        items (list[dict[str, Any]]): Items on this page.
        next_link (dict[str, Any] | None): ``rel="next"`` link, if any.

    Returns:
        dict[str, Any]: The page document.
    """
    links = [{"rel": "next", **next_link}] if next_link else []
    return {"type": "FeatureCollection", "features": items, "links": links}


def _add_static_catalog(stac: FakeSTAC, items: list[dict[str, Any]]) -> None:
    """
    Register a one-collection static catalog: catalog -> SRTM -> items.

    Args:
        stac (FakeSTAC): Router to register routes on.
        items (list[dict[str, Any]]): Items in the collection.

    Returns:
        None
    """
    stac.add("GET", _CATALOG_URL, {"links": [
        {"rel": "self", "href": _CATALOG_URL},
        {"rel": "child", "href": "./srtm/collection.json", "title": "SRTM GL1"},
        {"rel": "child", "href": "./cop30/collection.json", "title": "COP30"},
    ]})
    stac.add("GET", f"{_BASE}/srtm/collection.json", {"links": [
        {"rel": "items", "href": "./items"},
    ]})
    stac.add("GET", f"{_BASE}/srtm/items", _page(items))


# ---------------------------------------------------------------------------
# stac_search (dynamic API engine)
# ---------------------------------------------------------------------------

def test_stac_search_single_page(stac: FakeSTAC) -> None:
    """One page of results; the request body carries collections and bbox."""
    stac.add("POST", _SEARCH_URL, _page([_item("tile_a", "https://x/a.tif")]))

    items = stac_search(["some-collection"], (0, 0, 1, 1), api_url=_BASE)

    assert [item["id"] for item in items] == ["tile_a"]
    assert len(stac.calls) == 1
    body = stac.calls[0]["json"]
    assert body["collections"] == ["some-collection"]
    assert body["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert body["limit"] == 100


def test_stac_search_adds_datetime_and_extra_params(stac: FakeSTAC) -> None:
    """``datetime`` and ``extra_params`` are merged into the body."""
    stac.add("POST", _SEARCH_URL, _page([]))

    stac_search(
        ["c"], (0, 0, 1, 1), api_url=_BASE,
        datetime="2020-01-01/2021-01-01",
        extra_params={"query": {"eo:cloud_cover": {"lt": 10}}},
    )

    body = stac.calls[0]["json"]
    assert body["datetime"] == "2020-01-01/2021-01-01"
    assert body["query"] == {"eo:cloud_cover": {"lt": 10}}


def test_stac_search_follows_get_pagination(stac: FakeSTAC) -> None:
    """A GET ``next`` link is followed until a page has no ``next``."""
    page2_url = f"{_SEARCH_URL}?page=2"
    stac.add("POST", _SEARCH_URL, _page(
        [_item("tile_a", "https://x/a.tif")],
        {"href": page2_url, "method": "GET"},
    ))
    stac.add("GET", page2_url, _page([_item("tile_b", "https://x/b.tif")]))

    items = stac_search(["c"], (0, 0, 1, 1), api_url=_BASE)

    assert [item["id"] for item in items] == ["tile_a", "tile_b"]
    assert [call["method"] for call in stac.calls] == ["POST", "GET"]


def test_stac_search_post_pagination_merges_body(stac: FakeSTAC) -> None:
    """A ``merge: true`` POST link keeps the original collections and bbox.

    Regression test: the token-only body used to replace the original,
    dropping the search filters on every page after the first.
    """
    page2_url = f"{_SEARCH_URL}/page2"
    stac.add("POST", _SEARCH_URL, _page(
        [_item("tile_a", "https://x/a.tif")],
        {"href": page2_url, "method": "POST", "body": {"token": "abc"},
         "merge": True},
    ))
    stac.add("POST", page2_url, _page([_item("tile_b", "https://x/b.tif")]))

    items = stac_search(["arcticdem"], (0, 0, 1, 1), api_url=_BASE)

    assert [item["id"] for item in items] == ["tile_a", "tile_b"]
    second_body = stac.calls[1]["json"]
    assert second_body["token"] == "abc"
    assert second_body["collections"] == ["arcticdem"]
    assert second_body["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_stac_search_post_pagination_without_merge_uses_link_body(
    stac: FakeSTAC,
) -> None:
    """Without ``merge``, the link's body replaces the original."""
    page2_url = f"{_SEARCH_URL}/page2"
    stac.add("POST", _SEARCH_URL, _page(
        [], {"href": page2_url, "method": "POST", "body": {"token": "abc"}},
    ))
    stac.add("POST", page2_url, _page([]))

    stac_search(["c"], (0, 0, 1, 1), api_url=_BASE)

    assert stac.calls[1]["json"] == {"token": "abc"}


def test_stac_search_respects_max_items(stac: FakeSTAC) -> None:
    """``max_items`` truncates and stops before fetching the next page."""
    stac.add("POST", _SEARCH_URL, _page(
        [_item("a", "https://x/a.tif"), _item("b", "https://x/b.tif")],
        {"href": f"{_SEARCH_URL}?page=2", "method": "GET"},
    ))

    items = stac_search(["c"], (0, 0, 1, 1), api_url=_BASE, max_items=1)

    assert [item["id"] for item in items] == ["a"]
    assert len(stac.calls) == 1


def test_stac_search_stops_on_repeated_next_link(
    stac: FakeSTAC, caplog: pytest.LogCaptureFixture,
) -> None:
    """A ``next`` link pointing at itself stops with a warning, not a hang."""
    loop_url = f"{_SEARCH_URL}?page=loop"
    stac.add("POST", _SEARCH_URL, _page(
        [_item("a", "https://x/a.tif")], {"href": loop_url, "method": "GET"},
    ))
    stac.add("GET", loop_url, _page(
        [_item("b", "https://x/b.tif")], {"href": loop_url, "method": "GET"},
    ))

    with caplog.at_level(logging.WARNING, logger=sources.__name__):
        items = stac_search(["c"], (0, 0, 1, 1), api_url=_BASE)

    assert [item["id"] for item in items] == ["a", "b"]
    assert "repeated 'next' link" in caplog.text


def test_stac_search_allows_antimeridian_bbox(stac: FakeSTAC) -> None:
    """``min_lon > max_lon`` is valid STAC and is sent to the server as-is."""
    stac.add("POST", _SEARCH_URL, _page([]))

    stac_search(["c"], (170, 60, -170, 70), api_url=_BASE)

    assert stac.calls[0]["json"]["bbox"] == [170.0, 60.0, -170.0, 70.0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"collections": "abc"}, "list of IDs"),
        ({"collections": []}, "at least one"),
        ({"bbox": (0, 0, 1)}, "four numbers"),
        ({"bbox": (0, 1, 1, 0)}, "min_y < max_y"),
        ({"bbox": (0, 0, float("inf"), 1)}, "finite"),
        ({"max_items": 0}, "max_items"),
        ({"page_size": -5}, "page_size"),
        ({"timeout": 0}, "timeout"),
    ],
    ids=[
        "string-collections", "empty-collections", "short-bbox",
        "inverted-lat", "infinite-bbox", "zero-max-items",
        "negative-page-size", "zero-timeout",
    ],
)
def test_stac_search_rejects_bad_arguments(
    stac: FakeSTAC, kwargs: dict[str, Any], match: str,
) -> None:
    """Invalid arguments raise ValueError before any request is sent."""
    args: dict[str, Any] = {"collections": ["c"], "bbox": (0, 0, 1, 1)}
    args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        stac_search(api_url=_BASE, **args)

    assert stac.calls == []


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

def test_http_error_raises_stac_error_with_status(stac: FakeSTAC) -> None:
    """A non-2xx response raises STACError carrying the URL and status."""
    stac.add("POST", _SEARCH_URL, {"description": "bad bbox"}, status=400)

    with pytest.raises(STACError, match="HTTP 400") as exc_info:
        stac_search(["c"], (0, 0, 1, 1), api_url=_BASE)

    assert exc_info.value.status_code == 400
    assert exc_info.value.url == _SEARCH_URL
    assert "bad bbox" in str(exc_info.value)


def test_timeout_raises_stac_error(stac: FakeSTAC) -> None:
    """A timeout raises STACError, chaining the original exception."""
    original = requests.Timeout("read timed out")
    stac.fail("GET", _CATALOG_URL, original)

    with pytest.raises(STACError, match="timed out") as exc_info:
        list_stac_collections(_CATALOG_URL, timeout=5)

    assert exc_info.value.__cause__ is original
    assert exc_info.value.status_code is None


def test_connection_error_raises_stac_error(stac: FakeSTAC) -> None:
    """A connection failure raises STACError, not a bare requests error."""
    stac.fail("GET", _CATALOG_URL, requests.ConnectionError("refused"))

    with pytest.raises(STACError, match="refused"):
        list_stac_collections(_CATALOG_URL)


def test_non_json_body_raises_stac_error(stac: FakeSTAC) -> None:
    """An HTML error page served with status 200 raises STACError."""
    stac.add(
        "GET", _CATALOG_URL, raw=b"<html>maintenance</html>",
        content_type="text/html",
    )

    with pytest.raises(STACError, match="did not return JSON"):
        list_stac_collections(_CATALOG_URL)


def test_json_that_is_not_an_object_raises_stac_error(stac: FakeSTAC) -> None:
    """A JSON array (or other non-object) body raises STACError."""
    stac.add("GET", _CATALOG_URL, [1, 2, 3])

    with pytest.raises(STACError, match="expected an object"):
        list_stac_collections(_CATALOG_URL)


def test_stac_error_is_a_runtime_error() -> None:
    """STACError subclasses RuntimeError and keeps its attributes."""
    error = STACError("boom", url="https://x", status_code=503)

    assert isinstance(error, RuntimeError)
    assert (error.url, error.status_code) == ("https://x", 503)


def test_session_retries_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared session retries 429/5xx for both GET and POST."""
    monkeypatch.setattr(sources, "_session", None)

    session = sources._get_session()
    retry = session.get_adapter("https://example.com").max_retries

    assert retry.total == 3
    assert set(retry.status_forcelist) == {429, 500, 502, 503, 504}
    assert {"GET", "POST"} <= set(retry.allowed_methods)
    assert sources._get_session() is session  # cached, not rebuilt


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_item_asset_href_returns_href() -> None:
    """The asset's href is returned."""
    assert _item_asset_href(_item("a", "https://x/a.tif"), "dem") == "https://x/a.tif"


def test_item_asset_href_missing_key_lists_available() -> None:
    """A missing asset raises KeyError naming the key and the real keys."""
    item = _item("tile_a", "https://x/a.tif")

    with pytest.raises(KeyError, match="hillshade") as exc_info:
        _item_asset_href(item, "hillshade")

    assert "['dem']" in str(exc_info.value)


def test_item_asset_href_item_without_assets() -> None:
    """An item with no ``assets`` at all still gives a helpful KeyError."""
    with pytest.raises(KeyError, match="Available assets: \\[\\]"):
        _item_asset_href({"id": "bare"}, "dem")


@pytest.mark.parametrize(
    ("item_bbox", "expected"),
    [
        ((0, 0, 2, 2), True),
        ((1, 1, 3, 3), True),
        ((2, 2, 3, 3), True),
        ((5, 5, 6, 6), False),
        ((0, 0, -10, 2, 2, 10), True),
        ((5, 5, 0, 6, 6, 10), False),
    ],
    ids=["contains", "overlaps", "touches-corner", "disjoint", "3d-hit", "3d-miss"],
)
def test_bbox_intersects(item_bbox: tuple[float, ...], expected: bool) -> None:
    """2-D and 3-D item bboxes are compared correctly; touching counts."""
    assert _bbox_intersects(item_bbox, (1.0, 1.0, 2.0, 2.0)) is expected


def test_bbox_intersects_rejects_bad_length() -> None:
    """A bbox with neither 4 nor 6 values raises ValueError."""
    with pytest.raises(ValueError, match="bbox length"):
        _bbox_intersects((0, 0, 1), (0.0, 0.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# list_stac_collections / stac_collection_items (static engine)
# ---------------------------------------------------------------------------

def test_list_stac_collections_follows_child_links(stac: FakeSTAC) -> None:
    """Child links become collections; relative hrefs are resolved."""
    _add_static_catalog(stac, [])

    collections = list_stac_collections(_CATALOG_URL)

    assert [c["id"] for c in collections] == ["SRTM GL1", "COP30"]
    assert collections[0]["href"] == f"{_BASE}/srtm/collection.json"
    assert collections[1]["href"] == f"{_BASE}/cop30/collection.json"


def test_list_stac_collections_untitled_child_uses_href_as_id(
    stac: FakeSTAC,
) -> None:
    """A child link with no title falls back to its href as the ID."""
    stac.add("GET", _CATALOG_URL, {"links": [
        {"rel": "child", "href": "https://x/c.json"},
    ]})

    collections = list_stac_collections(_CATALOG_URL)

    assert collections == [
        {"id": "https://x/c.json", "title": None, "href": "https://x/c.json"}
    ]


def test_list_stac_collections_empty_catalog_warns(
    stac: FakeSTAC, caplog: pytest.LogCaptureFixture,
) -> None:
    """A catalog with no children returns [] and logs a warning."""
    stac.add("GET", _CATALOG_URL, {"links": [{"rel": "self", "href": "x"}]})

    with caplog.at_level(logging.WARNING, logger=sources.__name__):
        assert list_stac_collections(_CATALOG_URL) == []

    assert "no child collections" in caplog.text


def test_stac_collection_items_filters_by_bbox_client_side(
    stac: FakeSTAC,
) -> None:
    """Items outside the bbox are dropped even if the server returns them."""
    _add_static_catalog(stac, [
        _item("inside", "https://x/in.tif", [0, 0, 1, 1]),
        _item("outside", "https://x/out.tif", [10, 10, 11, 11]),
    ])

    items = stac_collection_items(
        {"href": f"{_BASE}/srtm/collection.json"}, bbox=_AOI
    )

    assert [item["id"] for item in items] == ["inside"]
    items_call = stac.calls[-1]
    assert items_call["params"]["bbox"] == "0.0,0.0,2.0,2.0"
    assert items_call["params"]["limit"] == 250


def test_stac_collection_items_keeps_items_without_bbox(
    stac: FakeSTAC, caplog: pytest.LogCaptureFixture,
) -> None:
    """Items with no bbox can't be ruled out, so they're kept and counted."""
    _add_static_catalog(stac, [_item("no-bbox", "https://x/a.tif")])

    with caplog.at_level(logging.WARNING, logger=sources.__name__):
        items = stac_collection_items(f"{_BASE}/srtm/collection.json", bbox=_AOI)

    assert [item["id"] for item in items] == ["no-bbox"]
    assert "1 item(s) had no bbox" in caplog.text


def test_stac_collection_items_skips_malformed_bbox(
    stac: FakeSTAC, caplog: pytest.LogCaptureFixture,
) -> None:
    """An item whose bbox has the wrong length is skipped with a warning."""
    _add_static_catalog(stac, [
        _item("broken", "https://x/a.tif", [0, 0, 1]),
        _item("ok", "https://x/b.tif", [0, 0, 1, 1]),
    ])

    with caplog.at_level(logging.WARNING, logger=sources.__name__):
        items = stac_collection_items(f"{_BASE}/srtm/collection.json", bbox=_AOI)

    assert [item["id"] for item in items] == ["ok"]
    assert "malformed bbox" in caplog.text


def test_stac_collection_items_paginates(stac: FakeSTAC) -> None:
    """``next`` links are followed; their own query string replaces params."""
    page2_url = f"{_BASE}/srtm/items?page=2"
    _add_static_catalog(stac, [])
    stac.add("GET", f"{_BASE}/srtm/items", _page(
        [_item("a", "https://x/a.tif", [0, 0, 1, 1])], {"href": "./items?page=2"},
    ))
    stac.add("GET", page2_url, _page([_item("b", "https://x/b.tif", [0, 0, 1, 1])]))

    items = stac_collection_items(f"{_BASE}/srtm/collection.json", bbox=_AOI)

    assert [item["id"] for item in items] == ["a", "b"]
    assert stac.calls[-1]["url"] == page2_url
    assert stac.calls[-1]["params"] is None


def test_stac_collection_items_respects_max_items(stac: FakeSTAC) -> None:
    """``max_items`` truncates and stops before the next page."""
    _add_static_catalog(stac, [])
    stac.add("GET", f"{_BASE}/srtm/items", _page(
        [_item("a", "https://x/a.tif", [0, 0, 1, 1]),
         _item("b", "https://x/b.tif", [0, 0, 1, 1])],
        {"href": "./items?page=2"},
    ))

    items = stac_collection_items(
        f"{_BASE}/srtm/collection.json", bbox=_AOI, max_items=1
    )

    assert [item["id"] for item in items] == ["a"]


def test_stac_collection_items_stops_on_repeated_next_link(
    stac: FakeSTAC, caplog: pytest.LogCaptureFixture,
) -> None:
    """A self-referencing ``next`` link stops with a warning."""
    loop_url = f"{_BASE}/loop"
    stac.add("GET", loop_url, _page([], {"href": loop_url}))

    with caplog.at_level(logging.WARNING, logger=sources.__name__):
        items = stac_collection_items({"links": [{"rel": "items", "href": loop_url}]})

    assert items == []
    assert "Repeated 'next' link" in caplog.text


def test_stac_collection_items_full_doc_resolves_via_self_link(
    stac: FakeSTAC,
) -> None:
    """A full Collection document resolves relative links via ``self``."""
    stac.add("GET", f"{_BASE}/srtm/items", _page([]))
    doc = {"links": [
        {"rel": "self", "href": f"{_BASE}/srtm/collection.json"},
        {"rel": "items", "href": "./items"},
    ]}

    stac_collection_items(doc)

    assert stac.calls[0]["url"] == f"{_BASE}/srtm/items"


def test_stac_collection_items_no_items_link_raises(stac: FakeSTAC) -> None:
    """A collection without an ``items`` link raises ValueError."""
    with pytest.raises(ValueError, match="no 'items' link"):
        stac_collection_items({"href": "https://x/c.json", "links": []})


@pytest.mark.parametrize(
    ("collection", "error", "match"),
    [
        ({"id": "srtm"}, ValueError, "needs an 'href'"),
        (42, TypeError, "URL string or a dict"),
    ],
    ids=["dict-without-href-or-links", "wrong-type"],
)
def test_stac_collection_items_rejects_bad_collection(
    stac: FakeSTAC, collection: Any, error: type[Exception], match: str,
) -> None:
    """Unusable ``collection`` arguments raise before any request."""
    with pytest.raises(error, match=match):
        stac_collection_items(collection)

    assert stac.calls == []


def test_stac_collection_items_rejects_antimeridian_bbox(stac: FakeSTAC) -> None:
    """Client-side filtering can't handle antimeridian boxes, so reject them."""
    with pytest.raises(ValueError, match="antimeridian"):
        stac_collection_items("https://x/c.json", bbox=(170, 60, -170, 70))


# ---------------------------------------------------------------------------
# OpenTopography wrappers
# ---------------------------------------------------------------------------

def test_opentopography_dem_urls_resolves_collection_by_id(
    stac: FakeSTAC,
) -> None:
    """A collection title is looked up in the catalog, then queried."""
    _add_static_catalog(stac, [
        _item("tile1", "https://x/tile1.tif", [0, 0, 1, 1], asset_key="data"),
    ])

    urls = opentopography_dem_urls("SRTM GL1", _AOI, catalog_url=_CATALOG_URL)

    assert urls == ["https://x/tile1.tif"]


def test_opentopography_dem_urls_accepts_direct_url(stac: FakeSTAC) -> None:
    """A collection URL skips the catalog lookup entirely."""
    _add_static_catalog(stac, [
        _item("tile1", "https://x/tile1.tif", [0, 0, 1, 1], asset_key="data"),
    ])

    opentopography_dem_urls(f"{_BASE}/srtm/collection.json", _AOI)

    assert all(call["url"] != _CATALOG_URL for call in stac.calls)


def test_opentopography_unknown_collection_lists_available(
    stac: FakeSTAC,
) -> None:
    """With no close match and a short catalog, every ID is listed."""
    _add_static_catalog(stac, [])

    with pytest.raises(ValueError, match="Available: .*SRTM GL1.*COP30"):
        opentopography_dem_urls(
            "NoSuchDataset", _AOI, catalog_url=_CATALOG_URL
        )

    catalog_fetches = [c for c in stac.calls if c["url"] == _CATALOG_URL]
    assert len(catalog_fetches) == 1  # regression: used to fetch twice


def test_opentopography_unknown_collection_suggests_close_match(
    stac: FakeSTAC,
) -> None:
    """A partial, case-insensitive match is suggested."""
    _add_static_catalog(stac, [])

    with pytest.raises(ValueError, match="Did you mean: \\['COP30'\\]"):
        opentopography_dem_urls("cop", _AOI, catalog_url=_CATALOG_URL)


def test_opentopography_unknown_collection_large_catalog_gives_count(
    stac: FakeSTAC,
) -> None:
    """Catalogs too big to list in an error report a count instead."""
    children = [
        {"rel": "child", "href": f"./c{i}.json", "title": f"DATASET_{i}"}
        for i in range(sources._MAX_LISTED_COLLECTIONS + 5)
    ]
    stac.add("GET", _CATALOG_URL, {"links": children})

    with pytest.raises(ValueError, match="25 collections available"):
        opentopography_dem_urls("zzz", _AOI, catalog_url=_CATALOG_URL)


def test_opentopography_wrong_asset_key_lists_real_keys(stac: FakeSTAC) -> None:
    """A wrong ``asset_key`` fails with the keys that do exist."""
    _add_static_catalog(stac, [
        _item("tile1", "https://x/tile1.tif", [0, 0, 1, 1], asset_key="data"),
    ])

    with pytest.raises(KeyError, match="\\['data'\\]"):
        opentopography_dem_urls(
            "SRTM GL1", _AOI, catalog_url=_CATALOG_URL, asset_key="elevation"
        )


# ---------------------------------------------------------------------------
# PGC wrappers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("func", "resolution", "collection"),
    [
        (arcticdem_mosaic_urls, 2, "arcticdem-mosaics-v4.1-2m"),
        (arcticdem_mosaic_urls, 32, "arcticdem-mosaics-v4.1-32m"),
        (rema_mosaic_urls, 10, "rema-mosaics-v2.0-10m"),
    ],
    ids=["arcticdem-2m", "arcticdem-32m", "rema-10m"],
)
def test_product_urls_build_correct_collection(
    stac: FakeSTAC,
    monkeypatch: pytest.MonkeyPatch,
    func: Callable[..., list[str]],
    resolution: int,
    collection: str,
) -> None:
    """Product wrappers search PGC's collection for the right version."""
    pgc_search = f"{sources.PGC_STAC_API}/search"
    stac.add("POST", pgc_search, _page([_item("t", "https://x/dem.tif")]))

    urls = func((0, 0, 1, 1), resolution=resolution)

    assert urls == ["https://x/dem.tif"]
    assert stac.calls[0]["json"]["collections"] == [collection]


def test_mosaic_dem_urls_custom_version(stac: FakeSTAC) -> None:
    """An explicit ``version`` overrides the default."""
    stac.add("POST", _SEARCH_URL, _page([]))

    mosaic_dem_urls("arcticdem", (0, 0, 1, 1), version="5.0", api_url=_BASE)

    assert stac.calls[0]["json"]["collections"] == ["arcticdem-mosaics-v5.0-32m"]


def test_mosaic_dem_urls_other_asset_key(stac: FakeSTAC) -> None:
    """``asset_key`` selects a different asset, e.g. the hillshade."""
    item = _item("t", "https://x/dem.tif")
    item["assets"]["hillshade"] = {"href": "https://x/hs.tif"}
    stac.add("POST", _SEARCH_URL, _page([item]))

    urls = mosaic_dem_urls(
        "arcticdem", (0, 0, 1, 1), api_url=_BASE, asset_key="hillshade"
    )

    assert urls == ["https://x/hs.tif"]


def test_mosaic_dem_urls_rejects_bad_resolution(stac: FakeSTAC) -> None:
    """Unpublished resolutions raise ValueError listing the valid ones."""
    with pytest.raises(ValueError, match="resolution=7.*\\(2, 10, 32\\)"):
        mosaic_dem_urls("arcticdem", (0, 0, 1, 1), resolution=7)

    assert stac.calls == []


def test_mosaic_dem_urls_rejects_unknown_product(stac: FakeSTAC) -> None:
    """An unknown product raises ValueError, not a bare KeyError."""
    with pytest.raises(ValueError, match="Unknown PGC product 'greenland'"):
        mosaic_dem_urls("greenland", (0, 0, 1, 1))


def test_mosaic_dem_urls_404_hints_at_missing_collection(
    stac: FakeSTAC,
) -> None:
    """A 404 from PGC points at the version/resolution collection."""
    stac.add("POST", _SEARCH_URL, {"description": "not found"}, status=404)

    with pytest.raises(STACError, match="arcticdem-mosaics-v9.9-32m") as exc_info:
        mosaic_dem_urls("arcticdem", (0, 0, 1, 1), version="9.9", api_url=_BASE)

    assert exc_info.value.status_code == 404
    assert "version='9.9'" in str(exc_info.value)


def test_mosaic_dem_urls_server_error_has_no_collection_hint(
    stac: FakeSTAC,
) -> None:
    """A 5xx is a server problem, so no misleading collection hint."""
    stac.add("POST", _SEARCH_URL, {"description": "down"}, status=503)

    with pytest.raises(STACError) as exc_info:
        mosaic_dem_urls("arcticdem", (0, 0, 1, 1), api_url=_BASE)

    assert exc_info.value.status_code == 503
    assert "check that collection" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# *_mosaic loaders (tile loading is stubbed out)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Replace ``load_dem_mosaic`` in :mod:`sources` with a recorder.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        dict[str, Any]: Filled with ``paths`` and ``kwargs`` once called.
    """
    captured: dict[str, Any] = {}

    def _fake(paths: list[str], **kwargs: Any) -> tuple[str, float, str, str]:
        captured["paths"] = paths
        captured["kwargs"] = kwargs
        return "DEM", 32.0, "TRANSFORM", "EPSG:3413"

    monkeypatch.setattr(sources, "load_dem_mosaic", _fake)
    return captured


def test_arcticdem_mosaic_clips_to_bounds(
    stac: FakeSTAC, fake_loader: dict[str, Any],
) -> None:
    """Tile URLs are loaded with the AOI bounds and CRS passed through."""
    pgc_search = f"{sources.PGC_STAC_API}/search"
    stac.add("POST", pgc_search, _page([_item("t", "https://x/dem.tif")]))

    result = arcticdem_mosaic((0, 0, 1, 1), target_crs="EPSG:3413")

    assert result == ("DEM", 32.0, "TRANSFORM", "EPSG:3413")
    assert fake_loader["paths"] == ["https://x/dem.tif"]
    assert fake_loader["kwargs"] == {
        "target_crs": "EPSG:3413",
        "bounds": (0, 0, 1, 1),
        "bounds_crs": "EPSG:4326",
    }


def test_arcticdem_mosaic_no_tiles_raises(
    stac: FakeSTAC, fake_loader: dict[str, Any],
) -> None:
    """No intersecting tiles raises ValueError naming the product."""
    stac.add("POST", f"{sources.PGC_STAC_API}/search", _page([]))

    with pytest.raises(ValueError, match="No tiles in ArcticDEM mosaics"):
        arcticdem_mosaic((0, 0, 1, 1))

    assert fake_loader == {}


def test_opentopography_mosaic_clips_to_bounds(
    stac: FakeSTAC, fake_loader: dict[str, Any],
) -> None:
    """Regression: OpenTopography tiles are now clipped like PGC tiles."""
    _add_static_catalog(stac, [
        _item("tile1", "https://x/tile1.tif", [0, 0, 1, 1], asset_key="data"),
    ])

    opentopography_mosaic("SRTM GL1", _AOI, catalog_url=_CATALOG_URL)

    assert fake_loader["kwargs"]["bounds"] == _AOI


def test_opentopography_mosaic_no_tiles_raises(
    stac: FakeSTAC, fake_loader: dict[str, Any],
) -> None:
    """No intersecting tiles raises ValueError naming the collection."""
    _add_static_catalog(stac, [
        _item("far", "https://x/far.tif", [50, 50, 51, 51], asset_key="data"),
    ])

    with pytest.raises(ValueError, match="collection 'SRTM GL1'"):
        opentopography_mosaic("SRTM GL1", _AOI, catalog_url=_CATALOG_URL)


# ---------------------------------------------------------------------------
# Tests that need rasterio
# ---------------------------------------------------------------------------

def test_reproject_bbox_from_projected_crs() -> None:
    """Polar stereographic bounds are reprojected to lon/lat for STAC."""
    pytest.importorskip("rasterio")

    west, south, east, north = sources._reproject_bbox_to_4326(
        (-200_000, -2_200_000, -190_000, -2_190_000), "EPSG:3413"
    )

    assert -180 <= west < east <= 180
    assert 60 < south < north < 90


def test_reproject_bbox_wgs84_passthrough() -> None:
    """WGS84 aliases skip reprojection (and never import rasterio)."""
    assert sources._reproject_bbox_to_4326((1, 2, 3, 4), "wgs84") == (1, 2, 3, 4)


def test_reproject_bbox_invalid_crs_raises_value_error() -> None:
    """An unknown CRS raises ValueError, not a raw rasterio CRSError."""
    pytest.importorskip("rasterio")

    with pytest.raises(ValueError, match="invalid CRS"):
        sources._reproject_bbox_to_4326((0, 0, 1, 1), "EPSG:999999")


def test_make_demo_geotiff_writes_file(tmp_path: Path) -> None:
    """The demo GeoTIFF is written with the requested footprint."""
    rasterio = pytest.importorskip("rasterio")
    out = tmp_path / "demo.tif"

    result = sources.make_demo_geotiff(out, bounds_4326=(-3.2, 54.4, -3.1, 54.5))

    assert result == str(out)
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 4326
        assert src.bounds.left == pytest.approx(-3.2)
        assert src.bounds.top == pytest.approx(54.5)


def test_make_demo_geotiff_missing_directory_raises(tmp_path: Path) -> None:
    """A missing output directory raises FileNotFoundError up front."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sources.make_demo_geotiff(tmp_path / "nope" / "demo.tif")


def test_make_demo_geotiff_rejects_bad_bounds(tmp_path: Path) -> None:
    """Inverted bounds raise ValueError before anything is written."""
    with pytest.raises(ValueError, match="bounds_4326"):
        sources.make_demo_geotiff(tmp_path / "x.tif", bounds_4326=(1, 1, 0, 0))
